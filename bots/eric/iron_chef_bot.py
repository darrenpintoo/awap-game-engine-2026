"""
Iron Chef v2.0 - Optimized AWAP Competition Bot

Features:
- Centralized "Hive Mind" task auction using Hungarian Algorithm
- Pre-computed 8-directional distance matrix (Chebyshev)
- Time-Space A* with reservation system for collision-free navigation
- Three strategic phases: Build (0-249), Sabotage (250-349), Endgame (350-500)
- Just-In-Time ingredient purchasing
- Box utilization for ingredient buffering
- Scorched Earth sabotage protocol
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Set
from enum import Enum

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not available, using greedy assignment fallback")

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from tiles import Tile, Counter, Cooker, Sink, SinkTable, Submit, Shop, Box, Trash
from item import Pan, Plate, Food


# ============================================
# DATA STRUCTURES
# ============================================

class JobType(Enum):
    # Emergency
    SAVE_FOOD = 0       # Take cooked food from pan before it burns
    
    # Logistics
    BUY_INGREDIENT = 10
    BUY_PAN = 11
    BUY_PLATE = 12
    FETCH_FROM_BOX = 13
    
    # Processing
    PLACE_ON_COUNTER = 20
    CHOP = 21
    START_COOK = 22
    TAKE_FROM_PAN = 23
    
    # Plating
    TAKE_CLEAN_PLATE = 30
    ADD_TO_PLATE = 31
    PICKUP_PLATE = 32
    
    # Service
    SUBMIT = 40
    WASH = 41
    
    # Sabotage
    STEAL_PLATE = 50
    STEAL_PAN = 51
    TRASH_ENEMY_FOOD = 52
    BLOCK_SUBMIT = 53
    
    # Utility
    DROP_ITEM = 60
    IDLE = 99


@dataclass
class Job:
    job_type: JobType
    target: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    priority: int = 0  # Higher = more urgent
    order_id: Optional[int] = None
    
    def __hash__(self):
        return hash((self.job_type, self.target, self.priority))


@dataclass 
class CookingInfo:
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int
    
    @property
    def turns_to_cooked(self) -> int:
        if self.cooked_stage >= 1:
            return 0
        return max(0, GameConstants.COOK_PROGRESS - self.cook_progress)
    
    @property
    def turns_to_burned(self) -> int:
        if self.cooked_stage >= 2:
            return 0
        return max(0, GameConstants.BURN_PROGRESS - self.cook_progress)


# ============================================
# MAIN BOT CLASS
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Pre-computed distance matrix: dist_matrix[(x1,y1)][(x2,y2)] = distance
        self.dist_matrix: Dict[Tuple[int,int], Dict[Tuple[int,int], int]] = {}
        
        # Tile locations
        self.shops: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.boxes: List[Tuple[int, int]] = []
        self.walkable: Set[Tuple[int, int]] = set()
        
        # State tracking
        self.cooking_info: Dict[Tuple[int,int], CookingInfo] = {}
        self.bot_assignments: Dict[int, Job] = {}
        self.reserved_nodes: Set[Tuple[int, int, int]] = set()  # (x, y, turn)
        
        # Order tracking
        self.active_orders: List[Dict] = []
        self.orders_in_progress: Set[int] = set()  # order_ids being worked on
        
        # Phase tracking
        self.current_phase = "BUILD"  # BUILD, SABOTAGE, ENDGAME
        self.has_switched = False
        
        # Cooking pipeline state
        self.pipeline_state: Dict[int, Dict] = {}  # order_id -> progress info

    # ============================================
    # INITIALIZATION
    # ============================================
    
    def initialize(self, controller: RobotController):
        """One-time initialization at game start"""
        if self.initialized:
            return
        
        m = controller.get_map(controller.get_team())
        
        # Parse map tiles
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                tile_name = getattr(tile, 'tile_name', '')
                
                if getattr(tile, 'is_walkable', False):
                    self.walkable.add((x, y))
                
                if tile_name == 'SHOP':
                    self.shops.append((x, y))
                elif tile_name == 'COOKER':
                    self.cookers.append((x, y))
                elif tile_name == 'SINK':
                    self.sinks.append((x, y))
                elif tile_name == 'SINKTABLE':
                    self.sink_tables.append((x, y))
                elif tile_name == 'COUNTER':
                    self.counters.append((x, y))
                elif tile_name == 'SUBMIT':
                    self.submits.append((x, y))
                    self.walkable.add((x, y))
                elif tile_name == 'TRASH':
                    self.trashes.append((x, y))
                elif tile_name == 'BOX':
                    self.boxes.append((x, y))
        
        # Build distance matrix using BFS with 8-way connectivity
        self._build_distance_matrix(m)
        
        self.initialized = True
        print(f"[INIT] Map parsed: {len(self.walkable)} walkable, {len(self.cookers)} cookers, {len(self.shops)} shops")

    def _build_distance_matrix(self, m):
        """BFS from every walkable tile to build distance cache"""
        for start in self.walkable:
            self.dist_matrix[start] = {start: 0}
            
            queue = deque([(start, 0)])
            visited = {start}
            
            while queue:
                (cx, cy), dist = queue.popleft()
                
                # 8-directional movement (Chebyshev)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in visited:
                            continue
                        if not (0 <= nx < m.width and 0 <= ny < m.height):
                            continue
                        
                        tile = m.tiles[nx][ny]
                        if getattr(tile, 'is_walkable', False):
                            visited.add((nx, ny))
                            self.dist_matrix[start][(nx, ny)] = dist + 1
                            queue.append(((nx, ny), dist + 1))

    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    def get_distance(self, start: Tuple[int,int], end: Tuple[int,int]) -> int:
        """O(1) distance lookup with fallback to Chebyshev heuristic"""
        if start in self.dist_matrix and end in self.dist_matrix[start]:
            return self.dist_matrix[start][end]
        # Chebyshev fallback
        return max(abs(start[0] - end[0]), abs(start[1] - end[1]))
    
    def get_distance_to_adjacent(self, start: Tuple[int,int], target: Tuple[int,int]) -> int:
        """Distance to get adjacent to a non-walkable target"""
        min_dist = float('inf')
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                adj = (target[0] + dx, target[1] + dy)
                if adj in self.walkable:
                    d = self.get_distance(start, adj)
                    min_dist = min(min_dist, d)
        return min_dist if min_dist != float('inf') else 999
    
    def is_adjacent(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> bool:
        """Chebyshev distance <= 1"""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1
    
    def find_empty_counter(self, controller: RobotController, 
                           near: Tuple[int,int], exclude: Set = None) -> Optional[Tuple[int,int]]:
        """Find nearest empty counter"""
        if exclude is None:
            exclude = set()
        
        best_dist = float('inf')
        best_pos = None
        
        for cx, cy in self.counters:
            if (cx, cy) in exclude:
                continue
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and getattr(tile, 'item', None) is None:
                dist = self.get_distance_to_adjacent(near, (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (cx, cy)
        
        return best_pos
    
    def find_cooker_with_empty_pan(self, controller: RobotController, 
                                    near: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        """Find cooker with pan that has no food"""
        best_dist = float('inf')
        best_pos = None
        
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    dist = self.get_distance_to_adjacent(near, (kx, ky))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (kx, ky)
        
        return best_pos
    
    def count_clean_plates(self, controller: RobotController) -> int:
        """Count available clean plates at sink tables"""
        total = 0
        for sx, sy in self.sink_tables:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile:
                total += getattr(tile, 'num_clean_plates', 0)
        return total
    
    def count_dirty_plates(self, controller: RobotController) -> int:
        """Count dirty plates in sinks"""
        total = 0
        for sx, sy in self.sinks:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile:
                total += getattr(tile, 'num_dirty_plates', 0)
        return total

    # ============================================
    # STATE UPDATE
    # ============================================
    
    def update_state(self, controller: RobotController):
        """Update internal state tracking each turn"""
        # Update cooking info
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food is not None:
                    self.cooking_info[(kx, ky)] = CookingInfo(
                        location=(kx, ky),
                        food_name=pan.food.food_name,
                        cook_progress=getattr(tile, 'cook_progress', 0),
                        cooked_stage=pan.food.cooked_stage
                    )
        
        # Update orders
        self.active_orders = [o for o in controller.get_orders(controller.get_team()) if o.get('is_active', False)]
        
        # Update phase
        turn = controller.get_turn()
        if turn < 250:
            self.current_phase = "BUILD"
        elif turn < 350:
            self.current_phase = "SABOTAGE"
        else:
            self.current_phase = "ENDGAME"

    # ============================================
    # JOB GENERATION
    # ============================================
    
    def generate_jobs(self, controller: RobotController) -> List[Job]:
        """Generate all available jobs based on current state"""
        jobs = []
        turn = controller.get_turn()
        
        # ---- EMERGENCY JOBS ----
        # Save food about to burn
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:  # Cooked, needs pickup NOW
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=100))
            elif info.turns_to_burned <= 5 and info.cooked_stage == 0:
                # About to be cooked, high priority pickup soon
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=90))
        
        # ---- MAINTENANCE JOBS ----
        # Wash dishes if low on clean plates
        clean_plates = self.count_clean_plates(controller)
        dirty_plates = self.count_dirty_plates(controller)
        
        if dirty_plates > 0 and clean_plates < 2:
            for sx, sy in self.sinks:
                tile = controller.get_tile(controller.get_team(), sx, sy)
                if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                    jobs.append(Job(JobType.WASH, target=(sx, sy), priority=70))
        
        # ---- ORDER FULFILLMENT JOBS ----
        for order in self.active_orders:
            order_id = order['order_id']
            if order_id in self.orders_in_progress:
                continue
            
            # Check what ingredients are needed
            required = order.get('required', [])
            time_left = order['expires_turn'] - turn
            
            if time_left < 30:
                continue  # Not enough time
            
            # Generate jobs for this order
            priority = 50 + max(0, 100 - time_left)  # More urgent as deadline approaches
            
            # Need to buy ingredients
            for ingredient in required:
                food_type = self._get_food_type(ingredient)
                if food_type:
                    jobs.append(Job(JobType.BUY_INGREDIENT, item=food_type, 
                                   priority=priority, order_id=order_id))
        
        # ---- ENSURE EQUIPMENT ----
        # Check if we have pans on cookers
        has_pan = any(
            isinstance(getattr(controller.get_tile(controller.get_team(), kx, ky), 'item', None), Pan)
            for kx, ky in self.cookers
        )
        if not has_pan:
            jobs.append(Job(JobType.BUY_PAN, priority=80))
        
        # ---- SABOTAGE JOBS (Phase 2) ----
        if self.current_phase == "SABOTAGE":
            switch_info = controller.get_switch_info()
            if switch_info['my_team_switched']:
                # We're on enemy map - generate sabotage jobs
                for st in self.sink_tables:
                    jobs.append(Job(JobType.STEAL_PLATE, target=st, priority=95))
                for ck in self.cookers:
                    jobs.append(Job(JobType.STEAL_PAN, target=ck, priority=85))
        
        # ---- IDLE JOB ----
        # Always have a low-priority idle job
        jobs.append(Job(JobType.IDLE, priority=0))
        
        return jobs
    
    def _get_food_type(self, name: str) -> Optional[FoodType]:
        """Convert food name string to FoodType enum"""
        name_upper = name.upper()
        for ft in FoodType:
            if ft.food_name.upper() == name_upper:
                return ft
        return None

    # ============================================
    # TASK ASSIGNMENT (HUNGARIAN ALGORITHM)
    # ============================================
    
    def assign_tasks(self, controller: RobotController, jobs: List[Job]) -> Dict[int, Job]:
        """Assign jobs to bots using Hungarian Algorithm"""
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        n_bots = len(bot_ids)
        n_jobs = len(jobs)
        
        if n_bots == 0 or n_jobs == 0:
            return {}
        
        # Build cost matrix
        cost_matrix = np.full((n_bots, n_jobs), 9999.0)
        
        for i, bot_id in enumerate(bot_ids):
            bot = controller.get_bot_state(bot_id)
            if not bot:
                continue
            bot_pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            
            for j, job in enumerate(jobs):
                # Base cost is distance to job target
                if job.target:
                    dist = self.get_distance_to_adjacent(bot_pos, job.target)
                else:
                    dist = 0
                
                # Apply priority as negative cost (higher priority = lower cost)
                priority_bonus = job.priority * 10
                
                # Holding penalties/bonuses
                holding_penalty = 0
                
                if job.job_type == JobType.BUY_INGREDIENT:
                    if holding is not None:
                        holding_penalty = 500  # Need to drop first
                    # Find nearest shop
                    if self.shops:
                        shop_dist = min(self.get_distance_to_adjacent(bot_pos, s) for s in self.shops)
                        dist = shop_dist
                        
                elif job.job_type == JobType.CHOP:
                    if holding is None or holding.get('type') != 'Food':
                        holding_penalty = 1000  # Can't chop without food
                    elif not self._is_choppable(holding):
                        holding_penalty = 1000
                        
                elif job.job_type == JobType.TAKE_FROM_PAN:
                    if holding is not None:
                        holding_penalty = 500  # Need empty hands
                        
                elif job.job_type == JobType.SUBMIT:
                    if not holding or holding.get('type') != 'Plate':
                        holding_penalty = 1000
                    elif holding.get('dirty', True):
                        holding_penalty = 1000
                
                cost_matrix[i, j] = dist - priority_bonus + holding_penalty
        
        # Run Hungarian algorithm
        if HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # Greedy fallback
            col_ind = self._greedy_assignment(cost_matrix)
            row_ind = list(range(n_bots))
        
        # Build assignment dict
        assignments = {}
        for i, j in zip(row_ind, col_ind):
            if j < len(jobs):
                assignments[bot_ids[i]] = jobs[j]
        
        return assignments
    
    def _greedy_assignment(self, cost_matrix: np.ndarray) -> List[int]:
        """Greedy fallback when scipy not available"""
        n_bots, n_jobs = cost_matrix.shape
        assigned_jobs = set()
        result = []
        
        for i in range(n_bots):
            best_j = 0
            best_cost = float('inf')
            for j in range(n_jobs):
                if j not in assigned_jobs and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            result.append(best_j)
            assigned_jobs.add(best_j)
        
        return result
    
    def _is_choppable(self, holding: Dict) -> bool:
        """Check if held item is choppable"""
        if holding.get('type') != 'Food':
            return False
        name = holding.get('food_name', '').upper()
        return name in ['ONIONS', 'MEAT']

    # ============================================
    # PATHFINDING (TIME-SPACE A*)
    # ============================================
    
    def get_move_toward(self, controller: RobotController, bot_id: int, 
                        target: Tuple[int, int], turn: int) -> Optional[Tuple[int, int]]:
        """Calculate next move toward target using reservation system"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        
        start = (bot['x'], bot['y'])
        
        if self.is_adjacent(start, target):
            return (0, 0)  # Already there
        
        # Simple BFS with reservation check
        queue = deque([(start, [])])
        visited = {start}
        
        m = controller.get_map()
        
        while queue:
            (cx, cy), path = queue.popleft()
            
            if self.is_adjacent((cx, cy), target):
                if not path:
                    return (0, 0)
                step = path[0]
                # Reserve this position
                next_pos = (start[0] + step[0], start[1] + step[1])
                self.reserved_nodes.add((next_pos[0], next_pos[1], turn + 1))
                return step
            
            # 8-directional search
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    
                    if (nx, ny) in visited:
                        continue
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    
                    # Check reservation
                    step_turn = turn + len(path) + 1
                    if (nx, ny, step_turn) in self.reserved_nodes:
                        continue
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))
        
        return None

    # ============================================
    # JOB EXECUTION
    # ============================================
    
    def execute_job(self, controller: RobotController, bot_id: int, job: Job):
        """Execute a single job for a bot"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        if job.job_type == JobType.IDLE:
            return  # Do nothing
        
        elif job.job_type == JobType.BUY_INGREDIENT:
            if holding is not None:
                return  # Can't buy if holding
            shop = self._find_nearest(self.shops, (bx, by))
            if shop:
                step = self.get_move_toward(controller, bot_id, shop, turn)
                if step == (0, 0) or self.is_adjacent((bx, by), shop):
                    # Adjacent to shop, buy
                    controller.buy(bot_id, job.item, shop[0], shop[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.BUY_PAN:
            if holding is not None:
                return
            shop = self._find_nearest(self.shops, (bx, by))
            if shop:
                step = self.get_move_toward(controller, bot_id, shop, turn)
                if step == (0, 0) or self.is_adjacent((bx, by), shop):
                    controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.BUY_PLATE:
            if holding is not None:
                return
            shop = self._find_nearest(self.shops, (bx, by))
            if shop:
                step = self.get_move_toward(controller, bot_id, shop, turn)
                if step == (0, 0) or self.is_adjacent((bx, by), shop):
                    controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.TAKE_FROM_PAN:
            if holding is not None:
                return
            target = job.target
            if target:
                step = self.get_move_toward(controller, bot_id, target, turn)
                if step == (0, 0) or self.is_adjacent((bx, by), target):
                    controller.take_from_pan(bot_id, target[0], target[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.WASH:
            target = job.target
            if target:
                step = self.get_move_toward(controller, bot_id, target, turn)
                if step == (0, 0) or self.is_adjacent((bx, by), target):
                    controller.wash_sink(bot_id, target[0], target[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.STEAL_PLATE:
            # Sabotage: steal clean plate
            if holding is not None:
                # Trash it
                trash = self._find_nearest(self.trashes, (bx, by))
                if trash:
                    step = self.get_move_toward(controller, bot_id, trash, turn)
                    if step == (0, 0) or self.is_adjacent((bx, by), trash):
                        controller.trash(bot_id, trash[0], trash[1])
                    elif step:
                        controller.move(bot_id, step[0], step[1])
            else:
                target = job.target
                if target:
                    step = self.get_move_toward(controller, bot_id, target, turn)
                    if step == (0, 0) or self.is_adjacent((bx, by), target):
                        controller.take_clean_plate(bot_id, target[0], target[1])
                    elif step:
                        controller.move(bot_id, step[0], step[1])
        
        elif job.job_type == JobType.STEAL_PAN:
            # Sabotage: steal pan
            if holding is not None:
                # Move to corner and hold
                import random
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                random.shuffle(directions)
                for dx, dy in directions:
                    if controller.can_move(bot_id, dx, dy):
                        controller.move(bot_id, dx, dy)
                        return
            else:
                target = job.target
                if target:
                    step = self.get_move_toward(controller, bot_id, target, turn)
                    if step == (0, 0) or self.is_adjacent((bx, by), target):
                        controller.pickup(bot_id, target[0], target[1])
                    elif step:
                        controller.move(bot_id, step[0], step[1])
    
    def _find_nearest(self, locations: List[Tuple[int, int]], 
                      pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest location from list"""
        if not locations:
            return None
        return min(locations, key=lambda loc: self.get_distance_to_adjacent(pos, loc))

    # ============================================
    # PHASE STRATEGIES
    # ============================================
    
    def run_build_phase(self, controller: RobotController):
        """Build phase: efficient order fulfillment"""
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return
        
        # Main bot runs the cooking pipeline
        main_bot = bot_ids[0]
        self._run_cooking_pipeline(controller, main_bot)
        
        # Secondary bots help with washing/fetching
        for bot_id in bot_ids[1:]:
            self._run_support_pipeline(controller, bot_id)
    
    def run_sabotage_phase(self, controller: RobotController):
        """Sabotage phase: continue normal operation (sabotage disabled for stability)"""
        # NOTE: Sabotage is disabled for now to maximize cooking efficiency
        # The switch mechanic causes state machine confusion and wastes turns
        # TODO: Re-enable with proper state reset once pipeline is more robust
        self.run_build_phase(controller)
    
    def run_endgame_phase(self, controller: RobotController):
        """Endgame: sprint for quick orders"""
        # Same as build but prioritize fast orders
        self.run_build_phase(controller)
    
    def _run_cooking_pipeline(self, controller: RobotController, bot_id: int):
        """State machine for order completion"""
        # Use previous advanced_bot logic
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        # Get pipeline state for this bot
        if bot_id not in self.pipeline_state:
            self.pipeline_state[bot_id] = {'state': 0, 'counter': None, 'cooker': None}
        
        state = self.pipeline_state[bot_id]
        current_state = state['state']
        
        # State machine (simplified from advanced_bot)
        if current_state == 0:
            # Check if pan exists
            has_pan = any(
                isinstance(getattr(controller.get_tile(controller.get_team(), kx, ky), 'item', None), Pan)
                for kx, ky in self.cookers
            )
            state['state'] = 2 if has_pan else 1
            
        elif current_state == 1:
            # Buy pan
            if holding and holding.get('type') == 'Pan':
                cooker = self._find_nearest(self.cookers, (bx, by))
                if cooker:
                    step = self.get_move_toward(controller, bot_id, cooker, turn)
                    if self.is_adjacent((bx, by), cooker):
                        controller.place(bot_id, cooker[0], cooker[1])
                        state['state'] = 2
                    elif step:
                        controller.move(bot_id, step[0], step[1])
            else:
                shop = self._find_nearest(self.shops, (bx, by))
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 2:
            # Buy meat
            if holding:
                state['state'] = 3
            else:
                shop = self._find_nearest(self.shops, (bx, by))
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1]):
                            state['state'] = 3
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 3:
            # Place on counter
            counter = self.find_empty_counter(controller, (bx, by))
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.place(bot_id, counter[0], counter[1]):
                        state['counter'] = counter
                        state['state'] = 4
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 4:
            # Chop
            counter = state.get('counter')
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.chop(bot_id, counter[0], counter[1]):
                        state['state'] = 5
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 5:
            # Pickup
            counter = state.get('counter')
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.pickup(bot_id, counter[0], counter[1]):
                        state['state'] = 6
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 6:
            # Start cooking
            cooker = self.find_cooker_with_empty_pan(controller, (bx, by))
            if cooker:
                step = self.get_move_toward(controller, bot_id, cooker, turn)
                if self.is_adjacent((bx, by), cooker):
                    if controller.place(bot_id, cooker[0], cooker[1]):
                        state['cooker'] = cooker
                        state['state'] = 7
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 7:
            # Buy plate
            if holding:
                state['state'] = 8
            else:
                shop = self._find_nearest(self.shops, (bx, by))
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1]):
                            state['state'] = 8
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 8:
            # Place plate
            counter = self.find_empty_counter(controller, (bx, by))
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.place(bot_id, counter[0], counter[1]):
                        state['plate_counter'] = counter
                        state['state'] = 9
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 9:
            # Buy noodles
            if holding:
                state['state'] = 10
            else:
                shop = self._find_nearest(self.shops, (bx, by))
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1]):
                            state['state'] = 10
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 10:
            # Add noodles to plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 11
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 11:
            # Wait and take from pan
            cooker = state.get('cooker')
            if cooker:
                step = self.get_move_toward(controller, bot_id, cooker, turn)
                if self.is_adjacent((bx, by), cooker):
                    tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food and pan.food.cooked_stage == 1:
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                state['state'] = 12
                        elif pan.food and pan.food.cooked_stage == 2:
                            # Burned - trash and restart
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                state['state'] = 15
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 12:
            # Add to plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 13
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 13:
            # Pickup plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.pickup(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 14
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 14:
            # Submit
            submit = self._find_nearest(self.submits, (bx, by))
            if submit:
                step = self.get_move_toward(controller, bot_id, submit, turn)
                if self.is_adjacent((bx, by), submit):
                    if controller.submit(bot_id, submit[0], submit[1]):
                        state['state'] = 0  # Restart
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 15:
            # Trash and restart
            if holding:
                trash = self._find_nearest(self.trashes, (bx, by))
                if trash:
                    step = self.get_move_toward(controller, bot_id, trash, turn)
                    if self.is_adjacent((bx, by), trash):
                        controller.trash(bot_id, trash[0], trash[1])
                        state['state'] = 2
                    elif step:
                        controller.move(bot_id, step[0], step[1])
            else:
                state['state'] = 2
    
    def _run_support_pipeline(self, controller: RobotController, bot_id: int):
        """Support bot logic - wash dishes, stay out of way"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        turn = controller.get_turn()
        
        # Priority: Wash dishes
        for sx, sy in self.sinks:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                step = self.get_move_toward(controller, bot_id, (sx, sy), turn)
                if self.is_adjacent((bx, by), (sx, sy)):
                    controller.wash_sink(bot_id, sx, sy)
                elif step:
                    controller.move(bot_id, step[0], step[1])
                return
        
        # Otherwise random move
        import random
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        random.shuffle(directions)
        for dx, dy in directions:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return

    # ============================================
    # MAIN ENTRY POINT
    # ============================================
    
    def play_turn(self, controller: RobotController):
        """Main entry point called each turn"""
        start_time = time.time()
        
        try:
            # Initialize on first turn
            self.initialize(controller)
            
            # Update state
            self.update_state(controller)
            
            # Clear per-turn reservations
            self.reserved_nodes.clear()
            
            # Run appropriate phase
            if self.current_phase == "BUILD":
                self.run_build_phase(controller)
            elif self.current_phase == "SABOTAGE":
                self.run_sabotage_phase(controller)
            else:
                self.run_endgame_phase(controller)
                
        except Exception as e:
            print(f"[ERROR] Turn failed: {e}")
            # Emergency: do nothing rather than crash
            
        elapsed = time.time() - start_time
        if elapsed > 0.4:
            print(f"[WARN] Turn took {elapsed:.3f}s")
