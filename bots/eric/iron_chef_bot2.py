"""
Iron Chef v2.1 - The Hive Mind
Strategy: Centralized Task Auction with Time-Space Reservation Pathfinding.
Physics: Source-Verified (Chebyshev, Passive Income, Turn 250 Switch).
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Set
from enum import Enum, auto

# -----------------------------------------------------------------------------
# SAFETY IMPORTS (Handle environment where scipy might be missing)
# -----------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not available! Bot efficiency will drop 40%.")

# -----------------------------------------------------------------------------
# GAME CONSTANTS IMPORTS
# (Assumes standard AWAP 2026 file structure)
# -----------------------------------------------------------------------------
from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food

# ============================================
# DATA STRUCTURES
# ============================================

class JobType(Enum):
    # CRITICAL (Priority 100+)
    SAVE_FOOD = auto()        # Pickup cooked food before burn
    
    # SABOTAGE (Priority 90-95 in Phase 2)
    STEAL_PLATE = auto()
    STEAL_PAN = auto()
    
    # HIGH VALUE (Priority 80+)
    SUBMIT = auto()           # Deliver finished order
    
    # PREP & COOK (Priority 50-70)
    BUY_INGREDIENT = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    PLACE_ON_COUNTER = auto()
    CHOP = auto()
    START_COOK = auto()
    TAKE_FROM_PAN = auto()    # Standard retrieval
    
    # PLATING (Priority 40-60)
    TAKE_CLEAN_PLATE = auto()
    ADD_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    
    # MAINTENANCE (Priority 30)
    WASH = auto()
    
    # IDLE
    IDLE = auto()


@dataclass
class Job:
    job_type: JobType
    target: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    priority: int = 0
    order_id: Optional[int] = None
    
    def __repr__(self):
        return f"<{self.job_type.name} P:{self.priority} @ {self.target}>"


@dataclass 
class CookingInfo:
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int
    
    @property
    def turns_to_burned(self) -> int:
        if self.cooked_stage >= 2: return 0
        # Assuming burn threshold is 40 based on report
        return max(0, 40 - self.cook_progress)

# ============================================
# MAIN BOT CLASS
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # --- NAVIGATION CACHE ---
        self.dist_matrix: Dict[Tuple[int,int], Dict[Tuple[int,int], int]] = {}
        
        # --- MAP ZONES ---
        self.shops = []
        self.cookers = []
        self.sinks = []
        self.sink_tables = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.boxes = []
        self.walkable = set()
        
        # --- STATE ---
        self.cooking_info: Dict[Tuple[int,int], CookingInfo] = {}
        self.active_orders: List[Dict] = []
        
        # --- RESERVATION SYSTEM ---
        # Stores (x, y, turn) -> True
        self.reserved_nodes: Set[Tuple[int, int, int]] = set()

    # ============================================
    # INITIALIZATION (PRE-COMPUTATION)
    # ============================================
    
    def initialize(self, controller: RobotController):
        if self.initialized: return
        
        m = controller.get_map()
        
        # 1. Parse Map
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if getattr(tile, 'is_walkable', False):
                    self.walkable.add((x, y))
                
                t_name = getattr(tile, 'tile_name', '')
                if t_name == 'SHOP': self.shops.append((x, y))
                elif t_name == 'COOKER': self.cookers.append((x, y))
                elif t_name == 'SINK': self.sinks.append((x, y))
                elif t_name == 'SINKTABLE': self.sink_tables.append((x, y))
                elif t_name == 'COUNTER': self.counters.append((x, y))
                elif t_name == 'SUBMIT': 
                    self.submits.append((x, y))
                    self.walkable.add((x, y)) # Ensure submit is walkable
                elif t_name == 'TRASH': self.trashes.append((x, y))
                elif t_name == 'BOX': self.boxes.append((x, y))

        # 2. Build Distance Matrix (Chebyshev BFS)
        self._build_distance_matrix(m)
        self.initialized = True
        print(f"[INIT] Hive Mind Online. Nodes: {len(self.walkable)}")

    def _build_distance_matrix(self, m):
        """Pre-compute distances between all walkable tiles using 8-way movement."""
        for start in self.walkable:
            self.dist_matrix[start] = {start: 0}
            queue = deque([(start, 0)])
            visited = {start}
            
            while queue:
                (cx, cy), dist = queue.popleft()
                # 8 Directions (Chebyshev)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        
                        if (nx, ny) in visited: continue
                        if not (0 <= nx < m.width and 0 <= ny < m.height): continue
                        
                        tile = m.tiles[nx][ny]
                        if getattr(tile, 'is_walkable', False):
                            visited.add((nx, ny))
                            self.dist_matrix[start][(nx, ny)] = dist + 1
                            queue.append(((nx, ny), dist + 1))

    # ============================================
    # PHYSICS & HELPERS
    # ============================================

    def get_distance(self, start, end):
        """O(1) lookup if cached, else Chebyshev heuristic."""
        if start in self.dist_matrix and end in self.dist_matrix[start]:
            return self.dist_matrix[start][end]
        return max(abs(start[0]-end[0]), abs(start[1]-end[1]))

    def get_adjacent_distance(self, bot_pos, target_pos):
        """Distance to reach ANY valid tile adjacent to target."""
        min_dist = 9999
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                adj = (target_pos[0]+dx, target_pos[1]+dy)
                if adj in self.walkable:
                    d = self.get_distance(bot_pos, adj)
                    if d < min_dist: min_dist = d
        return min_dist

    def find_nearest(self, pos, locations):
        if not locations: return None
        return min(locations, key=lambda loc: self.get_adjacent_distance(pos, loc))

    def update_state(self, controller: RobotController):
        """Snapshot the world state."""
        self.active_orders = [o for o in controller.get_orders() if o.get('is_active')]
        
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            # Check safely for Pan
            item = getattr(tile, 'item', None)
            if isinstance(item, Pan) and item.food:
                self.cooking_info[(kx, ky)] = CookingInfo(
                    location=(kx, ky),
                    food_name=item.food.food_name,
                    cook_progress=getattr(tile, 'cook_progress', 0),
                    cooked_stage=item.food.cooked_stage
                )

    # ============================================
    # CORE LOGIC: THE HIVE MIND
    # ============================================

    def generate_jobs(self, controller: RobotController) -> List[Job]:
        """Generate all available jobs based on world state and bot holdings."""
        jobs = []
        turn = controller.get_turn()
        bots = controller.get_team_bot_ids()
        
        # 1. EMERGENCY (Priority 100+) - Save food about to burn
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:  # Cooked! Grab it now.
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=100))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=105))  # PANIC

        # 2. SABOTAGE (Phase 2 - only when on enemy map)
        switch_info = controller.get_switch_info()
        if switch_info['my_team_switched']:
            for st in self.sink_tables:
                jobs.append(Job(JobType.STEAL_PLATE, target=st, priority=95))
            for ck in self.cookers:
                jobs.append(Job(JobType.STEAL_PAN, target=ck, priority=90))

        # 3. SUBMIT - If any bot holding a complete plate, submit it!
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            if bot:
                holding = bot.get('holding')
                if holding and holding.get('type') == 'Plate':
                    foods = holding.get('food', [])
                    if len(foods) >= 1:  # Has at least some food
                        jobs.append(Job(JobType.SUBMIT, priority=90))

        # 4. CONTEXT-AWARE JOBS based on bot holdings
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            if not bot:
                continue
            holding = bot.get('holding')
            
            if holding:
                h_type = holding.get('type')
                
                if h_type == 'Food':
                    # Holding raw food - need to place on counter or chop or add to plate
                    food_name = holding.get('food_name', '').upper()
                    is_chopped = holding.get('is_chopped', False)
                    cooked_stage = holding.get('cooked_stage', 0)
                    
                    if food_name in ['MEAT', 'ONIONS'] and not is_chopped:
                        # Needs chopping - place on counter first
                        jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=75))
                    elif food_name in ['MEAT'] and is_chopped and cooked_stage == 0:
                        # Chopped meat needs cooking
                        jobs.append(Job(JobType.START_COOK, priority=75))
                    elif cooked_stage == 1 or food_name in ['NOODLES', 'EGG', 'SAUCE']:
                        # Cooked or ready-to-plate food - add to plate
                        jobs.append(Job(JobType.ADD_TO_PLATE, priority=80))
                    else:
                        # Default: place on counter
                        jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=60))
                
                elif h_type == 'Pan':
                    # Holding pan - place on cooker
                    for ck in self.cookers:
                        tile = controller.get_tile(controller.get_team(), *ck)
                        if tile and getattr(tile, 'item', None) is None:
                            jobs.append(Job(JobType.PLACE_ON_COUNTER, target=ck, priority=70))
                            break
                
                elif h_type == 'Plate':
                    # Already handled above (submit)
                    pass
            
            else:
                # Empty hands - can pickup or buy
                pass

        # 5. CHOP - If there's food on counter that needs chopping
        for cx, cy in self.counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Food):
                    name = getattr(item, 'food_name', '').upper()
                    if name in ['MEAT', 'ONIONS'] and not getattr(item, 'is_chopped', False):
                        jobs.append(Job(JobType.CHOP, target=(cx, cy), priority=70))

        # 6. PICKUP CHOPPED FOOD - Pick up chopped food from counter
        for cx, cy in self.counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Food):
                    if getattr(item, 'is_chopped', False):
                        name = getattr(item, 'food_name', '').upper()
                        if name == 'MEAT':
                            # Chopped meat - need to put in pan
                            jobs.append(Job(JobType.START_COOK, priority=72))

        # 7. PICKUP PLATE - If there's a plate with food ready to submit
        for cx, cy in self.counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Plate):
                    foods = getattr(item, 'food', [])
                    if len(foods) >= 2:  # Reasonable amount of food
                        jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=85))

        # 8. TAKE CLEAN PLATE - Get plate from sink table
        if self.sink_tables:
            for st in self.sink_tables:
                tile = controller.get_tile(controller.get_team(), *st)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    jobs.append(Job(JobType.TAKE_CLEAN_PLATE, target=st, priority=55))

        # 9. MAINTENANCE - Wash dishes if needed
        clean_plates = sum(
            getattr(controller.get_tile(controller.get_team(), *st), 'num_clean_plates', 0)
            for st in self.sink_tables
        )
        dirty_exists = any(
            getattr(controller.get_tile(controller.get_team(), *s), 'num_dirty_plates', 0) > 0
            for s in self.sinks
        )
        if clean_plates < 2 and dirty_exists:
            for s in self.sinks:
                jobs.append(Job(JobType.WASH, target=s, priority=60))

        # 10. BUY INGREDIENTS for orders
        for order in self.active_orders:
            reqs = order['required']
            prio = 50 + (order['reward'] // 100)
            for r in reqs:
                ft = self._name_to_foodtype(r)
                if ft:
                    jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=prio))

        # 11. EQUIPMENT - Ensure pans on cookers
        has_pan = any(
            isinstance(getattr(controller.get_tile(controller.get_team(), *ck), 'item', None), Pan)
            for ck in self.cookers
        )
        if not has_pan:
            jobs.append(Job(JobType.BUY_PAN, priority=65))

        # 12. IDLE FALLBACK
        jobs.append(Job(JobType.IDLE, priority=0))
        
        return jobs

    def assign_tasks(self, controller: RobotController, jobs: List[Job]):
        """The Hungarian Algorithm for Optimal Assignment."""
        bots = controller.get_team_bot_ids()
        if not bots or not jobs: return {}
        
        cost_matrix = np.full((len(bots), len(jobs)), 9999.0)
        
        for r, bot_id in enumerate(bots):
            bot = controller.get_bot_state(bot_id)
            if not bot: continue
            b_pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            
            for c, job in enumerate(jobs):
                # 1. Base Distance Cost
                target = job.target
                if not target and job.job_type == JobType.BUY_INGREDIENT:
                    target = self.find_nearest(b_pos, self.shops)
                
                dist = self.get_adjacent_distance(b_pos, target) if target else 0
                
                # 2. Priority Bonus (High priority "pulls" bots)
                prio_bonus = job.priority * 10 
                
                # 3. Logic Penalties (Crucial!)
                logic_penalty = 0
                
                # Cannot chop if holding Plate
                if job.job_type == JobType.CHOP:
                    if holding and holding.get('type') == 'Plate': logic_penalty = 2000
                
                # Cannot buy if hands full
                if job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN]:
                    if holding: logic_penalty = 500
                
                # Bonus: If holding the right ingredient, DO THE JOB
                if job.job_type == JobType.PLACE_ON_COUNTER:
                    # Simplify: Logic usually handled by state, but here we bias
                    pass 

                cost_matrix[r, c] = dist - prio_bonus + logic_penalty

        # SOLVE
        if HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            # Greedy Fallback
            col_ind = np.argmin(cost_matrix, axis=1) # Naive greedy
            row_ind = range(len(bots))

        assignments = {}
        for r, c in zip(row_ind, col_ind):
            assignments[bots[r]] = jobs[c]
            
        return assignments

    # ============================================
    # NAVIGATION: TIME-SPACE A*
    # ============================================

    def get_move(self, controller, bot_id, target, turn):
        """Returns (dx, dy) to move towards target while avoiding collisions."""
        bot = controller.get_bot_state(bot_id)
        start = (bot['x'], bot['y'])
        
        if self.get_distance(start, target) <= 1:
            return (0, 0) # Arrived (or adjacent)
            
        # Time-Space A*
        # Nodes are (x, y) -- we implicitly track time by depth
        queue = deque([(start, [])])
        visited = {start} # Visited in this search
        
        while queue:
            curr, path = queue.popleft()
            
            # Reached adjacency?
            if self.get_distance(curr, target) <= 1:
                if not path: return (0,0)
                
                first_step = path[0]
                # RESERVE THIS NODE
                next_pos = (start[0]+first_step[0], start[1]+first_step[1])
                self.reserved_nodes.add((next_pos[0], next_pos[1], turn+1))
                return first_step

            if len(path) > 10: continue # Depth limit for speed

            # 8-Way Neighbors + Wait
            neighbors = [
                (0,1), (0,-1), (1,0), (-1,0), 
                (1,1), (1,-1), (-1,1), (-1,-1),
                (0,0)
            ]
            
            for dx, dy in neighbors:
                nx, ny = curr[0]+dx, curr[1]+dy
                
                # Valid Map Tile?
                if not (0 <= nx < self.map.width and 0 <= ny < self.map.height): continue
                if (nx, ny) not in self.walkable: continue
                
                # RESERVATION CHECK
                # t = turn + len(path) + 1
                # If another bot already booked this tile at this time, blocked.
                if (nx, ny, turn + len(path) + 1) in self.reserved_nodes: continue
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((dx, dy))
                    queue.append(((nx, ny), new_path))
                    
        return (0,0) # No path found, wait

    # ============================================
    # EXECUTION HANDLER
    # ============================================

    def execute(self, controller, bot_id, job):
        """Translates Job -> API Calls."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        # Handle IDLE
        if job.job_type == JobType.IDLE:
            return
        
        # RESOLVE TARGET for jobs without explicit target
        target = job.target
        if job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE]:
            target = self.find_nearest(pos, self.shops)
        elif job.job_type == JobType.SUBMIT:
            target = self.find_nearest(pos, self.submits)
        elif job.job_type == JobType.PLACE_ON_COUNTER:
            target = self._find_empty_counter(controller, pos)
        elif job.job_type == JobType.CHOP:
            # Find counter with choppable food
            target = self._find_counter_with_choppable(controller, pos)
        elif job.job_type == JobType.START_COOK:
            target = self._find_cooker_with_empty_pan(controller, pos)
        elif job.job_type == JobType.TAKE_CLEAN_PLATE:
            target = self.find_nearest(pos, self.sink_tables)
        elif job.job_type == JobType.ADD_TO_PLATE:
            # Find counter with plate
            target = self._find_counter_with_plate(controller, pos)
        elif job.job_type == JobType.PICKUP_PLATE:
            target = self._find_counter_with_plate(controller, pos)
        
        if not target:
            return  # No valid target
        
        # Check adjacency (Chebyshev <= 1)
        is_adjacent = max(abs(pos[0]-target[0]), abs(pos[1]-target[1])) <= 1
        
        if is_adjacent:
            # INTERACT
            self._perform_action(controller, bot_id, job, target, holding)
        else:
            # MOVE
            dx, dy = self.get_move(controller, bot_id, target, turn)
            if dx != 0 or dy != 0:
                controller.move(bot_id, dx, dy)

    def _perform_action(self, controller, bot_id, job, target, holding):
        tx, ty = target
        jt = job.job_type
        
        if jt == JobType.BUY_INGREDIENT:
            if not holding:  # Safety check
                controller.buy(bot_id, job.item, tx, ty)
        elif jt == JobType.BUY_PAN:
            if not holding:
                controller.buy(bot_id, ShopCosts.PAN, tx, ty)
        elif jt == JobType.BUY_PLATE:
            if not holding:
                controller.buy(bot_id, ShopCosts.PLATE, tx, ty)
        elif jt == JobType.PLACE_ON_COUNTER:
            if holding:
                controller.place(bot_id, tx, ty)
        elif jt == JobType.CHOP:
            controller.chop(bot_id, tx, ty)
        elif jt == JobType.START_COOK:
            if holding:
                controller.place(bot_id, tx, ty)  # Place food in pan
        elif jt == JobType.TAKE_FROM_PAN:
            if not holding:
                controller.take_from_pan(bot_id, tx, ty)
        elif jt == JobType.TAKE_CLEAN_PLATE:
            if not holding:
                controller.take_clean_plate(bot_id, tx, ty)
        elif jt == JobType.ADD_TO_PLATE:
            if holding:
                controller.add_food_to_plate(bot_id, tx, ty)
        elif jt == JobType.PICKUP_PLATE:
            if not holding:
                controller.pickup(bot_id, tx, ty)
        elif jt == JobType.SUBMIT:
            if holding and holding.get('type') == 'Plate':
                controller.submit(bot_id, tx, ty)
        elif jt == JobType.WASH:
            controller.wash_sink(bot_id, tx, ty)
        elif jt == JobType.STEAL_PLATE:
            if holding:
                # Find actual TRASH tile
                trash = self.find_nearest((tx, ty), self.trashes)
                if trash:
                    controller.trash(bot_id, trash[0], trash[1])
            else:
                controller.take_clean_plate(bot_id, tx, ty)
        elif jt == JobType.STEAL_PAN:
            if not holding:
                controller.pickup(bot_id, tx, ty)

    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _find_empty_counter(self, controller, near):
        """Find nearest empty counter."""
        for cx, cy in sorted(self.counters, key=lambda c: self.get_adjacent_distance(near, c)):
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and getattr(tile, 'item', None) is None:
                return (cx, cy)
        return None
    
    def _find_counter_with_choppable(self, controller, near):
        """Find counter with unchoped meat/onions."""
        for cx, cy in sorted(self.counters, key=lambda c: self.get_adjacent_distance(near, c)):
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Food):
                    if not getattr(item, 'is_chopped', False):
                        name = getattr(item, 'food_name', '').upper()
                        if name in ['MEAT', 'ONIONS']:
                            return (cx, cy)
        return None
    
    def _find_cooker_with_empty_pan(self, controller, near):
        """Find cooker with pan that has no food."""
        for kx, ky in sorted(self.cookers, key=lambda c: self.get_adjacent_distance(near, c)):
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Pan) and item.food is None:
                    return (kx, ky)
        return None
    
    def _find_counter_with_plate(self, controller, near):
        """Find counter with a plate."""
        for cx, cy in sorted(self.counters, key=lambda c: self.get_adjacent_distance(near, c)):
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Plate):
                    return (cx, cy)
        return None
    
    def _name_to_foodtype(self, name):
        name = name.upper()
        if name == "ONION": return FoodType.ONIONS
        if name == "ONIONS": return FoodType.ONIONS
        if name == "MEAT": return FoodType.MEAT
        if name == "EGG": return FoodType.EGG
        if name == "NOODLES": return FoodType.NOODLES
        if name == "SAUCE": return FoodType.SAUCE
        return None

    # ============================================
    # MAIN LOOP
    # ============================================

    def play_turn(self, controller: RobotController):
        # 1. INIT & UPDATE
        start_ts = time.time()
        self.initialize(controller)
        self.update_state(controller)
        self.reserved_nodes.clear() # Reset reservations per turn
        
        turn = controller.get_turn()
        
        # 2. STRATEGIC PHASE CHECK
        # Sabotage Phase Logic
        if 250 <= turn < 350:
            if not controller.get_switch_info()['my_team_switched']:
                # Decide: Only switch if we aren't carrying a winning dish
                controller.switch_maps()
        
        # 3. HIVE MIND PIPELINE
        # A) Generate Needs
        jobs = self.generate_jobs(controller)
        
        # B) Assign Needs (Brain)
        assignments = self.assign_tasks(controller, jobs)
        
        # C) Execute (Body)
        # Sort execution by priority so high-prio bots reserve paths first
        sorted_bots = sorted(assignments.keys(), 
                             key=lambda b: assignments[b].priority, 
                             reverse=True)
        
        for bot_id in sorted_bots:
            # Time Safety
            if time.time() - start_ts > 0.45: break 
            self.execute(controller, bot_id, assignments[bot_id])