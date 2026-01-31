"""
APEX CHEF v1.0
Codename: "The Planner Killer"
Strategy: Global Supply Chain Optimization + Aggressive Sabotage
"""

import time
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any, Set
from enum import Enum, auto

# -----------------------------------------------------------------------------
# SAFETY IMPORTS
# -----------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] Scipy missing. Bot running in degraded greedy mode.")

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food

# ============================================
# CONFIGURATION
# ============================================

# Priority Hierarchy (Higher = More Urgent)
P_CRITICAL     = 1000  # Save burning food
P_SABOTAGE     = 950   # Steal plates (Phase 2)
P_STATE_FIX    = 800   # "I hold X, I must do Y" (Prevents getting stuck)
P_SUBMIT       = 500   # Cash in
P_COOK_STEP    = 300   # Active cooking steps
P_LOGISTICS    = 100   # Buying/Fetching
P_MAINTENANCE  = 50    # Washing
P_IDLE         = 0

# Pathfinding Penalties
COST_IMPOSSIBLE = 99999
COST_HIGH       = 5000

class JobType(Enum):
    # Critical
    SAVE_FOOD = auto()
    
    # State Fixes
    FINISH_ITEM = auto()
    
    # Sabotage
    STEAL_PLATE = auto()
    
    # Standard Flow
    SUBMIT = auto()
    BUY_INGREDIENT = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    PLACE_ON_COUNTER = auto() # Includes Boxes
    CHOP = auto()
    START_COOK = auto()
    TAKE_FROM_PAN = auto()
    ADD_TO_PLATE = auto()
    WASH = auto()
    IDLE = auto()

@dataclass
class Job:
    job_type: JobType
    target: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    priority: int = 0
    
    def __repr__(self):
        return f"<{self.job_type.name} P:{self.priority}>"

@dataclass 
class CookingInfo:
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int
    
    @property
    def turns_to_burned(self) -> int:
        if self.cooked_stage >= 2: return 0
        return max(0, 40 - self.cook_progress)

# ============================================
# APEX CHEF CONTROLLER
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Navigation
        self.dist_matrix = {}
        self.walkable = set()
        self.reserved_nodes = set()
        
        # POIs
        self.shops = []
        self.cookers = []
        self.sinks = []
        self.sink_tables = []
        self.counters = [] # Includes Counters + Boxes
        self.submits = []
        self.trashes = []
        
        # State Tracking
        self.cooking_info = {}
        self.active_orders = []
        self.global_supply = Counter() # Supply Chain Tracker

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def initialize(self, controller: RobotController):
        if self.initialized: return
        m = controller.get_map(controller.get_team())
        
        # Parse Map
        for x in range(m.width):
            for y in range(m.height):
                t = m.tiles[x][y]
                if getattr(t, 'is_walkable', False): 
                    self.walkable.add((x, y))
                
                tn = getattr(t, 'tile_name', '')
                if tn == 'SHOP': self.shops.append((x, y))
                elif tn == 'COOKER': self.cookers.append((x, y))
                elif tn == 'SINK': self.sinks.append((x, y))
                elif tn == 'SINKTABLE': self.sink_tables.append((x, y))
                elif tn == 'COUNTER': self.counters.append((x, y))
                elif tn == 'BOX': self.counters.append((x, y)) # Treat Boxes as Counters
                elif tn == 'SUBMIT': 
                    self.submits.append((x, y))
                    self.walkable.add((x, y))
                elif tn == 'TRASH': self.trashes.append((x, y))

        # Build Chebyshev Distance Matrix (8-way movement)
        self._build_distance_matrix(m)
        self.initialized = True
        print(f"[APEX] System Online. {len(self.walkable)} nodes mapped.")

    def _build_distance_matrix(self, m):
        for start in self.walkable:
            self.dist_matrix[start] = {start: 0}
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                (cx, cy), dist = queue.popleft()
                # 8 Directions
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx==0 and dy==0: continue
                        nx, ny = cx+dx, cy+dy
                        
                        if (nx, ny) in visited: continue
                        if not (0 <= nx < m.width and 0 <= ny < m.height): continue
                        
                        if getattr(m.tiles[nx][ny], 'is_walkable', False):
                            visited.add((nx, ny))
                            self.dist_matrix[start][(nx, ny)] = dist + 1
                            queue.append(((nx, ny), dist + 1))

    # -------------------------------------------------------------------------
    # STATE ANALYSIS
    # -------------------------------------------------------------------------
    def update_state(self, controller: RobotController):
        # 1. Orders
        self.active_orders = [o for o in controller.get_orders(controller.get_team()) if o.get('is_active')]
        
        # 2. Cooking Status
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            t = controller.get_tile(controller.get_team(), kx, ky)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan) and item.food:
                self.cooking_info[(kx, ky)] = CookingInfo(
                    (kx, ky), item.food.food_name, 
                    getattr(t, 'cook_progress', 0), item.food.cooked_stage
                )

        # 3. Supply Chain Audit (Prevent Double-Buying)
        self.global_supply.clear()
        # Count held items
        for bid in controller.get_team_bot_ids(controller.get_team()):
            h = controller.get_bot_state(bid).get('holding')
            if h and h.get('type') == 'Food':
                self.global_supply[h['food_name']] += 1
        # Count items on counters/boxes
        for cx, cy in self.counters:
            t = controller.get_tile(controller.get_team(), cx, cy)
            if t and t.item and isinstance(t.item, Food):
                self.global_supply[t.item.food_name] += 1
        # Count items in pans
        for info in self.cooking_info.values():
            self.global_supply[info.food_name] += 1

    # -------------------------------------------------------------------------
    # HIVE MIND: JOB GENERATION
    # -------------------------------------------------------------------------
    def generate_jobs(self, controller) -> List[Job]:
        jobs = []
        bots = controller.get_team_bot_ids(controller.get_team())
        
        # 1. EMERGENCY (Save Burnt Food)
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL + 10))

        # 2. STATE-TRANSITION (Fix "Holding" States)
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            if not bot: continue
            holding = bot.get('holding')
            if not holding: continue

            h_type = holding.get('type')
            
            if h_type == 'Food':
                name = holding.get('food_name', '').upper()
                is_chopped = holding.get('chopped', False)
                stage = holding.get('cooked_stage', 0)
                
                # Logic tree for food processing
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    # Must place first!
                    jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX)) 
                elif (name == 'MEAT' and is_chopped and stage == 0) or (name == 'EGG' and stage == 0):
                    jobs.append(Job(JobType.START_COOK, priority=P_STATE_FIX))
                else:
                    jobs.append(Job(JobType.ADD_TO_PLATE, priority=P_STATE_FIX))

            elif h_type == 'Plate':
                if len(holding.get('food', [])) > 0:
                    jobs.append(Job(JobType.SUBMIT, priority=P_SUBMIT))
            
            elif h_type == 'Pan':
                jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX))

        # 2b. MAP SCANNING (Find items to work on)
        team = controller.get_team()
        for cx, cy in self.counters:
             tile = controller.get_tile(team, cx, cy)
             if tile and tile.item and isinstance(tile.item, Food):
                  # Check key: 'chopped' isn't on object implies not chopped? 
                  # Wait, tile.item is Item object. 'chopped' is attr.
                  if tile.item.food_name in ['MEAT', 'ONIONS'] and not tile.item.chopped:
                       jobs.append(Job(JobType.CHOP, target=(cx, cy), priority=P_COOK_STEP))

        # 3. SABOTAGE (Phase 2 Only)
        switch_info = controller.get_switch_info()
        if switch_info['my_team_switched']:
            # Aggressive Plate Denial
            for st in self.sink_tables:
                jobs.append(Job(JobType.STEAL_PLATE, target=st, priority=P_SABOTAGE))
            return jobs # Ignore normal orders during sabotage

        # 4. ORDER FULFILLMENT (Supply Chain Logic)
        for order in self.active_orders:
            # Calculate demand for this order
            needed_for_order = Counter()
            for req in order['required']:
                needed_for_order[req.upper()] += 1
            
            prio = P_LOGISTICS + (order['reward'] // 100)
            
            for item, count in needed_for_order.items():
                have = self.global_supply[item]
                needed = count - have
                
                if needed > 0:
                    ft = self._name_to_foodtype(item)
                    if ft:
                        # Generate jobs to fill deficit
                        for _ in range(needed):
                            jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=prio))
                            self.global_supply[item] += 1 # Virtual increment to prevent double-buy

        # 5. MAINTENANCE
        clean_plates = sum(getattr(controller.get_tile(controller.get_team(), *s), 'num_clean_plates', 0) for s in self.sink_tables)
        if clean_plates < 2:
            for s in self.sinks: 
                jobs.append(Job(JobType.WASH, target=s, priority=P_MAINTENANCE))
            
        jobs.append(Job(JobType.BUY_PAN, priority=40))
        jobs.append(Job(JobType.BUY_PLATE, priority=45))
        jobs.append(Job(JobType.IDLE, priority=P_IDLE))
        
        return jobs

    # -------------------------------------------------------------------------
    # HIVE MIND: TASK ASSIGNMENT
    # -------------------------------------------------------------------------
    def assign_tasks(self, controller, jobs):
        bots = controller.get_team_bot_ids(controller.get_team())
        if not bots or not jobs: return {}
        
        # Create Cost Matrix (Rows=Bots, Cols=Jobs)
        cost_matrix = np.full((len(bots), len(jobs)), float(COST_IMPOSSIBLE))
        
        for r, bid in enumerate(bots):
            bot = controller.get_bot_state(bid)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            
            for c, job in enumerate(jobs):
                # 1. Distance Cost
                target = job.target
                if not target:
                    # Dynamically resolve nearest target for generic jobs
                    if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
                    elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
                    elif job.job_type == JobType.START_COOK: target = self.get_nearest(pos, self.cookers)
                    elif job.job_type == JobType.ADD_TO_PLATE: target = self.get_nearest(pos, self.counters)
                    elif job.job_type == JobType.PLACE_ON_COUNTER: target = self._find_empty(controller, self.counters, pos)
                
                dist = self.get_dist(pos, target) if target else 0
                
                # 2. State Constraints (The Guard)
                penalty = 0
                jt = job.job_type
                
                # Physical impossibility checks
                if jt == JobType.CHOP:
                    if not holding or holding.get('type') == 'Plate': penalty = COST_HIGH
                    else: penalty = -500 # Bias to finish holding task
                elif jt == JobType.START_COOK:
                    if not holding or holding.get('type') != 'Food': penalty = COST_HIGH
                    else: penalty = -500
                elif jt == JobType.BUY_INGREDIENT or jt == JobType.BUY_PAN:
                    if holding: penalty = COST_HIGH
                elif jt == JobType.SUBMIT:
                    if not holding or holding.get('type') != 'Plate': penalty = COST_HIGH

                cost_matrix[r, c] = dist - job.priority + penalty

        # 3. Solve Assignment
        if HAS_SCIPY:
            row, col = linear_sum_assignment(cost_matrix)
        else:
            col = np.argmin(cost_matrix, axis=1) # Fallback
            row = range(len(bots))

        return {bots[r]: jobs[c] for r, c in zip(row, col)}

    # -------------------------------------------------------------------------
    # BODY: PATHFINDING (Time-Space A*)
    # -------------------------------------------------------------------------
    def get_move(self, controller, bot_id, target, turn):
        start = (controller.get_bot_state(bot_id)['x'], controller.get_bot_state(bot_id)['y'])
        if self.get_dist(start, target) <= 1: return (0,0)
        
        queue = deque([(start, [])])
        visited = {start}
        
        # Dynamic Obstacles (Prevent colliding with any bot)
        obstacles = set()
        all_bots = controller.get_team_bot_ids(controller.get_team()) + \
                   controller.get_team_bot_ids(controller.get_enemy_team())
        for bid in all_bots:
            if bid == bot_id: continue
            b = controller.get_bot_state(bid)
            if b: obstacles.add((b['x'], b['y']))
        
        while queue:
            curr, path = queue.popleft()
            
            # Adjacency check
            if self.get_dist(curr, target) <= 1:
                if not path: return (0,0)
                # Reservation Logic
                next_node = (start[0]+path[0][0], start[1]+path[0][1])
                self.reserved_nodes.add((next_node[0], next_node[1], turn+1))
                return path[0]
            
            if len(path) > 10: continue # Depth limit
            
            # 8-Way Movement (Chebyshev) + Wait
            moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
            
            for dx, dy in moves:
                nx, ny = curr[0]+dx, curr[1]+dy
                
                # Validations
                if not (0<=nx<self.map.width and 0<=ny<self.map.height): continue
                if (nx, ny) not in self.walkable: continue
                # Collision Check
                if (nx, ny, turn+len(path)+1) in self.reserved_nodes: continue
                
                # Dynamic Collision (Immediate step)
                if len(path) == 0 and (nx, ny) in obstacles: continue

                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))
        return (0,0)

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, controller, bot_id, job):
        bot = controller.get_bot_state(bot_id)
        pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        # 1. HANDS FULL OVERRIDE
        # If assigned to buy but hands full -> Trash it
        if job.job_type == JobType.BUY_INGREDIENT and holding:
            trash = self.get_nearest(pos, self.trashes)
            if self.get_dist(pos, trash) <= 1: 
                controller.trash(bot_id, trash[0], trash[1])
            else: 
                self.move_bot(controller, bot_id, trash)
            return

        # 2. RESOLVE TARGET
        target = job.target
        if not target:
            if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
            elif job.job_type == JobType.PLACE_ON_COUNTER: target = self._find_empty(controller, self.counters, pos)
            elif job.job_type == JobType.CHOP: target = self._find_choppable(controller, self.counters, pos)
            elif job.job_type == JobType.START_COOK: target = self._find_empty_pan(controller, self.cookers, pos)
            elif job.job_type == JobType.ADD_TO_PLATE: target = self._find_plate(controller, self.counters, pos)
            elif job.job_type == JobType.SUBMIT: target = self.get_nearest(pos, self.submits)
        
        if not target: return 
        
        # 3. ACT or MOVE
        if self.get_dist(pos, target) <= 1:
            self._perform_action(controller, bot_id, job, target)
        else:
            self.move_bot(controller, bot_id, target)

    def move_bot(self, controller, bot_id, target):
        dx, dy = self.get_move(controller, bot_id, target, controller.get_turn())
        controller.move(bot_id, dx, dy)

    def _perform_action(self, controller, bot_id, job, target):
        tx, ty = target
        jt = job.job_type
        
        if jt == JobType.BUY_INGREDIENT: controller.buy(bot_id, job.item, tx, ty)
        elif jt == JobType.BUY_PAN: controller.buy(bot_id, ShopCosts.PAN, tx, ty)
        elif jt == JobType.BUY_PLATE: controller.buy(bot_id, ShopCosts.PLATE, tx, ty)
        elif jt == JobType.CHOP: controller.chop(bot_id, tx, ty)
        elif jt == JobType.START_COOK: controller.place(bot_id, tx, ty)
        elif jt == JobType.TAKE_FROM_PAN: controller.take_from_pan(bot_id, tx, ty)
        elif jt == JobType.ADD_TO_PLATE: controller.add_food_to_plate(bot_id, tx, ty)
        elif jt == JobType.SUBMIT: controller.submit(bot_id, tx, ty)
        elif jt == JobType.PLACE_ON_COUNTER: controller.place(bot_id, tx, ty)
        elif jt == JobType.WASH: controller.wash_sink(bot_id, tx, ty)
        elif jt == JobType.STEAL_PLATE:
            if controller.get_bot_state(bot_id).get('holding'): 
                # Holding stolen plate? Trash it
                tr = self.get_nearest((tx, ty), self.trashes)
                controller.trash(bot_id, tr[0], tr[1])
            else: 
                controller.take_clean_plate(bot_id, tx, ty)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def get_dist(self, a, b):
        if a in self.dist_matrix and b in self.dist_matrix[a]: return self.dist_matrix[a][b]
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def get_nearest(self, pos, locs):
        if not locs: return None
        return min(locs, key=lambda l: self.get_dist(pos, l))

    def _find_empty(self, c, locs, pos):
        # Prefer closest empty counter/box
        sorted_locs = sorted(locs, key=lambda x: self.get_dist(pos, x))
        for l in sorted_locs:
            if not getattr(c.get_tile(c.get_team(), *l), 'item', None): return l
        return sorted_locs[0] if sorted_locs else None

    def _find_choppable(self, c, locs, pos):
        return self.get_nearest(pos, locs)

    def _find_empty_pan(self, c, locs, pos):
        for l in sorted(locs, key=lambda x: self.get_dist(pos, x)):
            it = getattr(c.get_tile(c.get_team(), *l), 'item', None)
            if isinstance(it, Pan) and not it.food: return l
        return None

    def _find_plate(self, c, locs, pos):
        for l in sorted(locs, key=lambda x: self.get_dist(pos, x)):
            it = getattr(c.get_tile(c.get_team(), *l), 'item', None)
            if isinstance(it, Plate): return l
        return None
        
    def _name_to_foodtype(self, n):
        n = n.upper()
        if n == 'ONION': return FoodType.ONIONS
        if n == 'ONIONS': return FoodType.ONIONS
        if n == 'MEAT': return FoodType.MEAT
        if n == 'EGG': return FoodType.EGG
        if n == 'NOODLES': return FoodType.NOODLES
        if n == 'SAUCE': return FoodType.SAUCE
        return None

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    def play_turn(self, controller: RobotController):
        self.initialize(controller)
        self.update_state(controller)
        self.reserved_nodes.clear()
        
        # Turn 250: Aggressive Switch
        turn = controller.get_turn()
        if 250 <= turn < 350 and not controller.get_switch_info()['my_team_switched']:
            controller.switch_maps()

        # 1. Think
        jobs = self.generate_jobs(controller)
        
        # 2. Pair
        assigns = self.assign_tasks(controller, jobs)
        
        # 3. Act (Priority Order)
        sorted_bots = sorted(assigns.keys(), key=lambda b: assigns[b].priority, reverse=True)
        
        start_ts = time.time()
        for bid in sorted_bots:
            if time.time() - start_ts > 0.45: break # Hard Time Limit
            self.execute(controller, bid, assigns[bid])