"""
APEX CHEF v2.2
Codename: "Bootstrap"
Strategy: Global Supply Chain + Aggressive Sabotage + API v2 Support
Fixes: Allows 'Pre-movement' to shops even when broke, preventing start-of-game stalls.
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

# Priority Hierarchy
P_CRITICAL     = 1000  # Save burning food
P_SABOTAGE     = 950   # Steal plates (Phase 2)
P_STATE_FIX    = 800   # "I hold X, I must do Y"
P_WORLD_STATE  = 700   # "Item on counter needs X"
P_SUBMIT       = 600   # Cash in
P_COOK_STEP    = 500   # Active cooking steps
P_LOGISTICS    = 100   # Buying/Fetching
P_MAINTENANCE  = 50    # Washing
P_IDLE         = 0

# Pathfinding Penalties
COST_IMPOSSIBLE = 99999
COST_HIGH       = 5000

class JobType(Enum):
    SAVE_FOOD = auto()
    FINISH_ITEM = auto()
    STEAL_PLATE = auto()
    STEAL_PAN = auto()
    SUBMIT = auto()
    BUY_INGREDIENT = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    PLACE_ON_COUNTER = auto() # Includes Boxes
    CHOP = auto()
    START_COOK = auto()
    TAKE_FROM_PAN = auto()
    ADD_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    TAKE_CLEAN_PLATE = auto()
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
        self.global_supply = Counter()
        self.current_money = 0
        self.my_team = None
        self.empty_counters = 0

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def initialize(self, controller: RobotController):
        if self.initialized: return
        
        self.my_team = controller.get_team()
        m = controller.get_map(self.my_team) 
        
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
                elif tn == 'BOX': self.counters.append((x, y))
                elif tn == 'SUBMIT': 
                    self.submits.append((x, y))
                    self.walkable.add((x, y))
                elif tn == 'TRASH': self.trashes.append((x, y))

        # Build Chebyshev Distance Matrix
        self._build_distance_matrix(m)
        self.initialized = True
        print(f"[APEX v2.2] System Online. Team: {self.my_team}")

    def _build_distance_matrix(self, m):
        for start in self.walkable:
            self.dist_matrix[start] = {start: 0}
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                (cx, cy), dist = queue.popleft()
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
        self.my_team = controller.get_team()
        
        # 1. Orders & Money
        self.active_orders = [o for o in controller.get_orders(self.my_team) if o.get('is_active')]
        self.current_money = controller.get_team_money(self.my_team)
        
        # 2. Cooking Status
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            t = controller.get_tile(self.my_team, kx, ky)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan) and item.food:
                self.cooking_info[(kx, ky)] = CookingInfo(
                    (kx, ky), item.food.food_name, 
                    getattr(t, 'cook_progress', 0), item.food.cooked_stage
                )

        # 3. Supply Chain Audit
        self.global_supply.clear()
        self.empty_counters = 0
        
        # Count held items
        for bid in controller.get_team_bot_ids(self.my_team):
            h = controller.get_bot_state(bid).get('holding')
            if h and h.get('type') == 'Food':
                self.global_supply[h['food_name']] += 1
                
        # Count items on counters/boxes
        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if t and t.item:
                if isinstance(t.item, Food):
                    self.global_supply[t.item.food_name] += 1
            else:
                self.empty_counters += 1
                
        # Count items in pans
        for info in self.cooking_info.values():
            self.global_supply[info.food_name] += 1

    # -------------------------------------------------------------------------
    # HIVE MIND: JOB GENERATION
    # -------------------------------------------------------------------------
    def generate_jobs(self, controller) -> List[Job]:
        jobs = []
        bots = controller.get_team_bot_ids(self.my_team)
        
        # 0. INVENTORY AUDIT (Tools Check)
        pan_count = 0
        plate_count = 0
        
        # Count Map Assets
        for loc in self.cookers:
            t = controller.get_tile(self.my_team, *loc)
            if isinstance(getattr(t, 'item', None), Pan): pan_count += 1
        
        for loc in self.sink_tables:
            t = controller.get_tile(self.my_team, *loc)
            plate_count += getattr(t, 'num_clean_plates', 0)
            
        for loc in self.sinks:
             t = controller.get_tile(self.my_team, *loc)
             plate_count += getattr(t, 'num_dirty_plates', 0)

        for loc in self.counters:
            t = controller.get_tile(self.my_team, *loc)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan): pan_count += 1
            if isinstance(item, Plate): plate_count += 1
            
        # Count Held Assets
        for bid in bots:
            h = controller.get_bot_state(bid).get('holding')
            if h:
                if h['type'] == 'Pan': pan_count += 1
                elif h['type'] == 'Plate': plate_count += 1

        # --- CRITICAL SETUP PHASE ---
        # FIX: Removed the 'if money >= 10' check.
        # Bots will now walk to shops even if broke, and wait for passive income.
        if pan_count < 1: 
            jobs.append(Job(JobType.BUY_PAN, priority=2000))
        elif pan_count < len(self.cookers):
                jobs.append(Job(JobType.BUY_PAN, priority=40))

        if plate_count < 2: 
            jobs.append(Job(JobType.BUY_PLATE, priority=1900))

        # 1. EMERGENCY (Save Burnt Food)
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL + 10))

        # 2. STATE-TRANSITION (Finish what you hold)
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            if not bot: continue
            holding = bot.get('holding')
            if not holding: continue

            h_type = holding.get('type')
            if h_type == 'Food':
                name = holding.get('food_name', '').upper()
                is_chopped = holding.get('chopped', False) # API v2
                stage = holding.get('cooked_stage', 0)
                
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    jobs.append(Job(JobType.CHOP, priority=P_STATE_FIX)) 
                elif (name == 'MEAT' and is_chopped and stage == 0) or (name == 'EGG' and stage == 0):
                    if pan_count > 0:
                        jobs.append(Job(JobType.START_COOK, priority=P_STATE_FIX))
                    else:
                        jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_CRITICAL))
                else:
                    jobs.append(Job(JobType.ADD_TO_PLATE, priority=P_STATE_FIX))

            elif h_type == 'Plate':
                if len(holding.get('food', [])) > 0:
                    jobs.append(Job(JobType.SUBMIT, priority=P_SUBMIT))
                else:
                     jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX - 50))
            
            elif h_type == 'Pan':
                jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX))

        # 3. WORLD STATE SCAN
        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if not t: continue
            item = getattr(t, 'item', None)
            
            if isinstance(item, Food):
                name = getattr(item, 'food_name', '').upper()
                is_chopped = getattr(item, 'chopped', False)
                
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    jobs.append(Job(JobType.CHOP, target=(cx, cy), priority=P_WORLD_STATE))
                elif name == 'MEAT' and is_chopped:
                    if pan_count > 0:
                        jobs.append(Job(JobType.START_COOK, priority=P_WORLD_STATE))
            
            elif isinstance(item, Plate):
                foods = getattr(item, 'food', [])
                if len(foods) >= 1:
                    jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=P_SUBMIT))

        # 4. SABOTAGE (Phase 2 Only)
        switch_info = controller.get_switch_info()
        if switch_info['my_team_switched']:
            for st in self.sink_tables:
                jobs.append(Job(JobType.STEAL_PLATE, target=st, priority=P_SABOTAGE))
            for ck in self.cookers:
                jobs.append(Job(JobType.STEAL_PAN, target=ck, priority=P_SABOTAGE - 50))
            return jobs

        # 5. ORDER FULFILLMENT
        kitchen_ready = (pan_count > 0 and plate_count > 0)
        
        # Only buy ingredients if we can afford them AND have tools
        if kitchen_ready:
            counters_full = (self.empty_counters == 0)
            
            if not counters_full:
                for order in self.active_orders:
                    needed_for_order = Counter()
                    for req in order['required']: needed_for_order[req.upper()] += 1
                    
                    prio = P_LOGISTICS + (order['reward'] // 100)
                    for item, count in needed_for_order.items():
                        have = self.global_supply[item]
                        needed = count - have
                        
                        if needed > 0:
                            ft = self._name_to_foodtype(item)
                            if ft:
                                cost = self._get_item_cost(item)
                                # Strict Money Check for Ingredients (Don't buy if broke)
                                if self.current_money >= cost:
                                    for _ in range(needed):
                                        jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=prio))
                                        self.global_supply[item] += 1
                                        self.current_money -= cost 
                                else:
                                    break

        # 6. MAINTENANCE
        clean_plates = sum(getattr(controller.get_tile(self.my_team, *s), 'num_clean_plates', 0) for s in self.sink_tables)
        if clean_plates < 2:
            for s in self.sinks: jobs.append(Job(JobType.WASH, target=s, priority=P_MAINTENANCE))
        if clean_plates > 0:
            for st in self.sink_tables: jobs.append(Job(JobType.TAKE_CLEAN_PLATE, target=st, priority=P_MAINTENANCE + 10))
            
        jobs.append(Job(JobType.IDLE, priority=P_IDLE))
        
        return jobs

    # -------------------------------------------------------------------------
    # HIVE MIND: TASK ASSIGNMENT
    # -------------------------------------------------------------------------
    def assign_tasks(self, controller, jobs):
        bots = controller.get_team_bot_ids(self.my_team)
        if not bots or not jobs: return {}
        
        cost_matrix = np.full((len(bots), len(jobs)), float(COST_IMPOSSIBLE))
        
        for r, bid in enumerate(bots):
            bot = controller.get_bot_state(bid)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            
            for c, job in enumerate(jobs):
                # 1. Distance Cost
                target = job.target
                if not target:
                    if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
                    elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
                    elif job.job_type == JobType.START_COOK: target = self.get_nearest(pos, self.cookers)
                    elif job.job_type == JobType.ADD_TO_PLATE: target = self.get_nearest(pos, self.counters)
                    elif job.job_type == JobType.PLACE_ON_COUNTER: target = self._find_empty(controller, self.counters, pos)
                
                dist = self.get_dist(pos, target) if target else 0
                
                # 2. State Guard
                penalty = 0
                jt = job.job_type
                
                if jt == JobType.CHOP:
                    if holding and holding.get('type') == 'Plate': penalty = COST_HIGH
                    elif holding and holding.get('type') == 'Food': penalty = -200 
                elif jt == JobType.START_COOK:
                    if not holding or holding.get('type') != 'Food': penalty = COST_HIGH
                    else: penalty = -500
                elif jt in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.TAKE_CLEAN_PLATE]:
                    if holding: penalty = COST_HIGH
                elif jt == JobType.SUBMIT:
                    if not holding or holding.get('type') != 'Plate': penalty = COST_HIGH

                cost_matrix[r, c] = dist - job.priority + penalty

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
        
        while queue:
            curr, path = queue.popleft()
            
            if self.get_dist(curr, target) <= 1:
                if not path: return (0,0)
                next_node = (start[0]+path[0][0], start[1]+path[0][1])
                self.reserved_nodes.add((next_node[0], next_node[1], turn+1))
                return path[0]
            
            if len(path) > 10: continue
            
            moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
            for dx, dy in moves:
                nx, ny = curr[0]+dx, curr[1]+dy
                if not (0<=nx<self.map.width and 0<=ny<self.map.height): continue
                if (nx, ny) not in self.walkable: continue
                if (nx, ny, turn+len(path)+1) in self.reserved_nodes: continue
                
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
        needs_empty = job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE, 
                                       JobType.TAKE_FROM_PAN, JobType.STEAL_PAN, JobType.TAKE_CLEAN_PLATE]
        if needs_empty and holding:
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
            elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
            elif job.job_type == JobType.START_COOK: target = self._find_empty_pan(controller, self.cookers, pos)
            elif job.job_type == JobType.ADD_TO_PLATE: target = self._find_plate(controller, self.counters, pos)
            elif job.job_type == JobType.SUBMIT: target = self.get_nearest(pos, self.submits)
            elif job.job_type == JobType.TAKE_CLEAN_PLATE: target = self.get_nearest(pos, self.sink_tables)
        
        if not target: return 
        
        # 3. ACT or MOVE
        if self.get_dist(pos, target) <= 1:
            self._perform_action(controller, bot_id, job, target, holding)
        else:
            self.move_bot(controller, bot_id, target)

    def move_bot(self, controller, bot_id, target):
        dx, dy = self.get_move(controller, bot_id, target, controller.get_turn())
        controller.move(bot_id, dx, dy)

    def _perform_action(self, controller, bot_id, job, target, holding):
        tx, ty = target
        jt = job.job_type
        
        if jt == JobType.BUY_INGREDIENT: 
            if not holding: controller.buy(bot_id, job.item, tx, ty)
        elif jt == JobType.BUY_PAN: 
            if not holding: controller.buy(bot_id, ShopCosts.PAN, tx, ty)
        elif jt == JobType.BUY_PLATE: 
            if not holding: controller.buy(bot_id, ShopCosts.PLATE, tx, ty)
            
        elif jt == JobType.CHOP:
            if holding: controller.place(bot_id, tx, ty)
            else: controller.chop(bot_id, tx, ty)
            
        elif jt == JobType.START_COOK:
            if holding: controller.place(bot_id, tx, ty)
            
        elif jt == JobType.TAKE_FROM_PAN: 
            if not holding: controller.take_from_pan(bot_id, tx, ty)
            
        elif jt == JobType.ADD_TO_PLATE:
            if holding: controller.add_food_to_plate(bot_id, tx, ty)
            
        elif jt == JobType.SUBMIT: 
            if holding and holding.get('type') == 'Plate': controller.submit(bot_id, tx, ty)
            
        elif jt == JobType.PLACE_ON_COUNTER: 
            if holding: controller.place(bot_id, tx, ty)
            
        elif jt == JobType.WASH:
            tile = controller.get_tile(self.my_team, tx, ty)
            if getattr(tile, 'num_dirty_plates', 0) > 0: controller.wash_sink(bot_id, tx, ty)
                
        elif jt == JobType.TAKE_CLEAN_PLATE:
            if not holding: controller.take_clean_plate(bot_id, tx, ty)
            
        elif jt == JobType.PICKUP_PLATE:
            if not holding: controller.pickup(bot_id, tx, ty)
            
        elif jt == JobType.STEAL_PLATE:
            if holding: 
                tr = self.get_nearest((tx, ty), self.trashes)
                controller.trash(bot_id, tr[0], tr[1])
            else: controller.take_clean_plate(bot_id, tx, ty)
                
        elif jt == JobType.STEAL_PAN:
            if not holding: controller.pickup(bot_id, tx, ty)

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
        sorted_locs = sorted(locs, key=lambda x: self.get_dist(pos, x))
        for l in sorted_locs:
            if not getattr(c.get_tile(self.my_team, *l), 'item', None): return l
        return sorted_locs[0] if sorted_locs else None

    def _find_empty_pan(self, c, locs, pos):
        for l in sorted(locs, key=lambda x: self.get_dist(pos, x)):
            it = getattr(c.get_tile(self.my_team, *l), 'item', None)
            if isinstance(it, Pan) and not it.food: return l
        return None

    def _find_plate(self, c, locs, pos):
        for l in sorted(locs, key=lambda x: self.get_dist(pos, x)):
            it = getattr(c.get_tile(self.my_team, *l), 'item', None)
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
        
    def _get_item_cost(self, name):
        name = name.upper()
        if name == 'MEAT': return 80
        if name == 'NOODLES': return 40
        if name == 'ONION' or name == 'ONIONS': return 30
        if name == 'EGG': return 20
        if name == 'SAUCE': return 10
        return 9999

    # -------------------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------------------
    def play_turn(self, controller: RobotController):
        self.initialize(controller)
        self.update_state(controller)
        self.reserved_nodes.clear()
        
        turn = controller.get_turn()
        if 250 <= turn < 350 and not controller.get_switch_info()['my_team_switched']:
            controller.switch_maps()

        jobs = self.generate_jobs(controller)
        assigns = self.assign_tasks(controller, jobs)
        
        sorted_bots = sorted(assigns.keys(), key=lambda b: assigns[b].priority, reverse=True)
        
        start_ts = time.time()
        for bid in sorted_bots:
            if time.time() - start_ts > 0.45: break 
            self.execute(controller, bid, assigns[bid])