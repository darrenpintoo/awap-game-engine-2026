"""
APEX CHEF v4.2
Codename: "Restored Vision"
Strategy: Dynamic Pipeline + Hallway Relay + API v3 Compliance
Fixes: Restored missing 'update_state' method that caused the crash.
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
P_CRITICAL     = 1000  
P_PARASITE     = 900   
P_STATE_FIX    = 800   
P_HANDOFF      = 750   
P_SUBMIT       = 600   
P_COOK_STEP    = 500   
P_LOGISTICS    = 100   
P_MAINTENANCE  = 50    
P_IDLE         = 0

# Penalties
COST_ZONE_VIOLATION = 2000 
COST_IMPOSSIBLE     = 99999
COST_HIGH           = 5000

class JobType(Enum):
    SAVE_FOOD = auto()
    FINISH_ITEM = auto()
    HOARD_PLATE = auto()
    
    SUBMIT = auto()
    BUY_INGREDIENT = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    PLACE_ON_COUNTER = auto()
    CHOP = auto()
    START_COOK = auto()
    TAKE_FROM_PAN = auto()
    ADD_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    TAKE_CLEAN_PLATE = auto()
    WASH = auto()
    IDLE = auto()
    
    STEAL_PLATE = auto()
    STEAL_PAN = auto()
    
    WAIT_AT_DOOR = auto()

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
        
        # Map Features
        self.shops = []
        self.cookers = []
        self.sinks = []
        self.sink_tables = []
        self.counters = [] 
        self.submits = []
        self.trashes = []
        
        # Topology
        self.bot_capabilities = {} 
        self.shared_counters = []  
        self.kitchen_zone = set()
        self.handoff_spot = None
        self.is_cramped_map = False
        
        # State
        self.cooking_info = {}
        self.active_orders = []
        self.global_supply = Counter()
        self.current_money = 0
        self.my_team = None
        self.empty_counters = 0
        self.is_invading = False
        
        # Strategies
        self.bot_roles = {} 
        self.kitchen_token_holder = None 

    # -------------------------------------------------------------------------
    # INITIALIZATION & TOPOLOGY
    # -------------------------------------------------------------------------
    def initialize(self, controller: RobotController):
        if self.initialized: return
        
        self.my_team = controller.get_team()
        m = controller.get_map(self.my_team) 
        
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

        self._build_distance_matrix(m)
        self._analyze_topology(controller)
        
        self.active_orders = [o for o in controller.get_orders(self.my_team) if o.get('is_active')]
        if self.is_cramped_map:
            print("[APEX v4.2] Mode: RELAY")
            ids = controller.get_team_bot_ids(self.my_team)
            if ids: self.kitchen_token_holder = ids[0]
        else:
            print("[APEX v4.2] Mode: PIPELINE")
            self._calculate_optimal_split(controller)
        
        self.initialized = True

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

    def _analyze_topology(self, controller):
        if not self.cookers: return
        
        start_node = self.cookers[0]
        queue = deque([(start_node, 0)])
        visited = {start_node}
        
        while queue:
            curr, dist = queue.popleft()
            if dist > 3: continue
            self.kitchen_zone.add(curr)
            
            if curr in self.dist_matrix:
                for neighbor in self.dist_matrix[curr]:
                    if self.dist_matrix[curr][neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        total_neighbors = 0
        kitchen_walkable = [t for t in self.kitchen_zone if t in self.walkable]
        
        if not kitchen_walkable: return
        
        for tile in kitchen_walkable:
            nb = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx==0 and dy==0: continue
                    nx, ny = tile[0]+dx, tile[1]+dy
                    if (nx, ny) in self.walkable: nb += 1
            total_neighbors += nb
            
        avg_neighbors = total_neighbors / len(kitchen_walkable)
        if avg_neighbors < 3.5:
            self.is_cramped_map = True
            
        candidates = []
        for t in self.walkable:
            if t not in self.kitchen_zone:
                d = self.get_dist(t, self.cookers[0])
                candidates.append((d, t))
        
        if candidates:
            candidates.sort()
            self.handoff_spot = candidates[0][1]

    # -------------------------------------------------------------------------
    # STATE ANALYSIS (RESTORED)
    # -------------------------------------------------------------------------
    def update_state(self, controller):
        self.my_team = controller.get_team()
        
        # Switch Status
        switch_info = controller.get_switch_info()
        self.is_invading = switch_info['has_switched'] # Fixed key for v3 API
        
        # Money & Orders
        self.current_money = controller.get_team_money(self.my_team)
        self.active_orders = [o for o in controller.get_orders(self.my_team) if o.get('is_active')]
        
        # Cooking Info
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            t = controller.get_tile(self.my_team, kx, ky)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan) and item.food:
                self.cooking_info[(kx, ky)] = CookingInfo(
                    (kx, ky), item.food.food_name, 
                    getattr(t, 'cook_progress', 0), item.food.cooked_stage
                )

        # Global Supply Audit
        self.global_supply.clear()
        self.empty_counters = 0
        
        # Count from Bots
        for bid in controller.get_team_bot_ids(self.my_team):
            h = controller.get_bot_state(bid).get('holding')
            if h and h.get('type') == 'Food':
                self.global_supply[h['food_name']] += 1
                
        # Count from Counters
        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if t and t.item:
                if isinstance(t.item, Food):
                    self.global_supply[t.item.food_name] += 1
            else:
                self.empty_counters += 1
                
        # Count from Pans
        for info in self.cooking_info.values():
            self.global_supply[info.food_name] += 1

    # -------------------------------------------------------------------------
    # JOB GENERATION (RELAY MODE)
    # -------------------------------------------------------------------------
    def generate_jobs_relay(self, controller) -> List[Job]:
        jobs = []
        bots = controller.get_team_bot_ids(self.my_team)
        holder = self.kitchen_token_holder
        
        # Holder Logic (Inside)
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL + 10))

        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if not t: continue
            item = getattr(t, 'item', None)
            if isinstance(item, Food):
                name = getattr(item, 'food_name', '').upper()
                is_chopped = getattr(item, 'chopped', False)
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    jobs.append(Job(JobType.CHOP, target=(cx, cy), priority=P_COOK_STEP))
                elif name == 'MEAT' and is_chopped:
                    jobs.append(Job(JobType.START_COOK, priority=P_COOK_STEP))

        jobs.append(Job(JobType.WASH, priority=P_MAINTENANCE))

        # Outsider Logic (Outside)
        for order in self.active_orders:
            needed = Counter()
            for req in order['required']: needed[req.upper()] += 1
            for item, count in needed.items():
                if self.global_supply[item] < count:
                    ft = self._name_to_foodtype(item)
                    if ft:
                        cost = self._get_item_cost(item)
                        if self.current_money >= cost:
                            jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=P_LOGISTICS))
                            self.global_supply[item] += 1
                            self.current_money -= cost

        return jobs

    # -------------------------------------------------------------------------
    # ASSIGNMENT (RELAY)
    # -------------------------------------------------------------------------
    def assign_tasks_relay(self, controller, jobs):
        bots = controller.get_team_bot_ids(self.my_team)
        assignments = {}
        holder = self.kitchen_token_holder
        
        cost_matrix = np.full((len(bots), len(jobs)), float(COST_IMPOSSIBLE))
        
        for r, bid in enumerate(bots):
            bot = controller.get_bot_state(bid)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            is_holder = (bid == holder)
            
            for c, job in enumerate(jobs):
                target = job.target
                if not target:
                    if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
                    elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
                    elif job.job_type == JobType.START_COOK: target = self.get_nearest(pos, self.cookers)
                    elif job.job_type == JobType.ADD_TO_PLATE: target = self.get_nearest(pos, self.counters)
                    elif job.job_type == JobType.PLACE_ON_COUNTER: 
                        if is_holder: target = self._find_empty(controller, self.counters, pos)
                        else: target = self.handoff_spot 
                    elif job.job_type == JobType.TAKE_CLEAN_PLATE: target = self.get_nearest(pos, self.sink_tables)
                    elif job.job_type == JobType.WASH: target = self.get_nearest(pos, self.sinks)
                
                dist = self.get_dist(pos, target) if target else 0
                
                zone_penalty = 0
                target_in_kitchen = (target in self.kitchen_zone) if target else False
                
                if not is_holder:
                    if target_in_kitchen: zone_penalty = COST_IMPOSSIBLE
                    
                if not is_holder and holding and job.job_type == JobType.PLACE_ON_COUNTER:
                    zone_penalty = -500 
                    
                state_penalty = 0
                jt = job.job_type
                if jt == JobType.CHOP:
                    if holding and holding.get('type') == 'Plate': state_penalty = COST_HIGH
                elif jt == JobType.START_COOK:
                    if not holding or holding.get('type') != 'Food': state_penalty = COST_HIGH
                elif jt in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.TAKE_CLEAN_PLATE]:
                    if holding: state_penalty = COST_HIGH
                elif jt == JobType.SUBMIT:
                    if not holding or holding.get('type') != 'Plate': state_penalty = COST_HIGH

                cost_matrix[r, c] = dist - job.priority + zone_penalty + state_penalty

        if HAS_SCIPY:
            row, col = linear_sum_assignment(cost_matrix)
        else:
            col = np.argmin(cost_matrix, axis=1)
            row = range(len(bots))

        assignments = {bots[r]: jobs[c] for r, c in zip(row, col)}
        
        holder_job = assignments.get(holder)
        if holder_job:
            target = holder_job.target
            if target and target not in self.kitchen_zone:
                # Token swap logic would go here
                pass
                
        return assignments

    # -------------------------------------------------------------------------
    # PIPELINE LOGIC (Open Maps)
    # -------------------------------------------------------------------------
    def _calculate_optimal_split(self, controller):
        OPS = ['BUY', 'CHOP', 'COOK', 'PLATE', 'SUBMIT']
        costs = {'BUY': 5, 'CHOP': 10, 'COOK': 20, 'PLATE': 5, 'SUBMIT': 5}
        
        best_score = float('inf')
        best_split = None
        bots = controller.get_team_bot_ids(self.my_team)
        
        if len(bots) < 2:
            self._apply_roles(bots[0], set(OPS))
            return

        for i in range(1, 32): 
            set_a = set()
            set_b = set()
            for idx, op in enumerate(OPS):
                if (i >> idx) & 1: set_a.add(op)
                else: set_b.add(op)
            
            if not self._is_feasible(bots[0], set_a) or not self._is_feasible(bots[1], set_b): continue
            
            needs_handoff = False
            if 'BUY' in set_a and ('CHOP' in set_b or 'COOK' in set_b): needs_handoff = True
            
            if needs_handoff and not self.shared_counters: continue 
            
            load_a = len(set_a) * 10 
            load_b = len(set_b) * 10
            score = max(load_a, load_b)
            if score < best_score:
                best_score = score
                best_split = (set_a, set_b)
        
        if not best_split:
            best_split = (set(OPS), set()) # Fallback
                
        self._apply_roles(bots[0], best_split[0])
        self._apply_roles(bots[1], best_split[1])

    def _is_feasible(self, bot_id, ops):
        caps = self.bot_capabilities.get(bot_id, {})
        for op in ops:
            if not caps.get(op, False): return False
        return True

    def _apply_roles(self, bot_id, op_set):
        allowed = set()
        if 'BUY' in op_set: allowed.update([JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE])
        if 'CHOP' in op_set: allowed.add(JobType.CHOP)
        if 'COOK' in op_set: allowed.update([JobType.START_COOK, JobType.TAKE_FROM_PAN])
        if 'PLATE' in op_set: allowed.update([JobType.ADD_TO_PLATE, JobType.PICKUP_PLATE, JobType.TAKE_CLEAN_PLATE, JobType.PLACE_ON_COUNTER])
        if 'SUBMIT' in op_set: allowed.add(JobType.SUBMIT)
        allowed.update([JobType.WASH, JobType.IDLE, JobType.SAVE_FOOD, JobType.FINISH_ITEM])
        allowed.add(JobType.HOARD_PLATE)
        allowed.add(JobType.STEAL_PLATE)
        allowed.add(JobType.STEAL_PAN)
        self.bot_roles[bot_id] = allowed

    # -------------------------------------------------------------------------
    # COMMON GENERATE/EXECUTE
    # -------------------------------------------------------------------------
    def generate_jobs(self, controller):
        if self.is_cramped_map:
            return self.generate_jobs_relay(controller) 
        else:
            return self.generate_jobs_standard(controller)

    def generate_jobs_standard(self, controller):
        jobs = []
        bots = controller.get_team_bot_ids(self.my_team)
        
        pan_count = 0
        plate_count = 0
        for bid in bots:
            h = controller.get_bot_state(bid).get('holding')
            if h:
                if h['type'] == 'Pan': pan_count += 1
                elif h['type'] == 'Plate': plate_count += 1
        for loc in self.cookers + self.sink_tables + self.sinks + self.counters:
            t = controller.get_tile(self.my_team, *loc)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan): pan_count += 1
            elif isinstance(item, Plate): plate_count += 1
            if hasattr(t, 'num_clean_plates'): plate_count += t.num_clean_plates
            if hasattr(t, 'num_dirty_plates'): plate_count += t.num_dirty_plates

        # Costs: Pan=$4, Plate=$2
        if pan_count < 1: jobs.append(Job(JobType.BUY_PAN, priority=2000))
        elif pan_count < len(self.cookers): jobs.append(Job(JobType.BUY_PAN, priority=40))
        if plate_count < 2: jobs.append(Job(JobType.BUY_PLATE, priority=1900))

        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL + 10))

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
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    jobs.append(Job(JobType.CHOP, priority=P_STATE_FIX)) 
                elif (name == 'MEAT' and is_chopped and stage == 0) or (name == 'EGG' and stage == 0):
                    if pan_count > 0: jobs.append(Job(JobType.START_COOK, priority=P_STATE_FIX))
                    else: jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_CRITICAL))
                else:
                    jobs.append(Job(JobType.ADD_TO_PLATE, priority=P_STATE_FIX))
            elif h_type == 'Plate':
                if len(holding.get('food', [])) > 0:
                    if self.is_invading: jobs.append(Job(JobType.HOARD_PLATE, priority=P_PARASITE))
                    else: jobs.append(Job(JobType.SUBMIT, priority=P_SUBMIT))
                else: jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX - 50))
            elif h_type == 'Pan':
                jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX))

        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if not t: continue
            item = getattr(t, 'item', None)
            if isinstance(item, Food):
                name = getattr(item, 'food_name', '').upper()
                is_chopped = getattr(item, 'chopped', False)
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    jobs.append(Job(JobType.CHOP, target=(cx, cy), priority=P_COOK_STEP))
                elif name == 'MEAT' and is_chopped:
                    if pan_count > 0: jobs.append(Job(JobType.START_COOK, priority=P_COOK_STEP))
            elif isinstance(item, Plate):
                foods = getattr(item, 'food', [])
                if len(foods) >= 1:
                     if self.is_invading: jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=P_PARASITE))
                     else: jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=P_SUBMIT))

        switch_info = controller.get_switch_info()
        if switch_info['has_switched']:
            for st in self.sink_tables: jobs.append(Job(JobType.STEAL_PLATE, target=st, priority=P_SABOTAGE))
            for ck in self.cookers: jobs.append(Job(JobType.STEAL_PAN, target=ck, priority=P_SABOTAGE - 50))
            return jobs

        kitchen_ready = (pan_count > 0 and plate_count > 0)
        if self.current_money > 0 and kitchen_ready:
            if self.empty_counters > 0:
                for order in self.active_orders:
                    needed = Counter()
                    for req in order['required']: needed[req.upper()] += 1
                    prio = P_LOGISTICS + (order['reward'] // 100)
                    for item, count in needed.items():
                        if self.global_supply[item] < count:
                            ft = self._name_to_foodtype(item)
                            if ft:
                                cost = self._get_item_cost(item)
                                if self.current_money >= cost:
                                    jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=prio))
                                    self.global_supply[item] += 1
                                    self.current_money -= cost

        clean_plates = sum(getattr(controller.get_tile(self.my_team, *s), 'num_clean_plates', 0) for s in self.sink_tables)
        if clean_plates < 2:
            for s in self.sinks: jobs.append(Job(JobType.WASH, target=s, priority=P_MAINTENANCE))
        if clean_plates > 0:
            for st in self.sink_tables: jobs.append(Job(JobType.TAKE_CLEAN_PLATE, target=st, priority=P_MAINTENANCE + 10))
            
        jobs.append(Job(JobType.IDLE, priority=P_IDLE))
        return jobs

    def assign_tasks_standard(self, controller, jobs):
        bots = controller.get_team_bot_ids(self.my_team)
        if not bots or not jobs: return {}
        
        cost_matrix = np.full((len(bots), len(jobs)), float(COST_IMPOSSIBLE))
        
        for r, bid in enumerate(bots):
            bot = controller.get_bot_state(bid)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            allowed_jobs = self.bot_roles.get(bid, set())
            
            for c, job in enumerate(jobs):
                target = job.target
                if not target:
                    if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
                    elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
                    elif job.job_type == JobType.START_COOK: target = self.get_nearest(pos, self.cookers)
                    elif job.job_type == JobType.ADD_TO_PLATE: target = self.get_nearest(pos, self.counters)
                    elif job.job_type == JobType.PLACE_ON_COUNTER: 
                        if job.priority == P_HANDOFF and self.shared_counters:
                            target = self._find_empty(controller, self.shared_counters, pos)
                        else:
                            target = self._find_empty(controller, self.counters, pos)
                    elif job.job_type == JobType.TAKE_CLEAN_PLATE: target = self.get_nearest(pos, self.sink_tables)
                    elif job.job_type == JobType.WASH: target = self.get_nearest(pos, self.sinks)
                
                dist = self.get_dist(pos, target) if target else 0
                if dist >= 9999: dist = COST_IMPOSSIBLE

                role_penalty = 0
                if job.job_type not in allowed_jobs and job.priority < P_STATE_FIX:
                    role_penalty = COST_ZONE_VIOLATION

                state_penalty = 0
                jt = job.job_type
                if jt == JobType.CHOP:
                    if holding and holding.get('type') == 'Plate': state_penalty = COST_HIGH
                elif jt == JobType.START_COOK:
                    if not holding or holding.get('type') != 'Food': state_penalty = COST_HIGH
                elif jt in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.TAKE_CLEAN_PLATE]:
                    if holding: state_penalty = COST_HIGH
                elif jt == JobType.SUBMIT:
                    if not holding or holding.get('type') != 'Plate': state_penalty = COST_HIGH

                cost_matrix[r, c] = dist - job.priority + role_penalty + state_penalty

        if HAS_SCIPY:
            row, col = linear_sum_assignment(cost_matrix)
        else:
            col = np.argmin(cost_matrix, axis=1)
            row = range(len(bots))

        return {bots[r]: jobs[c] for r, c in zip(row, col)}

    def assign_tasks(self, controller, jobs):
        if self.is_cramped_map:
            return self.assign_tasks_relay(controller, jobs)
        else:
            return self.assign_tasks_standard(controller, jobs)

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

    def execute(self, controller, bot_id, job):
        bot = controller.get_bot_state(bot_id)
        pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        if job.job_type == JobType.HOARD_PLATE: return 

        needs_empty = job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE, 
                                       JobType.TAKE_FROM_PAN, JobType.STEAL_PAN, JobType.TAKE_CLEAN_PLATE]
        if needs_empty and holding:
            trash = self.get_nearest(pos, self.trashes)
            if self.get_dist(pos, trash) <= 1: 
                controller.trash(bot_id, trash[0], trash[1])
            else: 
                self.move_bot(controller, bot_id, trash)
            return

        target = job.target
        if not target:
            if job.job_type == JobType.BUY_INGREDIENT: target = self.get_nearest(pos, self.shops)
            elif job.job_type == JobType.PLACE_ON_COUNTER: 
                if job.priority == P_HANDOFF and self.shared_counters:
                    target = self._find_empty(controller, self.shared_counters, pos)
                else:
                    target = self._find_empty(controller, self.counters, pos)
            elif job.job_type == JobType.CHOP: target = self.get_nearest(pos, self.counters) 
            elif job.job_type == JobType.START_COOK: target = self._find_empty_pan(controller, self.cookers, pos)
            elif job.job_type == JobType.ADD_TO_PLATE: target = self._find_plate(controller, self.counters, pos)
            elif job.job_type == JobType.SUBMIT: target = self.get_nearest(pos, self.submits)
            elif job.job_type == JobType.TAKE_CLEAN_PLATE: target = self.get_nearest(pos, self.sink_tables)
            elif job.job_type == JobType.WASH: target = self.get_nearest(pos, self.sinks)
        
        if not target: return 
        
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

    def get_dist(self, a, b):
        if a in self.dist_matrix and b in self.dist_matrix[a]: return self.dist_matrix[a][b]
        return 9999

    def get_nearest(self, pos, locs):
        if not locs: return None
        valid_locs = [l for l in locs if self.get_dist(pos, l) < 9999]
        if not valid_locs: return None
        return min(valid_locs, key=lambda l: self.get_dist(pos, l))

    def _find_empty(self, c, locs, pos):
        valid_locs = [l for l in locs if self.get_dist(pos, l) < 9999]
        sorted_locs = sorted(valid_locs, key=lambda x: self.get_dist(pos, x))
        for l in sorted_locs:
            if not getattr(c.get_tile(self.my_team, *l), 'item', None): return l
        return sorted_locs[0] if sorted_locs else None

    def _find_empty_pan(self, c, locs, pos):
        valid_locs = [l for l in locs if self.get_dist(pos, l) < 9999]
        for l in sorted(valid_locs, key=lambda x: self.get_dist(pos, x)):
            it = getattr(c.get_tile(self.my_team, *l), 'item', None)
            if isinstance(it, Pan) and not it.food: return l
        return None

    def _find_plate(self, c, locs, pos):
        valid_locs = [l for l in locs if self.get_dist(pos, l) < 9999]
        for l in sorted(valid_locs, key=lambda x: self.get_dist(pos, x)):
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

    def play_turn(self, controller: RobotController):
        self.initialize(controller)
        self.update_state(controller)
        self.reserved_nodes.clear()
        
        turn = controller.get_turn()
        if 250 <= turn < 350 and not controller.get_switch_info()['has_switched']:
            enemy_gone = controller.get_switch_info()['enemy_has_switched']
            if enemy_gone: controller.switch_maps()

        jobs = self.generate_jobs(controller)
        assigns = self.assign_tasks(controller, jobs)
        
        # Token Logic for Relay
        if self.is_cramped_map:
            holder = self.kitchen_token_holder
            holder_job = assigns.get(holder)
            leaving = False
            if holder_job and holder_job.target and holder_job.target not in self.kitchen_zone:
                leaving = True
            
            if leaving or not holder_job:
                bots = controller.get_team_bot_ids(self.my_team)
                others = [b for b in bots if b != holder]
                if others:
                    # Handoff distance check could go here
                    self.kitchen_token_holder = others[0]

        sorted_bots = sorted(assigns.keys(), key=lambda b: assigns[b].priority, reverse=True)
        
        start_ts = time.time()
        for bid in sorted_bots:
            if time.time() - start_ts > 0.45: break 
            self.execute(controller, bid, assigns[bid])