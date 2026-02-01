"""
APEX CHEF v6.0
Codename: "Fluid Dynamics"
Strategy: Pure Cost-Matrix Optimization + Topology Anchors + Relay Fallback
Changes: Removed rigid roles. Added detailed position logging. Defined specific counters for Plating vs Handoff.
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
from tiles import Box, Counter as TileCounter, Sink, SinkTable, Cooker, Shop

# ============================================
# CONFIGURATION
# ============================================

DEBUG = True  

# Priority Hierarchy (Granular)
P_CRITICAL     = 1000  # Save burning food
P_PARASITE     = 900   # Invasion Cooking
P_STATE_FIX    = 800   # I am holding X, I must process X
P_HANDOFF      = 750   # Clear the passing counter
P_SUBMIT       = 600   # Cash in
P_COOK_STEP    = 500   # Active cooking
P_LOGISTICS    = 100   # Buying
P_MAINTENANCE  = 50    # Washing
P_IDLE         = 0

# Penalties
COST_IMPOSSIBLE     = 99999
COST_HIGH           = 5000
COST_MEDIUM         = 1000

class JobType(Enum):
    SAVE_FOOD = auto()
    HOARD_PLATE = auto()
    
    SUBMIT = auto()
    BUY_INGREDIENT = auto()
    BUY_PAN = auto()
    BUY_PLATE = auto()
    
    # Specific Locations
    DROP_AT_HANDOFF = auto() # Outsider drops here
    PICKUP_FROM_HANDOFF = auto() # Insider picks up
    
    PLACE_ON_COUNTER = auto() # Generic
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

@dataclass
class Job:
    job_type: JobType
    target: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    priority: int = 0
    
    def __repr__(self):
        t_str = f"@{self.target}" if self.target else ""
        i_str = f"({self.item})" if self.item else ""
        return f"[{self.job_type.name}{t_str}{i_str} P:{self.priority}]"

@dataclass 
class CookingInfo:
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int

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
        self.kitchen_zone = set()
        self.is_cramped_map = False
        
        # Smart Anchors
        self.handoff_spot = None # Counter nearest to exit
        self.plating_spot = None # Counter nearest to cookers
        
        # State
        self.turn = 0
        self.cooking_info = {}
        self.active_orders = []
        self.global_supply = Counter()
        self.current_money = 0
        self.my_team = None
        self.empty_counters = 0
        self.is_invading = False
        
        # Relay Token
        self.kitchen_token_holder = None 
        
        # Logging
        self.log_file = None

    def log(self, msg):
        if DEBUG and self.log_file:
            entry = f"[T{self.turn}] {msg}"
            try:
                self.log_file.write(entry + "\n")
                self.log_file.flush()
            except: pass

    # -------------------------------------------------------------------------
    # INITIALIZATION & TOPOLOGY
    # -------------------------------------------------------------------------
    def initialize(self, controller: RobotController):
        if self.initialized: return
        
        self.my_team = controller.get_team()
        
        try:
            fname = f"bot_log_{self.my_team.name}.txt"
            self.log_file = open(fname, "w")
            self.log(f"--- GAME START: {self.my_team.name} ---")
        except: pass

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
            self.log("MODE: RELAY (Cramped Map)")
            ids = controller.get_team_bot_ids(self.my_team)
            if ids: self.kitchen_token_holder = ids[0]
        else:
            self.log("MODE: FLUID (Open Map)")
        
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
        
        # 1. Define Kitchen Zone
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

        # 2. Check Density
        kitchen_walkable = [t for t in self.kitchen_zone if t in self.walkable]
        
        if kitchen_walkable:
            total_neighbors = 0
            for tile in kitchen_walkable:
                nb = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx==0 and dy==0: continue
                        nx, ny = tile[0]+dx, tile[1]+dy
                        if (nx, ny) in self.walkable: nb += 1
                total_neighbors += nb
            avg = total_neighbors / len(kitchen_walkable)
            self.log(f"Avg Neighbors: {avg:.2f}")
            if avg < 3.5: self.is_cramped_map = True
            
        # 3. Identify Strategic Counters
        # Handoff: Closest to Shop/Start
        if self.counters and self.shops:
            shop_loc = self.shops[0]
            # Find counter min dist to shop
            self.handoff_spot = min(self.counters, key=lambda c: self.get_dist(shop_loc, c))
            self.log(f"Handoff Spot: {self.handoff_spot}")
            
        # Plating: Closest to Cooker
        if self.counters and self.cookers:
            cook_loc = self.cookers[0]
            self.plating_spot = min(self.counters, key=lambda c: self.get_dist(cook_loc, c))
            self.log(f"Plating Spot: {self.plating_spot}")

    # -------------------------------------------------------------------------
    # STATE
    # -------------------------------------------------------------------------
    def update_state(self, controller):
        self.my_team = controller.get_team()
        switch_info = controller.get_switch_info()
        self.is_invading = switch_info.get('my_team_switched', False)
        
        self.current_money = controller.get_team_money(self.my_team)
        self.active_orders = [o for o in controller.get_orders(self.my_team) if o.get('is_active')]
        
        self.cooking_info.clear()
        for kx, ky in self.cookers:
            t = controller.get_tile(self.my_team, kx, ky)
            item = getattr(t, 'item', None)
            if isinstance(item, Pan) and item.food:
                self.cooking_info[(kx, ky)] = CookingInfo(
                    (kx, ky), item.food.food_name, 
                    getattr(t, 'cook_progress', 0), item.food.cooked_stage
                )

        self.global_supply.clear()
        self.empty_counters = 0
        
        for bid in controller.get_team_bot_ids(self.my_team):
            h = controller.get_bot_state(bid).get('holding')
            if h and h.get('type') == 'Food':
                self.global_supply[h['food_name']] += 1
                
        for cx, cy in self.counters:
            t = controller.get_tile(self.my_team, cx, cy)
            if isinstance(t, Box):
                if t.item and isinstance(t.item, Food):
                    self.global_supply[t.item.food_name] += t.count
                if t.count == 0: self.empty_counters += 1
            else:
                if t.item:
                    if isinstance(t.item, Food):
                        self.global_supply[t.item.food_name] += 1
                else:
                    self.empty_counters += 1
                
        for info in self.cooking_info.values():
            self.global_supply[info.food_name] += 1

    # -------------------------------------------------------------------------
    # JOB GENERATION
    # -------------------------------------------------------------------------
    def generate_jobs(self, controller) -> List[Job]:
        """Generates all possible tasks based on current state."""
        jobs = []
        bots = controller.get_team_bot_ids(self.my_team)
        
        # 1. Tools Check
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

        # Critical Buys
        if pan_count < 1: jobs.append(Job(JobType.BUY_PAN, priority=2000))
        elif pan_count < len(self.cookers): jobs.append(Job(JobType.BUY_PAN, priority=40))
        if plate_count < 2: jobs.append(Job(JobType.BUY_PLATE, priority=1900))

        # 2. Cooking Management
        for loc, info in self.cooking_info.items():
            if info.cooked_stage == 1:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL))
            elif info.cooked_stage == 0 and info.turns_to_burned < 5:
                jobs.append(Job(JobType.TAKE_FROM_PAN, target=loc, priority=P_CRITICAL + 10))

        # 3. Holding State Processing
        # "I am holding X, therefore I should do Y"
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            if not bot: continue
            holding = bot.get('holding')
            if not holding: continue
            
            h_type = holding.get('type')
            if h_type == 'Food':
                name = holding.get('food_name', '').upper()
                is_chopped = holding.get('chopped', False)
                
                if name in ['MEAT', 'ONIONS'] and not is_chopped:
                    # Prefer Chop
                    jobs.append(Job(JobType.CHOP, priority=P_STATE_FIX))
                elif (name == 'MEAT' and is_chopped) or name == 'EGG':
                    # Prefer Cook
                    if pan_count > 0: jobs.append(Job(JobType.START_COOK, priority=P_STATE_FIX))
                    else: jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_CRITICAL))
                else:
                    # Ready to plate
                    jobs.append(Job(JobType.ADD_TO_PLATE, priority=P_STATE_FIX))
                    
            elif h_type == 'Plate':
                if len(holding.get('food', [])) > 0:
                    if self.is_invading: jobs.append(Job(JobType.HOARD_PLATE, priority=P_PARASITE))
                    else: jobs.append(Job(JobType.SUBMIT, priority=P_SUBMIT))
                else:
                    # Empty plate: put down near chef
                    jobs.append(Job(JobType.PLACE_ON_COUNTER, target=self.plating_spot, priority=P_HANDOFF))
                    
            elif h_type == 'Pan':
                jobs.append(Job(JobType.PLACE_ON_COUNTER, priority=P_STATE_FIX))

        # 4. Counter Scanning
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
                # Pick up plate to submit or add food
                foods = getattr(item, 'food', [])
                if len(foods) >= 1:
                     if self.is_invading: jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=P_PARASITE))
                     else: jobs.append(Job(JobType.PICKUP_PLATE, target=(cx, cy), priority=P_SUBMIT))

        # 5. Supply & Logistics
        for order in self.active_orders:
            needed = Counter()
            for req in order['required']: needed[req.upper()] += 1
            for item, count in needed.items():
                if self.global_supply[item] < count:
                    ft = self._name_to_foodtype(item)
                    if ft:
                        cost = self._get_item_cost(item)
                        if self.current_money >= cost:
                            # Buy job
                            jobs.append(Job(JobType.BUY_INGREDIENT, item=ft, priority=P_LOGISTICS))
                            self.global_supply[item] += 1
                            self.current_money -= cost

        jobs.append(Job(JobType.WASH, priority=P_MAINTENANCE))
        jobs.append(Job(JobType.IDLE, priority=P_IDLE))
        return jobs

    # -------------------------------------------------------------------------
    # ASSIGNMENT (THE FLUID MATRIX)
    # -------------------------------------------------------------------------
    def assign_tasks(self, controller, jobs):
        bots = controller.get_team_bot_ids(self.my_team)
        if not bots or not jobs: return {}
        
        cost_matrix = np.full((len(bots), len(jobs)), float(COST_IMPOSSIBLE))
        
        # Determine Relay Roles if needed
        holder = self.kitchen_token_holder if self.is_cramped_map else None
        
        for r, bid in enumerate(bots):
            bot = controller.get_bot_state(bid)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            is_holder = (bid == holder) if holder is not None else True
            
            for c, job in enumerate(jobs):
                # 1. Resolve Target
                target = job.target
                if not target:
                    if job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE]:
                        target = self.get_nearest(pos, self.shops)
                    elif job.job_type == JobType.CHOP: 
                        target = self.get_nearest(pos, self.counters) 
                    elif job.job_type == JobType.START_COOK: 
                        target = self.get_nearest(pos, self.cookers)
                    elif job.job_type == JobType.ADD_TO_PLATE: 
                        target = self.get_nearest(pos, self.counters)
                    elif job.job_type == JobType.PLACE_ON_COUNTER: 
                        if self.is_cramped_map and not is_holder:
                            target = self.handoff_spot # Outsider drops here
                        else:
                            # Prefer plating spot for plates, handoff for raw ingredients?
                            # Simplify: nearest empty
                            target = self._find_empty(controller, self.counters, pos)
                    elif job.job_type == JobType.TAKE_CLEAN_PLATE: 
                        target = self.get_nearest(pos, self.sink_tables)
                    elif job.job_type == JobType.WASH: 
                        target = self.get_nearest(pos, self.sinks)
                
                dist = self.get_dist(pos, target) if target else 0
                if dist >= 9999: dist = COST_IMPOSSIBLE

                # 2. CONTINUITY BONUS (The "State Fix" Logic)
                # If I am holding the item needed for this job, I MUST do it.
                state_bonus = 0
                jt = job.job_type
                
                if holding:
                    # Holding Food
                    if holding['type'] == 'Food':
                        if jt == JobType.CHOP and not holding.get('chopped'): state_bonus = -2000
                        if jt == JobType.START_COOK and (holding.get('chopped') or holding['food_name'] == 'EGG'): state_bonus = -2000
                        if jt == JobType.ADD_TO_PLATE: state_bonus = -2000
                        
                    # Holding Plate
                    if holding['type'] == 'Plate':
                        if jt == JobType.SUBMIT: state_bonus = -2000
                        if jt == JobType.PLACE_ON_COUNTER: state_bonus = -500 # Encourage dropping if not submitting
                        
                    # Holding Pan
                    if holding['type'] == 'Pan':
                        if jt == JobType.PLACE_ON_COUNTER: state_bonus = -2000

                # 3. PENALTIES
                # Logic Penalties (Physical impossibility)
                logic_penalty = 0
                if jt == JobType.CHOP and (not holding or holding['type'] == 'Plate'): logic_penalty = COST_HIGH
                if jt == JobType.SUBMIT and (not holding or holding['type'] != 'Plate'): logic_penalty = COST_HIGH
                if jt in [JobType.BUY_INGREDIENT, JobType.BUY_PAN] and holding: logic_penalty = COST_HIGH

                # Relay Penalties (Cramped Map Only)
                relay_penalty = 0
                if self.is_cramped_map:
                    target_in_kitchen = (target in self.kitchen_zone) if target else False
                    if not is_holder and target_in_kitchen:
                        relay_penalty = COST_IMPOSSIBLE
                    
                    # Force Outsider to prioritize Buying/Dropping
                    if not is_holder and jt == JobType.PLACE_ON_COUNTER:
                        relay_penalty = -500 # Bonus to drop off

                # Final Score
                score = dist - job.priority + state_bonus + logic_penalty + relay_penalty
                cost_matrix[r, c] = score

        if HAS_SCIPY:
            row, col = linear_sum_assignment(cost_matrix)
        else:
            col = np.argmin(cost_matrix, axis=1)
            row = range(len(bots))

        assignments = {bots[r]: jobs[c] for r, c in zip(row, col)}
        
        # Log Assignments
        if DEBUG and self.log_file:
            for b, j in assignments.items():
                st = controller.get_bot_state(b)
                self.log(f"Bot {b} @ ({st['x']},{st['y']}) -> {j}")
                
        return assignments

    # -------------------------------------------------------------------------
    # EXECUTION
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
                # Reservation Check could go here
                return path[0]
            if len(path) > 10: continue
            moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
            for dx, dy in moves:
                nx, ny = curr[0]+dx, curr[1]+dy
                if not (0<=nx<self.map.width and 0<=ny<self.map.height): continue
                if (nx, ny) not in self.walkable: continue
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))
        return (0,0)

    def execute(self, controller, bot_id, job):
        bot = controller.get_bot_state(bot_id)
        pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        if job.job_type in [JobType.IDLE, JobType.HOARD_PLATE]: return 

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
            # Fallback Resolve (should be handled in assign but just in case)
            if job.job_type in [JobType.BUY_INGREDIENT, JobType.BUY_PAN, JobType.BUY_PLATE]:
                target = self.get_nearest(pos, self.shops)
            # ... (Full resolve logic from previous versions if needed)
        
        if not target: return 
        
        if self.get_dist(pos, target) <= 1:
            self._perform_action(controller, bot_id, job, target, holding)
        else:
            self.move_bot(controller, bot_id, target)

    def move_bot(self, controller, bot_id, target):
        dx, dy = self.get_move(controller, bot_id, target, controller.get_turn())
        if dx == 0 and dy == 0: pass 
        else: controller.move(bot_id, dx, dy)

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
        self.turn = controller.get_turn()
        self.update_state(controller)
        self.reserved_nodes.clear()
        
        switch_info = controller.get_switch_info()
        my_switched = switch_info.get("my_team_switched", False)
        enemy_switched = switch_info.get("enemy_team_switched", False)
        
        if 250 <= self.turn < 350 and not my_switched:
            if enemy_switched: controller.switch_maps()

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
                    self.kitchen_token_holder = others[0]

        sorted_bots = sorted(assigns.keys(), key=lambda b: assigns[b].priority, reverse=True)
        
        start_ts = time.time()
        for bid in sorted_bots:
            if time.time() - start_ts > 0.45: break 
            self.execute(controller, bid, assigns[bid])