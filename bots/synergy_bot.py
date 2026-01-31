# synergy_bot.py
"""
A highly coordinated dual-bot system designed for maximum efficiency.

Architecture:
- Centralized Coordinator tracks all game state and assigns roles
- Bot 0: "CHEF" - Handles food preparation (chopping, cooking, plating)
- Bot 1: "RUNNER" - Handles logistics (buying, fetching, submitting)

Pipeline for NOODLES + MEAT order:
1. RUNNER: Buy Pan -> Place on Cooker
2. RUNNER: Buy Meat -> Place on Staging Counter
3. CHEF: Chop Meat on Counter
4. CHEF: Pickup Meat -> Place on Pan (starts cooking)
5. RUNNER: Buy Plate -> Place on Assembly Counter
6. RUNNER: Buy Noodles -> Add to Plate
7. CHEF: (Wait for cook) Take Meat from Pan -> Add to Plate
8. RUNNER: Pickup Plate -> Submit

Both bots work in parallel on non-blocking tasks.
"""

import random
from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict, Set, Any
from enum import Enum

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food, Item
except ImportError:
    pass

# --- Configuration ---
DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[SynergyBot] {msg}")

# --- Role Definitions ---
class Role(Enum):
    CHEF = "CHEF"
    RUNNER = "RUNNER"

# --- Task State Machine ---
class TaskState(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    INTERACTING = "INTERACTING"
    WAITING = "WAITING"
    DONE = "DONE"

class Task:
    """A single atomic task for a bot."""
    def __init__(self, name: str, target: Tuple[int, int], action: str, 
                 item=None, prereq=None, priority: int = 0):
        self.name = name
        self.target = target  # (x, y) to interact with
        self.action = action  # "BUY", "PLACE", "PICKUP", "CHOP", "ADD_TO_PLATE", "TAKE_FROM_PAN", "SUBMIT"
        self.item = item      # For BUY actions
        self.prereq = prereq  # Function that must return True before task can start
        self.priority = priority
        self.state = TaskState.IDLE
        self.assigned_to = None

    def __repr__(self):
        return f"Task({self.name}, {self.action}@{self.target}, {self.state.value})"

# --- Pathfinding ---
class Pathfinding:
    @staticmethod
    def dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def get_path(controller, start: Tuple[int, int], target: Tuple[int, int], 
                 stop_dist: int, avoid: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding with obstacle avoidance."""
        import heapq
        
        m = controller.get_map()
        w, h = m.width, m.height
        
        # Priority Queue: (f_score, g_score, current_node, path)
        start_h = abs(start[0] - target[0]) + abs(start[1] - target[1])
        queue = [(start_h, 0, start, [])]
        visited = {start: 0}
        
        while queue:
            f, g, curr, path = heapq.heappop(queue)
            
            dist = abs(curr[0] - target[0]) + abs(curr[1] - target[1])
            if dist <= stop_dist:
                return path
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < w and 0 <= ny < h:
                    is_walkable = m.is_tile_walkable(nx, ny) and (neighbor not in avoid)
                    
                    if is_walkable:
                        new_g = g + 1
                        if neighbor not in visited or new_g < visited[neighbor]:
                            visited[neighbor] = new_g
                            h_score = abs(nx - target[0]) + abs(ny - target[1])
                            heapq.heappush(queue, (new_g + h_score, new_g, neighbor, path + [(dx, dy)]))
                            
        return None

# --- Centralized Coordinator ---
class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map POIs
        self.counters = []
        self.cookers = []
        self.sinks = []
        self.sink_tables = []
        self.submit_locs = []
        self.shops = []
        self.trash_locs = []
        
        # Bot Roles
        self.roles: Dict[int, Role] = {}
        
        # Global State Tracking
        self.pan_placed = False
        self.pan_loc = None
        self.plate_placed = False
        self.plate_loc = None
        self.assembly_counter = None  # Where we build orders
        self.staging_counter = None   # Where we prep ingredients
        
        # Pipeline State
        self.pipeline_stage = 0
        """
        Pipeline Stages:
        0: Start - need to place pan
        1: Pan placed - need to buy meat
        2: Meat on staging counter - need to chop
        3: Meat chopped - need to place on pan
        4: Meat cooking - need plate + noodles
        5: Plate with noodles ready - waiting for meat
        6: Meat cooked - need to add to plate
        7: Plate complete - need to submit
        8: Submitted - reset
        """
        
        # Per-bot state
        self.bot_tasks: Dict[int, Optional[Task]] = {}
        self.bot_last_pos: Dict[int, Tuple[int, int]] = {}
        self.bot_stuck_count: Dict[int, int] = {}

    def _init_map_pois(self, controller):
        """Cache map points of interest."""
        m = controller.get_map()
        log(f"Map size: {m.width}x{m.height}")
        for x in range(m.width):
            for y in range(m.height):
                t = m.tiles[x][y].tile_name
                p = (x, y)
                if t == "COUNTER": 
                    self.counters.append(p)
                    log(f"Found COUNTER at {p}")
                elif t == "COOKER": 
                    self.cookers.append(p)
                    log(f"Found COOKER at {p}")
                elif t == "SINK": 
                    self.sinks.append(p)
                    log(f"Found SINK at {p}")
                elif t == "SINKTABLE": 
                    self.sink_tables.append(p)
                    log(f"Found SINKTABLE at {p}")
                elif t == "SUBMIT": 
                    self.submit_locs.append(p)
                    log(f"Found SUBMIT at {p}")
                elif t == "SHOP": 
                    self.shops.append(p)
                    log(f"Found SHOP at {p}")
                elif t == "TRASH": 
                    self.trash_locs.append(p)
                    log(f"Found TRASH at {p}")
        
        # Pre-assign stations
        if self.cookers:
            self.pan_loc = self.cookers[0]
        
        # Use different counters for staging and assembly if possible
        if len(self.counters) >= 2:
            self.staging_counter = self.counters[0]
            self.assembly_counter = self.counters[1]
        elif len(self.counters) == 1:
            # Only one counter - use it for both (will need to be careful)
            self.staging_counter = self.counters[0]
            self.assembly_counter = self.counters[0]
            log("WARNING: Only 1 counter available, using for both staging and assembly")
        else:
            log("ERROR: No counters found!")
        
        self.initialized = True
        log(f"Initialized. Cooker: {self.pan_loc}, Staging: {self.staging_counter}, Assembly: {self.assembly_counter}")

    def _init_roles(self, controller):
        """Assign roles to bots."""
        bots = controller.get_team_bot_ids()
        for i, bot_id in enumerate(bots):
            if i == 0:
                self.roles[bot_id] = Role.CHEF
            else:
                self.roles[bot_id] = Role.RUNNER
            self.bot_tasks[bot_id] = None
            self.bot_last_pos[bot_id] = (-1, -1)
            self.bot_stuck_count[bot_id] = 0
        log(f"Roles assigned: {self.roles}")

    def get_closest(self, pos, locs) -> Optional[Tuple[int, int]]:
        if not locs: return None
        return min(locs, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))

    def get_avoid_set(self, controller, exclude_bot_id: int) -> Set[Tuple[int, int]]:
        """Get positions of all other bots to avoid."""
        avoid = set()
        for bid in controller.get_team_bot_ids():
            if bid != exclude_bot_id:
                st = controller.get_bot_state(bid)
                avoid.add((st['x'], st['y']))
        return avoid

    def move_bot_towards(self, controller, bot_id: int, target: Tuple[int, int], stop_dist: int = 1) -> bool:
        """Move bot towards target. Returns True if already adjacent."""
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        
        dist = Pathfinding.dist((bx, by), target)
        if dist <= stop_dist:
            log(f"move_bot_towards: Bot {bot_id} at ({bx},{by}) is adjacent to {target}")
            return True  # Already there
        
        avoid = self.get_avoid_set(controller, bot_id)
        path = Pathfinding.get_path(controller, (bx, by), target, stop_dist, avoid)
        
        if path:
            dx, dy = path[0]
            controller.move(bot_id, dx, dy)
            log(f"move_bot_towards: Bot {bot_id} moving ({dx},{dy}) towards {target}")
            return False
        else:
            # No path found, try random wiggle
            log(f"move_bot_towards: Bot {bot_id} NO PATH from ({bx},{by}) to {target}, avoid={avoid}")
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    return False
            return False

    def check_cooker_state(self, controller, team) -> Dict[str, Any]:
        """Get the state of the cooker (pan, food, cook stage)."""
        if not self.pan_loc:
            return {"has_pan": False}
        
        tile = controller.get_tile(team, self.pan_loc[0], self.pan_loc[1])
        if not tile or not hasattr(tile, 'item') or not tile.item:
            return {"has_pan": False}
        
        if isinstance(tile.item, Pan):
            pan = tile.item
            if pan.food:
                return {
                    "has_pan": True,
                    "has_food": True,
                    "food_name": pan.food.food_name,
                    "cooked_stage": pan.food.cooked_stage,
                    "chopped": getattr(pan.food, 'chopped', False)
                }
            else:
                return {"has_pan": True, "has_food": False}
        return {"has_pan": False}

    def check_counter_state(self, controller, team, counter_loc) -> Dict[str, Any]:
        """Get the state of a counter."""
        if not counter_loc:
            return {"empty": True}
        
        tile = controller.get_tile(team, counter_loc[0], counter_loc[1])
        if not tile or not hasattr(tile, 'item') or not tile.item:
            return {"empty": True}
        
        item = tile.item
        if isinstance(item, Plate):
            foods = [f.food_name if isinstance(f, Food) else str(f) for f in item.food]
            return {"empty": False, "type": "Plate", "foods": foods, "dirty": item.dirty}
        elif isinstance(item, Food):
            return {
                "empty": False, 
                "type": "Food", 
                "food_name": item.food_name,
                "chopped": item.chopped,
                "cooked_stage": item.cooked_stage
            }
        elif isinstance(item, Pan):
            return {"empty": False, "type": "Pan"}
        return {"empty": False, "type": "Unknown"}

    def execute_chef(self, controller, bot_id: int, team):
        """Execute CHEF role logic."""
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        holding = state['holding']
        
        cooker_state = self.check_cooker_state(controller, team)
        staging_state = self.check_counter_state(controller, team, self.staging_counter)
        assembly_state = self.check_counter_state(controller, team, self.assembly_counter)
        
        log(f"CHEF@({bx},{by}) holding={holding}, cooker={cooker_state}, staging={staging_state}, assembly={assembly_state}")
        
        # --- PRIORITY 1: If holding complete plate, submit ---
        if holding and holding.get('type') == 'Plate':
            foods = holding.get('food', [])
            food_names = [f.get('food_name', '').upper() for f in foods]
            if 'NOODLES' in food_names and 'MEAT' in food_names:
                # Complete! Submit.
                if self.submit_locs:
                    if self.move_bot_towards(controller, bot_id, self.submit_locs[0]):
                        controller.submit(bot_id, self.submit_locs[0][0], self.submit_locs[0][1])
                        log(f"CHEF: Submitted order!")
                return
        
        # --- PRIORITY 2: Add cooked meat to plate ---
        if cooker_state.get("has_pan") and cooker_state.get("has_food"):
            if cooker_state.get("cooked_stage") == 1:  # Cooked!
                if holding:
                    if holding.get('type') == 'Food' and holding.get('food_name', '').upper() == 'MEAT':
                        # We have the meat, add to plate if plate exists
                        if assembly_state.get("type") == "Plate":
                            if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                                controller.add_food_to_plate(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                                log(f"CHEF: Added meat to plate")
                            return
                else:
                    # Take meat from pan
                    if self.move_bot_towards(controller, bot_id, self.pan_loc):
                        controller.take_from_pan(bot_id, self.pan_loc[0], self.pan_loc[1])
                        log(f"CHEF: Took meat from pan")
                    return
        
        # --- PRIORITY 3: Handle held meat ---
        if holding:
            if holding.get('type') == 'Food' and holding.get('food_name', '').upper() == 'MEAT':
                cooked = holding.get('cooked_stage', 0)
                chopped = holding.get('chopped', False)
                
                if cooked == 1:
                    # Cooked meat - add to plate
                    if assembly_state.get("type") == "Plate":
                        if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                            controller.add_food_to_plate(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                            log(f"CHEF: Added cooked meat to plate")
                        return
                    else:
                        # No plate yet, wait near assembly
                        self.move_bot_towards(controller, bot_id, self.assembly_counter, stop_dist=2)
                        log(f"CHEF: Waiting for plate with cooked meat")
                        return
                elif chopped and cooked == 0:
                    # Chopped but raw - put on pan to cook
                    if cooker_state.get("has_pan") and not cooker_state.get("has_food"):
                        if self.move_bot_towards(controller, bot_id, self.pan_loc):
                            controller.start_cook(bot_id, self.pan_loc[0], self.pan_loc[1])
                            log(f"CHEF: Started cooking chopped meat on pan")
                        return
        
        # --- PRIORITY 4: Chop meat on staging counter ---
        if staging_state.get("type") == "Food" and str(staging_state.get("food_name", "")).upper() == "MEAT":
            if not staging_state.get("chopped"):
                if self.move_bot_towards(controller, bot_id, self.staging_counter):
                    controller.chop(bot_id, self.staging_counter[0], self.staging_counter[1])
                    log(f"CHEF: Chopping meat")
                return
            else:
                # Chopped - pick it up
                if self.move_bot_towards(controller, bot_id, self.staging_counter):
                    controller.pickup(bot_id, self.staging_counter[0], self.staging_counter[1])
                    log(f"CHEF: Picked up chopped meat")
                return
        
        # --- PRIORITY 5: Pick up complete plate if ready ---
        if assembly_state.get("type") == "Plate":
            foods = [str(f).upper() for f in assembly_state.get("foods", [])]
            if "NOODLES" in foods and "MEAT" in foods:
                if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                    controller.pickup(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                    log(f"CHEF: Picked up complete plate")
                return
        
        # --- IDLE: Help with whatever is needed ---
        # If meat is cooking, go near cooker to be ready
        if cooker_state.get("has_food") and cooker_state.get("cooked_stage") == 0:
            # Cooking in progress, move near cooker
            self.move_bot_towards(controller, bot_id, self.pan_loc, stop_dist=2)
            log(f"CHEF: Waiting near cooker for meat to cook")
            return
        
        # Otherwise, wiggle randomly
        log(f"CHEF: Idle, wiggling")
        moved = False
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                moved = True
                break
        if not moved:
            log(f"CHEF: Cannot move at all!")

    def execute_runner(self, controller, bot_id: int, team):
        """Execute RUNNER role logic."""
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        holding = state['holding']
        money = controller.get_team_money()
        
        cooker_state = self.check_cooker_state(controller, team)
        staging_state = self.check_counter_state(controller, team, self.staging_counter)
        assembly_state = self.check_counter_state(controller, team, self.assembly_counter)
        
        shop = self.get_closest((bx, by), self.shops)
        
        log(f"RUNNER@({bx},{by}) holding={holding}, money={money}, staging={staging_state}")
        
        # --- PRIORITY 1: If holding something, place it appropriately ---
        if holding:
            itype = holding.get('type')
            
            if itype == 'Pan':
                # Place pan on cooker
                if self.move_bot_towards(controller, bot_id, self.pan_loc):
                    controller.place(bot_id, self.pan_loc[0], self.pan_loc[1])
                    log(f"RUNNER: Placed pan on cooker")
                return
            
            elif itype == 'Plate':
                # Place plate on assembly counter
                if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                    controller.place(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                    log(f"RUNNER: Placed plate on assembly counter")
                return
            
            elif itype == 'Food':
                fname = holding.get('food_name', '').upper()
                if fname == 'MEAT':
                    # Place on staging for chef to chop
                    if self.move_bot_towards(controller, bot_id, self.staging_counter):
                        controller.place(bot_id, self.staging_counter[0], self.staging_counter[1])
                        log(f"RUNNER: Placed meat on staging counter")
                    return
                elif fname == 'NOODLES':
                    # Add to plate
                    if assembly_state.get("type") == "Plate":
                        if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                            controller.add_food_to_plate(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                            log(f"RUNNER: Added noodles to plate")
                        return
                    else:
                        # No plate yet, wait
                        pass
            return
        
        # --- PRIORITY 2: Buy pan if needed ---
        if not cooker_state.get("has_pan"):
            if money >= ShopCosts.PAN.buy_cost:
                if self.move_bot_towards(controller, bot_id, shop):
                    controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    log(f"RUNNER: Bought pan")
                return
        
        # --- PRIORITY 3: Buy meat if staging is empty and pan has no food ---
        if staging_state.get("empty") and not cooker_state.get("has_food"):
            if money >= FoodType.MEAT.buy_cost:
                if self.move_bot_towards(controller, bot_id, shop):
                    controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1])
                    log(f"RUNNER: Bought meat")
                return
        
        # --- PRIORITY 4: Buy plate if assembly counter is empty ---
        if assembly_state.get("empty"):
            if money >= ShopCosts.PLATE.buy_cost:
                if self.move_bot_towards(controller, bot_id, shop):
                    controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
                    log(f"RUNNER: Bought plate")
                return
        
        # --- PRIORITY 5: Buy noodles if plate has no noodles ---
        if assembly_state.get("type") == "Plate":
            foods = [str(f).upper() for f in assembly_state.get("foods", [])]
            if "NOODLES" not in foods:
                if money >= FoodType.NOODLES.buy_cost:
                    if self.move_bot_towards(controller, bot_id, shop):
                        controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1])
                        log(f"RUNNER: Bought noodles")
                    return
        
        # --- PRIORITY 6: Pick up complete plate and submit ---
        if assembly_state.get("type") == "Plate":
            foods = [str(f).upper() for f in assembly_state.get("foods", [])]
            if "NOODLES" in foods and "MEAT" in foods:
                if self.move_bot_towards(controller, bot_id, self.assembly_counter):
                    controller.pickup(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                    log(f"RUNNER: Picked up complete plate")
                return
        
        # --- IDLE: Wiggle ---
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map_pois(controller)
            self._init_roles(controller)
        
        team = controller.get_team()
        bots = controller.get_team_bot_ids()
        
        # Check for stuck bots
        for bot_id in bots:
            st = controller.get_bot_state(bot_id)
            pos = (st['x'], st['y'])
            if pos == self.bot_last_pos.get(bot_id):
                self.bot_stuck_count[bot_id] = self.bot_stuck_count.get(bot_id, 0) + 1
            else:
                self.bot_stuck_count[bot_id] = 0
            self.bot_last_pos[bot_id] = pos
            
            if self.bot_stuck_count[bot_id] > 10:
                log(f"Bot {bot_id} stuck for {self.bot_stuck_count[bot_id]} turns")
        
        # Execute roles
        for bot_id in bots:
            role = self.roles.get(bot_id, Role.RUNNER)
            if role == Role.CHEF:
                self.execute_chef(controller, bot_id, team)
            else:
                self.execute_runner(controller, bot_id, team)
