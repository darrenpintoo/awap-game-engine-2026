
import random
from collections import deque, defaultdict
import heapq
from typing import Tuple, Optional, List, Dict, Set, Any, Union
import math

# Try imports
try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food, Item
except ImportError:
    pass

# --- Configuration ---
DEBUG_MODE = True
MAX_PLAN_DEPTH = 50

def log(msg):
    if DEBUG_MODE:
        print(f"[AdvBot] {msg}")

# --- Data Classes & Enums ---

class TaskType:
    FETCH_INGREDIENT = "FETCH_INGREDIENT" # [Buy/Pick] -> [Dest]
    PROCESS_FOOD = "PROCESS_FOOD"  # [Chop/Cook]
    PLATE_FOOD = "PLATE_FOOD"      # [Put on Plate]
    DELIVER_ORDER = "DELIVER_ORDER" # [Submit]
    WASH_DISHES = "WASH_DISHES"    # [Wash Sink]
    BUY_UTENSIL = "BUY_UTENSIL"    # [Buy Pan/Plate]

class Task:
    def __init__(self, task_type, priority, **kwargs):
        self.type = task_type
        self.priority = priority # Higher is better
        self.data = kwargs
        self.assigned_to = None # bot_id

    def __repr__(self):
        return f"Task({self.type}, p={self.priority}, {self.data})"

# --- Pathfinding ---
class Pathfinding:
    @staticmethod
    def get_path(controller: RobotController, start: Tuple[int, int], target: Tuple[int, int], 
                 stop_dist: int, avoid: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        m = controller.get_map()
        w, h = m.width, m.height
        
        # Priority Queue for A*: (f_score, g_score, current_node, path)
        # heuristic: Manhattan distance
        start_h = abs(start[0] - target[0]) + abs(start[1] - target[1])
        queue = [(start_h, 0, start, [])]
        visited = {start: 0} # node -> g_score
        
        while queue:
            f, g, curr, path = heapq.heappop(queue)
            
            # Check Goal
            dist = abs(curr[0] - target[0]) + abs(curr[1] - target[1])
            if dist <= stop_dist:
                # If stop_dist > 0, we need to ensure we are not ON the target if it is unwalkable? 
                # Actually, standard A* to an adjacent tile is better for interaction.
                # But this function supports "move to X" or "move next to X"
                return path
            
            cx, cy = curr
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < w and 0 <= ny < h:
                    # Walkability check
                    # We treat dynamic obstacles (avoid set) as walls
                    is_walkable = m.is_tile_walkable(nx, ny) and (neighbor not in avoid)
                    
                    if is_walkable:
                        new_g = g + 1
                        if neighbor not in visited or new_g < visited[neighbor]:
                            visited[neighbor] = new_g
                            h_score = abs(nx - target[0]) + abs(ny - target[1])
                            heapq.heappush(queue, (new_g + h_score, new_g, neighbor, path + [(dx, dy)]))
                            
        return None

    @staticmethod
    def dist(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

# --- Agent Wrapper ---
class Agent:
    def __init__(self, bot_id):
        self.id = bot_id
        self.current_task: Optional[Task] = None
        self.path = deque()
        self.state_description = "IDLE"
        self.last_pos = (-1, -1)
        self.stuck_count = 0

    def clear_task(self):
        self.current_task = None
        self.path.clear()
        self.state_description = "IDLE"

# --- Coordinator ---
class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.agents: Dict[int, Agent] = {}
        self.tasks: List[Task] = []
        self.initialized = False
        
        # Static Map POIs
        self.counters = []
        self.cookers = []
        self.sinks = []
        self.sink_tables = []
        self.submit_locs = []
        self.shops = []
        self.trash_locs = []
        self.floor_tiles = []
        
        # Game State Cache
        self.turn = 0
        self.money = 0
        self.orders = []
        self.reserved_tiles = {} # (x, y) -> bot_id (next turn)

    def _init_map_pois(self, controller: RobotController):
        m = controller.get_map()
        for x in range(m.width):
            for y in range(m.height):
                t = m.tiles[x][y].tile_name
                p = (x, y)
                if t == "COUNTER": self.counters.append(p)
                elif t == "COOKER": self.cookers.append(p)
                elif t == "SINK": self.sinks.append(p)
                elif t == "SINKTABLE": self.sink_tables.append(p)
                elif t == "SUBMIT": self.submit_locs.append(p)
                elif t == "SHOP": self.shops.append(p)
                elif t == "TRASH": self.trash_locs.append(p)
                elif t == "FLOOR": self.floor_tiles.append(p)
        self.initialized = True

    def get_world_state(self, controller):
        # Update internal view of the world
        self.turn = controller.get_turn()
        self.money = controller.get_team_money()
        self.orders = controller.get_orders()
        
        # Sync agents
        team_bots = controller.get_team_bot_ids()
        for bid in team_bots:
            if bid not in self.agents:
                self.agents[bid] = Agent(bid)
            
            # Stuck check
            bot_state = controller.get_bot_state(bid)
            curr = (bot_state['x'], bot_state['y'])
            if curr == self.agents[bid].last_pos:
                self.agents[bid].stuck_count += 1
            else:
                self.agents[bid].stuck_count = 0
                self.agents[bid].last_pos = curr

    # --- Strategy Helpers ---

    def get_closest(self, pos, locs) -> Optional[Tuple[int, int]]:
        if not locs: return None
        return min(locs, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))

    def get_path_to(self, controller, bot_id, target, stop_dist=0):
        agent = self.agents[bot_id]
        start = (agent.last_pos[0], agent.last_pos[1])
        
        # Avoid ALL teammates (live positions)
        avoid = set()
        others = controller.get_team_bot_ids()
        for oid in others:
            if oid == bot_id: continue
            ostate = controller.get_bot_state(oid)
            if ostate:
                avoid.add((ostate['x'], ostate['y']))
        
        path = Pathfinding.get_path(controller, start, target, stop_dist, avoid)
        return path

    def reserve_path(self, bot_id: int, path: List[Tuple[int, int]]):
        pass # Not needed with sequential logic + live checks

    # --- Logic ---

    def generate_tasks(self, controller):
        pass 

    def distribute_tasks_greedy(self, controller):
        pass

    def execute_agents(self, controller):
        my_bots = controller.get_team_bot_ids()
        team = controller.get_team()
        
        # 0. Global Switch Check
        if controller.can_switch_maps():
            if controller.get_turn() > 255: 
                controller.switch_maps()
                return 

        # --- 1. GLOBAL DEMAND CALCULATION ---
        # Calculate what is needed vs what exists to prevent duplicate buying
        active_orders = [o for o in self.orders if o['is_active']]
        active_orders.sort(key=lambda x: x['reward'], reverse=True)
        
        needed_counts = defaultdict(int)
        for o in active_orders:
            for ing in o['required']:
                 needed_counts[ing] += 1
        
        # Subtract items currently on map or held (Supply)
        # Scan Bots
        for bid in my_bots:
            st = controller.get_bot_state(bid)
            h = st['holding']
            if h:
                if h['type'] == 'Food':
                    needed_counts[h['food_name']] -= 1
                elif h['type'] == 'Plate':
                    for f in h.get('food', []):
                         needed_counts[f['food_name']] -= 1
                elif h['type'] == 'Pan':
                    if h.get('food'):
                        needed_counts[h['food']['food_name']] -= 1
        
        # Scan Counters/Cookers
        for cx, cy in self.counters + self.cookers:
            tile = controller.get_tile(team, cx, cy)
            if tile and hasattr(tile, 'item') and tile.item:
                it = tile.item
                if isinstance(it, Food):
                    needed_counts[it.food_name] -= 1
                elif isinstance(it, Pan) and it.food:
                    needed_counts[it.food.food_name] -= 1
                elif isinstance(it, Plate):
                    for f in it.food:
                         if isinstance(f, Food):
                             needed_counts[f.food_name] -= 1
        
        # Flatten needed list for assignment
        # e.g. ["Onion", "Meat"]
        pending_needs = []
        for name, count in needed_counts.items():
            if count > 0:
                pending_needs.extend([name] * count)

        # ----------------------------------------------------------------

        for bot_id in my_bots:
            agent = self.agents[bot_id]
            state = controller.get_bot_state(bot_id)
            bx, by = state['x'], state['y']
            agent.last_pos = (bx, by)
            did_act = False

            # --- 1. PRIORITY: WASH ---
            local_sink = None
            for sx, sy in self.sinks:
                tile = controller.get_tile(team, sx, sy)
                if tile and tile.num_dirty_plates > 0:
                    dist = abs(bx - sx) + abs(by - sy)
                    if dist <= 1:
                        local_sink = (sx, sy)
                        break
            if local_sink:
                controller.wash_sink(bot_id, local_sink[0], local_sink[1])
                did_act = True
                continue 

            # --- 2. HOLDING LOGIC (Unchanged - simplified for brevity of diff, but logic remains valid) ---
            holding = state['holding']
            if holding:
                itype = holding['type']
                # ... (Existing Holding Logic is fine, assumes we process what we hold)
                # Just ensuring we keep the code valid.
                # Paste the exact previous HOLDING logic here to ensure we don't delete it.
                
                # A. Holding Plate
                if itype == 'Plate':
                    foods = holding.get('food', [])
                    if holding.get('dirty'):
                         target = self.get_closest((bx, by), self.sinks)
                         if target:
                             if Pathfinding.dist((bx, by), target) <= 1:
                                 controller.put_dirty_plate_in_sink(bot_id, target[0], target[1])
                                 did_act = True
                             else:
                                 path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                 if path:
                                     dx, dy = path[0]
                                     controller.move(bot_id, dx, dy)
                    elif len(foods) > 0:
                        # Check for Order Completeness
                        current_ingredients = [f['food_name'] for f in foods]
                        best_match = None
                        missing_needed = []
                        
                        # Find the best order that this plate could satisfy
                        for o in self.orders:
                            if not o['is_active']: continue
                            req = o['required'] # List of strings
                            
                            # Check if current is a subset
                            # Use counters for multiset comparison
                            req_counts = defaultdict(int)
                            for r in req: req_counts[r] += 1
                            curr_counts = defaultdict(int)
                            for c in current_ingredients: curr_counts[c] += 1
                            
                            is_subset = True
                            for k, v in curr_counts.items():
                                if req_counts[k] < v:
                                    is_subset = False
                                    break
                            
                            if is_subset:
                                # Calculate missing
                                missing = []
                                for k, v in req_counts.items():
                                    rem = v - curr_counts[k]
                                    if rem > 0:
                                        missing.extend([k]*rem)
                                
                                # Prefer exact match (len(missing) == 0)
                                if len(missing) == 0:
                                    best_match = o
                                    missing_needed = []
                                    break # DONE
                                else:
                                    # Candidate for partial
                                    # Pick the one with fewest missing? Or highest reward?
                                    if best_match is None or len(missing) < len(missing_needed):
                                        best_match = o
                                        missing_needed = missing
                        
                        if best_match and len(missing_needed) == 0:
                            # COMPLETE ORDER -> SUBMIT
                            target = self.get_closest((bx, by), self.submit_locs)
                            if target:
                                 dist = Pathfinding.dist((bx, by), target)
                                 if dist <= 1:
                                     if controller.submit(bot_id, target[0], target[1]):
                                         did_act = True
                                         agent.state_description = "Submitted"
                                     else:
                                         # Weird fail? Trash.
                                         trash = self.get_closest((bx, by), self.trash_locs)
                                         if trash and Pathfinding.dist((bx, by), trash) <= 1:
                                             controller.trash(bot_id, trash[0], trash[1])
                                 else:
                                     path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                     if path:
                                         dx, dy = path[0]
                                         controller.move(bot_id, dx, dy)
                        elif best_match:
                            # PARTIAL ORDER -> FIND MISSING
                            # Look for missing items on counters/cookers
                            found_ing_loc = None
                            for ing_name in missing_needed:
                                # Check Cookers (Pans with Cooked Food)
                                for cx, cy in self.cookers:
                                    tile = controller.get_tile(team, cx, cy)
                                    if tile and isinstance(tile.item, Pan) and tile.item.food:
                                        # Check if cooked?
                                        # Assume valid checking logic
                                        f = tile.item.food
                                        if f.food_name == ing_name and f.cooked_stage == 1:
                                            found_ing_loc = (cx, cy)
                                            break
                                if found_ing_loc: break
                                
                                # Check Counters (Chopped/Cooked Food)
                                for cx, cy in self.counters:
                                    tile = controller.get_tile(team, cx, cy)
                                    if tile and isinstance(tile.item, Food):
                                        f = tile.item
                                        # Check ready conditions
                                        is_ready = True
                                        if f.food_name in ["Meat", "Egg"] and f.cooked_stage != 1: is_ready = False
                                        if f.food_name in ["Meat", "Onion"] and not f.chopped: is_ready = False
                                        
                                        if f.food_name == ing_name and is_ready:
                                            found_ing_loc = (cx, cy)
                                            break
                                if found_ing_loc: break
                            
                            if found_ing_loc:
                                # Go pick it up (Add to plate)
                                target = found_ing_loc
                                if Pathfinding.dist((bx, by), target) <= 1:
                                    controller.add_food_to_plate(bot_id, target[0], target[1])
                                    did_act = True
                                else:
                                    path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                    if path:
                                        dx, dy = path[0]
                                        controller.move(bot_id, dx, dy) 
                            else:
                                # MISSING ITEM NOT FOUND.
                                # Place Plate on Counter so we can work on the missing item
                                best_counter = None
                                min_dist = 999
                                for cx, cy in self.counters:
                                    tile = controller.get_tile(team, cx, cy)
                                    if tile and tile.item is None:
                                        d = abs(bx-cx) + abs(by-cy)
                                        if d < min_dist:
                                            min_dist = d
                                            best_counter = (cx, cy)
                                
                                if best_counter:
                                    if min_dist <= 1:
                                        controller.place(bot_id, best_counter[0], best_counter[1])
                                        did_act = True
                                    else:
                                        path = self.get_path_to(controller, bot_id, best_counter, stop_dist=1)
                                        if path:
                                            dx, dy = path[0]
                                            controller.move(bot_id, dx, dy)
                        else:
                             # No matching order? Trash it?
                             # Or put on counter?
                             # Trash to free up plate.
                             trash = self.get_closest((bx, by), self.trash_locs)
                             if trash:
                                 if Pathfinding.dist((bx, by), trash) <= 1:
                                     controller.trash(bot_id, trash[0], trash[1])
                                 else:
                                     path = self.get_path_to(controller, bot_id, trash, stop_dist=1)
                                     if path:
                                         dx, dy = path[0]
                                         controller.move(bot_id, dx, dy)
                    else:
                        processed_food_loc = None
                        for cx, cy in self.cookers:
                            tile = controller.get_tile(team, cx, cy)
                            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                                pan = tile.item
                                if pan.food and pan.food.cooked_stage == 1:
                                    processed_food_loc = (cx, cy)
                                    break
                        if processed_food_loc:
                            target = processed_food_loc
                            if Pathfinding.dist((bx, by), target) <= 1:
                                controller.add_food_to_plate(bot_id, target[0], target[1])
                                did_act = True
                            else:
                                path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                if path:
                                     dx, dy = path[0]
                                     controller.move(bot_id, dx, dy)
                        else:
                             best_counter = None
                             min_dist = 999
                             for cx, cy in self.counters:
                                 tile = controller.get_tile(team, cx, cy)
                                 if tile and tile.item is None:
                                     d = abs(bx-cx) + abs(by-cy)
                                     if d < min_dist:
                                         min_dist = d
                                         best_counter = (cx, cy)
                             if best_counter:
                                 if min_dist <= 1:
                                     controller.place(bot_id, best_counter[0], best_counter[1])
                                     did_act = True
                                 else:
                                     path = self.get_path_to(controller, bot_id, best_counter, stop_dist=1)
                                     if path:
                                         dx, dy = path[0]
                                         controller.move(bot_id, dx, dy)
                # B. Holding Pan
                elif itype == 'Pan':
                    target = None
                    for cx, cy in self.cookers:
                        tile = controller.get_tile(team, cx, cy)
                        if tile and getattr(tile, 'item', None) is None:
                            target = (cx, cy)
                            break
                    if target:
                        if Pathfinding.dist((bx, by), target) <= 1:
                            controller.place(bot_id, target[0], target[1])
                            did_act = True
                        else:
                            path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                            if path:
                                dx, dy = path[0]
                                controller.move(bot_id, dx, dy)
                    else:
                        target = None
                        min_dist = 999
                        for cx, cy in self.counters:
                             tile = controller.get_tile(team, cx, cy)
                             if tile and tile.item is None:
                                 d = abs(bx-cx) + abs(by-cy)
                                 if d < min_dist:
                                     min_dist = d
                                     target = (cx, cy)
                        if target and Pathfinding.dist((bx, by), target) <= 1:
                            controller.place(bot_id, target[0], target[1])
                # C. Holding Food
                elif itype == 'Food':
                    fname = holding['food_name']
                    chopped = holding.get('chopped', False)
                    cooked = holding.get('cooked_stage', 0)
                    needs_chop = (not chopped) and (fname in ["Onion", "Meat"])
                    needs_cook = (cooked == 0) and (fname in ["Meat", "Egg"])
                    if needs_chop:
                        target = None
                        min_dist = 999
                        for cx, cy in self.counters:
                             tile = controller.get_tile(team, cx, cy)
                             if tile and tile.item is None:
                                 d = abs(bx-cx) + abs(by-cy)
                                 if d < min_dist:
                                     min_dist = d
                                     target = (cx, cy)
                        if target:
                            if Pathfinding.dist((bx, by), target) <= 1:
                                if controller.place(bot_id, target[0], target[1]):
                                    controller.chop(bot_id, target[0], target[1])
                                    did_act = True
                            else:
                                path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                if path:
                                    dx, dy = path[0]
                                    controller.move(bot_id, dx, dy)
                    elif needs_cook:
                        target = None
                        for cx, cy in self.cookers:
                            tile = controller.get_tile(team, cx, cy)
                            if tile and isinstance(tile.item, Pan) and tile.item.food is None:
                                target = (cx, cy)
                                break
                        if target:
                            if Pathfinding.dist((bx, by), target) <= 1:
                                if controller.place(bot_id, target[0], target[1]):
                                    did_act = True
                            else:
                                path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                if path:
                                    dx, dy = path[0]
                                    controller.move(bot_id, dx, dy)
                        else:
                            target = None
                            min_dist = 999
                            for cx, cy in self.counters:
                                 tile = controller.get_tile(team, cx, cy)
                                 if tile and tile.item is None:
                                     d = abs(bx-cx) + abs(by-cy)
                                     if d < min_dist:
                                         min_dist = d
                                         target = (cx, cy)
                            if target and Pathfinding.dist((bx, by), target) <= 1:
                                controller.place(bot_id, target[0], target[1])
                    else:
                        target = None
                        for cx, cy in self.counters:
                            tile = controller.get_tile(team, cx, cy)
                            if tile and isinstance(tile.item, Plate) and not tile.item.dirty:
                                target = (cx, cy)
                                break
                        if target:
                            if Pathfinding.dist((bx, by), target) <= 1:
                                controller.add_food_to_plate(bot_id, target[0], target[1])
                                did_act = True
                            else:
                                path = self.get_path_to(controller, bot_id, target, stop_dist=1)
                                if path:
                                    dx, dy = path[0]
                                    controller.move(bot_id, dx, dy)
                        else:
                            target = None
                            min_dist = 999
                            for cx, cy in self.counters:
                                 tile = controller.get_tile(team, cx, cy)
                                 if tile and tile.item is None:
                                     d = abs(bx-cx) + abs(by-cy)
                                     if d < min_dist:
                                         min_dist = d
                                         target = (cx, cy)
                            if target and Pathfinding.dist((bx, by), target) <= 1:
                                controller.place(bot_id, target[0], target[1])

            # --- 3. EMPTY HANDED ---
            else:
                # A. WASH (Remote check)
                wash_target = None
                for sx, sy in self.sinks:
                    tile = controller.get_tile(team, sx, sy)
                    if tile and tile.num_dirty_plates > 0:
                        wash_target = (sx, sy)
                        break
                if wash_target:
                    dist = Pathfinding.dist((bx, by), wash_target)
                    if dist <= 1:
                        controller.wash_sink(bot_id, wash_target[0], wash_target[1])
                        did_act = True
                        continue 
                    elif dist < 5: 
                        path = self.get_path_to(controller, bot_id, wash_target, stop_dist=1)
                        if path:
                            dx, dy = path[0]
                            controller.move(bot_id, dx, dy)
                            continue

                # B. PICKUP EXISTING PLATES (Crucial for Assembly)
                pickup_target = None
                # Prioritize plates with food (Work in Progress)
                for cx, cy in self.counters:
                    tile = controller.get_tile(team, cx, cy)
                    if tile and isinstance(tile.item, Plate):
                        if tile.item.food: # Has food
                            pickup_target = (cx, cy)
                            break
                
                # If no food plates, pickup empty clean plates if we might need them?
                # Actually, blindly picking up empty plates is fine, we will fill them.
                if not pickup_target:
                    for cx, cy in self.counters:
                         tile = controller.get_tile(team, cx, cy)
                         if tile and isinstance(tile.item, Plate) and not tile.item.dirty:
                             pickup_target = (cx, cy)
                             break
                
                if pickup_target:
                    if Pathfinding.dist((bx, by), pickup_target) <= 1:
                        controller.pickup(bot_id, pickup_target[0], pickup_target[1])
                        did_act = True
                    else:
                        path = self.get_path_to(controller, bot_id, pickup_target, stop_dist=1)
                        if path:
                            dx, dy = path[0]
                            controller.move(bot_id, dx, dy)
                            did_act = True # Prevent random move
                    # If we decide to pickup, we skip buying (did_act check below covers this implicitly if we acted, but if moving?)
                    # If we just moved, we should NOT fall through to Buy.
                    # Because Buy logic might send us elsewhere.
                    # So checks order.
                    # Add explicit continue or did_act check wrapping.
                
                if did_act or pickup_target: 
                    # If we acted or are moving to pickup, stop decision
                    pass
                else:
                    # C. ADVANCE INGREDIENTS (Maintenance)
                    # Check for food on counters that needs processing (Chopping or Moving to Cooker)
                    maint_target = None
                    action_type = None # "chop" or "pickup"
                    
                    for cx, cy in self.counters:
                        tile = controller.get_tile(team, cx, cy)
                        if tile and isinstance(tile.item, Food):
                            f = tile.item
                            # 1. Needs Chop?
                            if (not f.chopped) and f.food_name in ["Onion", "Meat"]:
                                maint_target = (cx, cy)
                                action_type = "chop"
                                break
                            # 2. Needs Cook? (And is chopped if needed)
                            # Meat needs chop first. Egg doesn't.
                            # If Meat is chopped (or Egg), and raw -> Pickup to move to cooker
                            needs_cook = False
                            if f.food_name == "Meat" and f.chopped and f.cooked_stage == 0: needs_cook = True
                            if f.food_name == "Egg" and f.cooked_stage == 0: needs_cook = True
                            
                            if needs_cook:
                                maint_target = (cx, cy)
                                action_type = "pickup"
                                break
                    
                    if maint_target:
                        dist = Pathfinding.dist((bx, by), maint_target)
                        if dist <= 1:
                            if action_type == "chop":
                                controller.chop(bot_id, maint_target[0], maint_target[1])
                                did_act = True
                            elif action_type == "pickup":
                                controller.pickup(bot_id, maint_target[0], maint_target[1])
                                did_act = True
                        else:
                            path = self.get_path_to(controller, bot_id, maint_target, stop_dist=1)
                            if path:
                                dx, dy = path[0]
                                controller.move(bot_id, dx, dy)
                                did_act = True # Prevent random move
                    
                    # D. BUYING (Only if maintenance didn't act/move)
                    if not did_act: 
                         # Note: maintenance might have failed to find target, so check needed_item
                         needed_item = None
                         if pending_needs:
                             needed_item = pending_needs.pop(0) # Pick first needed item and CLAIM it
                         
                         if needed_item:
                             buy_target = None
                             buy_enum = None
                             name_map = {
                                 "Onion": FoodType.ONIONS,
                                 "ONIONS": FoodType.ONIONS,
                                 "Meat": FoodType.MEAT,
                                 "MEAT": FoodType.MEAT,
                                 "Egg": FoodType.EGG,
                                 "EGG": FoodType.EGG,
                                 "Noodles": FoodType.NOODLES,
                                 "NOODLES": FoodType.NOODLES,
                                 "Sauce": FoodType.SAUCE,
                                 "SAUCE": FoodType.SAUCE
                             }
                             if needed_item in name_map:
                                 buy_enum = name_map[needed_item]
                             
                             if buy_enum:
                                  shop = self.get_closest((bx, by), self.shops)
                                  if shop:
                                      if Pathfinding.dist((bx, by), shop) <= 1:
                                          # Buy
                                          controller.buy(bot_id, buy_enum, shop[0], shop[1])
                                          did_act = True
                                      else:
                                          path = self.get_path_to(controller, bot_id, shop, stop_dist=1)
                                          if path:
                                              dx, dy = path[0]
                                              controller.move(bot_id, dx, dy)
                                              did_act = True # Prevent random move
                
            # Random Move if nothing happened
            if not did_act:
                 # Wiggle
                 valid = []
                 for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                     if controller.can_move(bot_id, dx, dy):
                         valid.append((dx, dy))
                 if valid:
                     dx, dy = random.choice(valid)
                     controller.move(bot_id, dx, dy)

    def play_turn(self, controller: RobotController):
        if not self.initialized: self._init_map_pois(controller)
        self.get_world_state(controller)
        
        # Reset reservations for this turn
        self.reserved_tiles = {}
        
        # Strategy here
        self.generate_tasks(controller)
        self.distribute_tasks_greedy(controller)
        self.execute_agents(controller)
