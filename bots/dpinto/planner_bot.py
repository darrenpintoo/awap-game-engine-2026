
"""
PlannerBot - GOAP Architecture (Goal-Oriented Action Planning)
Uses a dependency graph to resolve complex orders dynamically.
"""

import math
import heapq
import random
from collections import deque, defaultdict
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Set, Any

# Ensure we can import from src
try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food, Item
except ImportError:
    # Graceful fallback for IDE/Linter
    pass

DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[PlannerBot] {msg}")

# --- 1. Recipe Graph Definition ---

class ItemState(Enum):
    RAW = auto()
    CHOPPED = auto()
    COOKED = auto()
    PLATED = auto()

class Ingredient:
    def __init__(self, name: str, state: ItemState):
        self.name = name  # e.g., "MEAT", "ONIONS"
        self.state = state

    def __repr__(self):
        return f"{self.name}({self.state.name})"

    def __eq__(self, other):
        if not isinstance(other, Ingredient):
            return False
        return self.name == other.name and self.state == other.state

    def __hash__(self):
        return hash((self.name, self.state))

class RecipeGraph:
    """
    Defines how to get from State A to State B.
    """
    @staticmethod
    def get_prerequisites(target: Ingredient) -> List[Tuple[str, Optional[Ingredient]]]:
        """
        Returns (Action, RequiredIngredient) to achieve target.
        """
        # --- MEAT ---
        if target.name == "MEAT":
            if target.state == ItemState.COOKED:
                return [("COOK", Ingredient("MEAT", ItemState.CHOPPED))]
            elif target.state == ItemState.CHOPPED:
                return [("CHOP", Ingredient("MEAT", ItemState.RAW))]
            elif target.state == ItemState.RAW:
                return [("BUY", None)] # Leaf node

        # --- ONIONS ---
        elif target.name == "ONIONS":
            if target.state == ItemState.COOKED:
                # Onions cook directly from chopped? Wait, let's check game rules.
                # Assuming standard: Chop -> Cook
                return [("COOK", Ingredient("ONIONS", ItemState.CHOPPED))] 
            elif target.state == ItemState.CHOPPED:
                return [("CHOP", Ingredient("ONIONS", ItemState.RAW))]
            elif target.state == ItemState.RAW:
                return [("BUY", None)]

        # --- EGG ---
        elif target.name == "EGG":
            # Eggs usually don't need chopping?
            if target.state == ItemState.COOKED:
                return [("COOK", Ingredient("EGG", ItemState.RAW))]
            elif target.state == ItemState.RAW:
                return [("BUY", None)]

        # --- NOODLES ---
        elif target.name == "NOODLES":
            # Noodles just need to be bought
            if target.state == ItemState.RAW:
                return [("BUY", None)]
        
        # --- SAUCE ---
        elif target.name == "SAUCE":
            if target.state == ItemState.RAW:
                return [("BUY", None)]

        return []

# --- 2. Action Planning System ---

class TaskType(Enum):
    BUY = auto()
    CHOP = auto()
    COOK = auto()       # Put on pan
    FETCH_COOKED = auto() # Take from pan
    PLATE = auto()      # Put ingredient on plate
    DELIVER = auto()    # Submit plate
    
    # Pre-req tasks
    DISCARD = auto()
    MOVE_ITEM = auto()  # Move item X to loc Y
    PICKUP = auto()

class Task:
    def __init__(self, ttype: TaskType, target_loc: Tuple[int, int], 
                 item: Any = None, priority: float = 0.0):
        self.type = ttype
        self.target_loc = target_loc # Where to perform action
        self.item = item            # What item is involved (Ingredient or str)
        self.priority = priority    # Higher is better
        self.assigned_to = None     # bot_id
        self.dependencies = []      # List[Task] - must be done first

    def __repr__(self):
        return f"Task({self.type.name} @ {self.target_loc}, item={self.item})"

# --- 3. Pathfinding with Reservations ---

class ReservationTable:
    def __init__(self):
        # Maps (x, y, time) -> bot_id
        self.reserved: Dict[Tuple[int, int, int], int] = {}
        self.max_time = 50 # Increased from 20 to allow finding paths around dynamic obstacles

    def reserve(self, path: List[Tuple[int, int]], start_time: int, bot_id: int):
        for i, (x, y) in enumerate(path):
            t = start_time + i
            self.reserved[(x, y, t)] = bot_id
            
    def is_blocked(self, x, y, t, my_id):
        # Basic reservation check
        r = self.reserved.get((x, y, t))
        if r is not None and r != my_id:
            return True
        return False

    def clear(self):
        self.reserved.clear()

class AStar:
    @staticmethod
    def get_path(controller, start: Tuple[int, int], target: Tuple[int, int], 
                 start_time: int, reservation: ReservationTable, my_id: int, 
                 stop_dist: int = 1) -> Optional[List[Tuple[int, int]]]:
        
        m = controller.get_map(controller.get_team())
        w, h = m.width, m.height
        
        # (f_score, g_score, x, y, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start[0], start[1], []))
        
        # Visited must include time because (x,y) might be blocked at t=5 but free at t=6
        visited_nodes = {} # (x, y, t) -> min_g

        while open_set:
            f, g, cx, cy, path = heapq.heappop(open_set)
            
            curr_time = start_time + g
            
            # Check target reached (Chebyshev distance)
            dist = max(abs(cx - target[0]), abs(cy - target[1]))
            if dist <= stop_dist:
                return path

            if g > 50: # Increased depth limit
                continue

            # Generate moves (wait included) - Full 8-direction Chebyshev movement
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
            
            for dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                nt = curr_time + 1
                
                if 0 <= nx < w and 0 <= ny < h:
                    # Check static map
                    if not m.is_tile_walkable(nx, ny):
                        continue
                        
                    # Check dynamic reservation
                    if reservation.is_blocked(nx, ny, nt, my_id):
                        continue
                    
                    # Vertex conflict/Swap check (if someone was at nx,ny at t and goes to cx,cy at t+1)
                    # Simplified: if we want to enter nx,ny at t+1, check if anyone reserved it. Done above.
                    # But if we swap? A at (0,0) -> (0,1). B at (0,1) -> (0,0).
                    # A reserves (0,1)@t+1. B reserves (0,0)@t+1.
                    # Valid reservations. But collision happens mid-edge.
                    # Ignore for now as game engine prevents it usually (sequential moves?).
                    # Actually game engine processes concurrent. 
                    # But let's assume reservation table is enough.

                    new_g = g + 1
                    state = (nx, ny, nt)
                    
                    if state not in visited_nodes or new_g < visited_nodes[state]:
                        visited_nodes[state] = new_g
                        h_score = max(abs(nx - target[0]), abs(ny - target[1]))
                        # Prioritize NON-WAIT moves in h_score? No, standard A* is fine.
                        heapq.heappush(open_set, (new_g + h_score, new_g, nx, ny, path + [(dx, dy)]))
                        
        return None

# --- 4. High-Level Planning ---

class Goal:
    def __init__(self, order_id: int, requirements: List[Ingredient], reward: int, time_left: int):
        self.order_id = order_id
        self.requirements = requirements # List[Ingredient]
        self.reward = reward
        self.time_left = time_left
        self.score = 0.0
        
    def __repr__(self):
        return f"Goal(id={self.order_id}, score={self.score:.1f})"

class OrderManager:
    def __init__(self):
        self.active_goals: List[Goal] = []
        
    def update_goals(self, controller, current_turn):
        self.active_goals = []
        orders = controller.get_orders(controller.get_team())
        
        for order in orders:
            # Parse requirements
            reqs = []
            for item_name in order['required']:
                # Determine needed state
                name = item_name
                state = ItemState.PLATED
                # Note: "PLATED" is the end state. The sub-requirements are what we track.
                # Actually, the order just wants "MEAT" in the plate.
                # So the requirement is Ingredient("MEAT", PLATED)
                reqs.append(Ingredient(name, ItemState.PLATED))
                
            rem_time = order['expires_turn'] - current_turn
            if rem_time < 20: continue # Skip if expiring soon
            
            goal = Goal(order['order_id'], reqs, order['reward'], rem_time)
            
            # Simple heuristic: Reward / Items needed
            goal.score = goal.reward / (len(reqs) * 50) 
            self.active_goals.append(goal)
            
        self.active_goals.sort(key=lambda x: x.score, reverse=True)

class TaskAllocator:
    def __init__(self, bot_player):
        self.bp = bot_player
        
    def get_available_items(self, controller):
        """
        Returns dict: Ingredient(Name, State) -> List[(x,y)]
        """
        items = defaultdict(list)
        team = controller.get_team()
        
        # Scan counters
        for cx, cy in self.bp.counters:
            tile = controller.get_tile(team, cx, cy)
            if tile and tile.item:
                if isinstance(tile.item, Food):
                    state = ItemState.RAW
                    if tile.item.chopped: state = ItemState.CHOPPED
                    # Note: Cooked items on counter are just regular food or burnt?
                    if tile.item.cooked_stage == 1: state = ItemState.COOKED
                    
                    ing = Ingredient(tile.item.food_name, state)
                    items[str(ing)].append((cx, cy))
                    
        # Scan cookers
        for cx, cy in self.bp.cookers:
            tile = controller.get_tile(team, cx, cy)
            if isinstance(tile, TileType) or not tile: continue 
            # Note: tile is Cooker object
            pan = tile.item # might be Pan object
            if isinstance(pan, Pan) and pan.food:
                f = pan.food
                if f.cooked_stage == 1:
                    # Ready to pick up
                    ing = Ingredient(f.food_name, ItemState.COOKED)
                    items[str(ing)].append((cx, cy))
                else:
                    # Still cooking (stage 0) - mark as COOKING so we don't re-buy
                    # Use a special "COOKING" key
                    items[f"{f.food_name}(COOKING)"].append((cx, cy))
                    
        return items

    def _validate_tasks(self, controller, bots, world_items):
        for bot_id in bots:
            task = self.bp.bot_assignments.get(bot_id)
            if not task: continue
            
            valid = True
            tx, ty = task.target_loc
            tile = controller.get_tile(controller.get_team(), tx, ty)
            
            # bot state
            st = controller.get_bot_state(bot_id)
            held = st.get('holding')

            if task.type == TaskType.PICKUP:
                if not tile or not tile.item: valid = False
            elif task.type == TaskType.CHOP:
                # Valid if tile has item OR bot is holding the item to be chopped
                if (not tile or not tile.item) and not held: valid = False
            elif task.type == TaskType.COOK:
                # Valid if tile has pan OR bot is holding food to cook
                if not tile or not isinstance(tile.item, Pan): valid = False
                elif not tile.item.food and not held: valid = False
            elif task.type == TaskType.PLATE:
                if not tile or not isinstance(tile.item, Plate): valid = False
            elif task.type == TaskType.FETCH_COOKED:
                if not tile or not isinstance(tile.item, Pan) or not tile.item.food: valid = False
            elif task.type == TaskType.DELIVER:
                if not held or held.get('type') != 'Plate': valid = False
            
            if not valid:
                log(f"Bot {bot_id} CLEARING INVALID task {task.type}")
                self.bp.bot_assignments[bot_id] = None

    def find_free_cooker(self, controller):
        for cx, cy in self.bp.cookers:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if isinstance(tile, TileType) or not tile: continue
            if not tile.item: # Empty cooker
                return (cx, cy)
            elif isinstance(tile.item, Pan) and not tile.item.food: # Empty pan
                return (cx, cy)
        return None

    def assign_tasks(self, controller: RobotController, bots: List[int]):
        if not self.bp.order_manager.active_goals: return
        top_goal = self.bp.order_manager.active_goals[0]

        # Get world state
        world_items = self.get_available_items(controller)
        available_bots = [bid for bid in bots if not self.bp.bot_assignments.get(bid)]
        
        # Helper to find specific item in world
        def find_item_loc(ing, bot_pos):
            locs = world_items.get(str(ing), [])
            if not locs: return None
            return min(locs, key=lambda p: max(abs(p[0]-bot_pos[0]), abs(p[1]-bot_pos[1])))

        # Helper to check if bot holding something relevant
        def check_holding(bot_id):
            st = controller.get_bot_state(bot_id)
            if not st['holding']: return None
            h = st['holding']
            if h['type'] == 'Food':
                state = ItemState.RAW
                if h.get('chopped'): state = ItemState.CHOPPED
                if h.get('cooked_stage') == 1: state = ItemState.COOKED
                return Ingredient(h['food_name'], state)
            elif h['type'] == 'Pan':
                return "PAN"
            elif h['type'] == 'Plate':
                return "PLATE"
            return None

        # Analyze assembly plate
        assembly_loc = self.bp.counters[0] if self.bp.counters else None
        if not assembly_loc: return
        
        self._validate_tasks(controller, bots, world_items)

        # 1. Plate Analysis
        all_plates = []
        tile = controller.get_tile(controller.get_team(), assembly_loc[0], assembly_loc[1])
        if tile and tile.item and isinstance(tile.item, Plate):
            all_plates.append(([f.food_name for f in tile.item.food], assembly_loc, None))
            
        for bid in bots:
            st = controller.get_bot_state(bid)
            holding = st.get('holding')
            if holding and holding.get('type') == 'Plate':
                contents = [f['food_name'] for f in holding.get('food', [])]
                all_plates.append((contents, (st['x'], st['y']), bid))
        
        best_plate = None
        if all_plates:
            best_plate = max(all_plates, key=lambda p: (len([f for f in p[0] if any(r.name == f for r in top_goal.requirements)]), -len(p[0])))
        
        plate_contents = best_plate[0] if best_plate else []
        has_plate = best_plate is not None
        best_plate_held_by = best_plate[2] if best_plate else None
        missing_reqs = [r for r in top_goal.requirements if r.name not in plate_contents]

        # 2. Preparation Check (Advanced)
        # We need this to decide when to buy the plate
        all_choppable_ready = True
        for req in top_goal.requirements:
            if req.name in plate_contents: continue
            if req.name in ["MEAT", "ONIONS"]:
                found = False
                # Is it chopped or cooked or cooking?
                for k in [f"{req.name}(COOKED)", f"{req.name}(COOKING)", f"{req.name}(CHOPPED)"]:
                    if k in world_items: found = True; break
                if not found:
                    # Check if someone holds it chopped or better
                    for bid in bots:
                        h = check_holding(bid)
                        if isinstance(h, Ingredient) and h.name == req.name and h.state != ItemState.RAW:
                            found = True; break
                if not found: all_choppable_ready = False; break

        # 3. Blocker Handling
        occupied_by_food = False
        if tile and tile.item and isinstance(tile.item, Food): occupied_by_food = True

        # 4. Delivery
        if has_plate and not missing_reqs:
            assigned_deliver = any(t and t.type == TaskType.DELIVER for t in self.bp.bot_assignments.values())
            if not assigned_deliver:
                if best_plate_held_by is not None:
                    target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                    self.bp.bot_assignments[best_plate_held_by] = Task(TaskType.DELIVER, target, item="PLATE")
                elif available_bots:
                    bid = available_bots.pop(0)
                    self.bp.bot_assignments[bid] = Task(TaskType.PICKUP, assembly_loc)

        # 5. Greedy Assign
        for bot_id in available_bots[:]:
            held = check_holding(bot_id)
            if held:
                # If holding plate, deliver or place on counter
                if str(held) == "PLATE":
                    log(f"  Bot {bot_id} has PLATE")
                    if not missing_reqs and best_plate_held_by == bot_id:
                        target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                        self.bp.bot_assignments[bot_id] = Task(TaskType.DELIVER, target, item="PLATE")
                    else:
                        # Proactive Plating: If holding plate, pick up ready ingredients from world
                        plate_task = None
                        st = controller.get_bot_state(bot_id)
                        for req in missing_reqs:
                            for state in ["COOKED", "RAW"]:
                                if state == "RAW" and req.name not in ["NOODLES", "SAUCE"]: continue
                                loc = find_item_loc(f"{req.name}({state})", (st['x'], st['y']))
                                if loc:
                                    plate_task = Task(TaskType.PLATE, loc, item=req)
                                    break
                            if plate_task: break
                        
                        if plate_task:
                            self.bp.bot_assignments[bot_id] = plate_task
                        else:
                            # Not full, place on assembly counter if empty
                            if not tile.item or tile.item == held:
                                self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, assembly_loc, item="PLATE")
                            else:
                                # Search for ANY counter
                                for c in self.bp.counters:
                                    tc = controller.get_tile(controller.get_team(), c[0], c[1])
                                    if not tc.item:
                                        self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, c, item="PLATE")
                                        break
                    if bot_id in self.bp.bot_assignments:
                        available_bots.remove(bot_id)
                        continue
                
                # If holding ingredient, process it
                if isinstance(held, Ingredient):
                    if held.state == ItemState.COOKED or held.name in ["NOODLES", "SAUCE"]:
                        if has_plate and best_plate_held_by is None:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.PLATE, assembly_loc, item=held)
                            available_bots.remove(bot_id)
                            log(f"Bot {bot_id} ASSIGNED PLATE @ {assembly_loc}")
                            continue
                    elif held.state == ItemState.RAW and held.name in ["MEAT", "ONIONS"]:
                        # CHOP
                        target = None
                        # ONLY use counters for chopping!
                        for c in self.bp.counters:
                            if c == assembly_loc and (has_plate or occupied_by_food): continue
                            tc = controller.get_tile(controller.get_team(), c[0], c[1])
                            if tc and tc.item is None: target = c; break
                        
                        if not target:
                            target = assembly_loc
                        
                        self.bp.bot_assignments[bot_id] = Task(TaskType.CHOP, target, item=held)
                        available_bots.remove(bot_id)
                        continue
                    elif (held.state == ItemState.CHOPPED) or (held.name == "EGG" and held.state == ItemState.RAW):
                        # COOK
                        cooker = self.find_free_cooker(controller)
                        if cooker:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.COOK, cooker, item=held)
                            available_bots.remove(bot_id)
                            continue

            # Idle bot work
            if sorted_missing := sorted(missing_reqs, key=lambda r: 0 if r.name in ["MEAT", "EGG", "ONIONS"] else 1):
                for req in sorted_missing:
                    # Coordination check: Is this ingredient ALREADY being handled?
                    is_handled = False
                    for bid in bots:
                        h = check_holding(bid)
                        if isinstance(h, Ingredient) and h.name == req.name: is_handled = True; break
                        t = self.bp.bot_assignments.get(bid)
                        if t and t.item and isinstance(t.item, Ingredient) and t.item.name == req.name: is_handled = True; break
                    
                    if is_handled: continue

                    # Supply chain logic
                    st = controller.get_bot_state(bot_id)
                    bot_pos = (st['x'], st['y'])
                    for state, ttype in [("COOKED", TaskType.FETCH_COOKED), ("CHOPPED", TaskType.PICKUP), ("RAW", TaskType.CHOP)]:
                        loc = find_item_loc(f"{req.name}({state})", bot_pos)
                        if loc:
                            self.bp.bot_assignments[bot_id] = Task(ttype, loc, item=req)
                            log(f"Bot {bot_id} ASSIGNED {ttype} @ {loc}")
                            break
                    if bot_id in self.bp.bot_assignments: break
                    
                    # BUY
                    cost = getattr(FoodType, req.name).buy_cost if hasattr(FoodType, req.name) else 80
                    if controller.get_team_money(controller.get_team()) >= cost:
                        shop_loc = self.bp.shops.get(req.name, self.bp.shops.get("GENERAL"))
                        if shop_loc:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.BUY, shop_loc, item=req)
                    if bot_id in self.bp.bot_assignments: break
            
            # Buy Plate - deadlock avoidance
            # On single-counter maps, don't buy plate until CHOP is done
            if not has_plate and all_choppable_ready and not self.bp.bot_assignments.get(bot_id):
                # If holding something, we MUST drop it to buy a plate
                if held:
                    # Drop on ANY free counter/placeable
                    drop_loc = None
                    for loc in self.bp.all_placeable:
                        tc = controller.get_tile(controller.get_team(), loc[0], loc[1])
                        if tc and tc.item is None:
                            drop_loc = loc; break
                    
                    if drop_loc:
                        self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, drop_loc, item=held)
                else:
                    # Check if anyone is already buying a plate
                    if not any(t and t.type == TaskType.BUY and t.item == "PLATE" for t in self.bp.bot_assignments.values()):
                        shop_loc = self.bp.shops.get("PLATE", self.bp.shops.get("GENERAL"))
                        if shop_loc:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.BUY, shop_loc, item="PLATE")

            if bot_id in self.bp.bot_assignments: available_bots.remove(bot_id)
        


# --- 5. Main Bot Logic ---

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # POIs
        self.counters = []
        self.cookers = []
        self.sinks = []
        self.submit_locs = []
        self.shops = []
        self.trash_locs = []
        
        # Components
        self.reservations = ReservationTable()
        self.order_manager = OrderManager()
        self.allocator = TaskAllocator(self)
        
        self.bot_assignments: Dict[int, Optional[Task]] = {}
        
    def _init_map(self, controller):
        # Scan map for key features
        self.counters = []
        self.cookers = []
        self.sinks = [] # Re-initialize here to ensure it's populated
        self.submit_locs = []
        self.shops = {} # name -> (x,y)
        self.trash_locs = [] # Re-initialize here
        self.all_placeable = []
        
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                p = (x, y)
                if tile.tile_name == "COUNTER":
                    self.counters.append(p)
                elif tile.tile_name == "COOKER":
                    self.cookers.append(p)
                elif tile.tile_name == "SINK":
                    self.sinks.append(p)
                elif tile.tile_name == "SUBMIT":
                    self.submit_locs.append(p)
                elif tile.tile_name == "SHOP":
                    self.shops["GENERAL"] = p
                    # Safely handle specific shop items if they exist
                    if hasattr(tile, 'item_name'):
                        self.shops[tile.item_name] = p
                elif tile.tile_name == "TRASH":
                    self.trash_locs.append(p)
                
                # Check is_placeable
                if getattr(tile, 'is_placeable', False) or tile.tile_name in ["COUNTER", "BOX", "SINK", "COOKER", "TRASH", "SUBMIT"]:
                    self.all_placeable.append(p)
        
        # Sort counters by proximity to center or something? 
        # For now, just keep them as they are.
        if not self.counters and self.all_placeable:
            # If no "COUNTER" but other placeable, use those as counters
            self.counters = [self.all_placeable[0]]
        
        self.initialized = True
        log(f"Init complete: {len(self.counters)} counters, {len(self.cookers)} cookers")

    def execute_task(self, controller, bot_id, task):
        """Standard execution logic for a task."""
        if not task: return
        
        state = controller.get_bot_state(bot_id)
        pos = (state['x'], state['y'])
        
        # Check proximity
        dist = max(abs(pos[0] - task.target_loc[0]), abs(pos[1] - task.target_loc[1]))
        
        if dist <= 1:
            # We are ready to interact
            success = False
            
            if task.type == TaskType.BUY:
                name = task.item.name if isinstance(task.item, Ingredient) else str(task.item)
                if hasattr(FoodType, name):
                    success = controller.buy(bot_id, getattr(FoodType, name), task.target_loc[0], task.target_loc[1])
                elif hasattr(ShopCosts, name):
                    success = controller.buy(bot_id, getattr(ShopCosts, name), task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} BOUGHT {name}")

            elif task.type == TaskType.CHOP:
                if state['holding']:
                     success = controller.place(bot_id, task.target_loc[0], task.target_loc[1])
                     if success: log(f"Bot {bot_id} PLACED for CHOP")
                     return 
                success = controller.chop(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} CHOPPED")

            elif task.type == TaskType.COOK:
                success = controller.start_cook(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} STARTED COOKING")

            elif task.type == TaskType.FETCH_COOKED:
                success = controller.take_from_pan(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} TOOK FROM PAN")

            elif task.type == TaskType.PLATE:
                success = controller.add_food_to_plate(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PLATED FOOD")

            elif task.type == TaskType.PICKUP:
                success = controller.pickup(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PICKED UP item")

            elif task.type == TaskType.MOVE_ITEM:
                success = controller.place(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PLACED {task.item}")

            elif task.type == TaskType.DELIVER:
                success = controller.submit(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} SUBMITTED ORDER")
            
            # If interaction attempted but failed, we might need to clear task or wait
            if not success:
                log(f"Bot {bot_id} FAILED action {task.type} @ {task.target_loc}, clearing task")
                self.bot_assignments[bot_id] = None
                return
            else:
                self.bot_assignments[bot_id] = None # Task done
                return

        # Move to target
        path = AStar.get_path(controller, pos, task.target_loc, 0, self.reservations, bot_id)
        if path is not None:
            if not path: # Empty path = at target (but dist check failed?)
                 pass # Should have been handled by dist<=1 check
            else:
                dx, dy = path[0]
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    self.reservations.reserve(path, 0, bot_id) # Reserve the path
        else:
            # Wiggle or re-plan
            log(f"Bot {bot_id} NO PATH to {task}")
            pass

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map(controller)
            
        current_turn = controller.get_turn()
        bots = controller.get_team_bot_ids(controller.get_team())
        self.reservations.clear()
        
        # 1. Update Goals
        self.order_manager.update_goals(controller, current_turn)
        
        # 2. Assign Tasks
        self.allocator.assign_tasks(controller, bots)
        
        # 3. Execute
        for bot_id in bots:
            task = self.bot_assignments.get(bot_id)
            if task:
                self.execute_task(controller, bot_id, task)
            else:
                 # Wiggle
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    if controller.can_move(bot_id, dx, dy):
                        controller.move(bot_id, dx, dy)
                        break