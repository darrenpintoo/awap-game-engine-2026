
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
            
            # Check target reached
            dist = abs(cx - target[0]) + abs(cy - target[1])
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
            
            if task.type == TaskType.PICKUP:
                if not tile or not tile.item: valid = False
            elif task.type == TaskType.CHOP:
                if not tile or not tile.item: valid = False
            elif task.type == TaskType.COOK:
                if not tile or not isinstance(tile.item, Pan) or not tile.item.food: valid = False
            elif task.type == TaskType.PLATE:
                if not tile or not isinstance(tile.item, Plate): valid = False
            
            if not valid:
                log(f"Bot {bot_id} CLEARING INVALID task {task.type}")
                self.bp.bot_assignments[bot_id] = None

    def assign_tasks(self, controller, bots):
        available_bots = [b for b in bots if self.bp.bot_assignments.get(b) is None]
        if not available_bots: return
        
        if not self.bp.order_manager.active_goals: return
        top_goal = self.bp.order_manager.active_goals[0]
        
        # Current world state
        world_items = self.get_available_items(controller)
        
        # Helper to find closest item
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
        
        # 0. Validate existing tasks
        self._validate_tasks(controller, bots, world_items)

        # Needs Analysis
        plate_contents = []
        has_plate = False
        tile = controller.get_tile(controller.get_team(), assembly_loc[0], assembly_loc[1])
        if tile and tile.item and isinstance(tile.item, Plate):
            has_plate = True
            for f in tile.item.food:
                plate_contents.append(f.food_name)
        
        missing_reqs = [r for r in top_goal.requirements if r.name not in plate_contents]

        # Ingredient State Check
        all_ingredients_ready = True
        ingredients_to_process = []
        for req in top_goal.requirements:
            ingredient_name = req.name
            needs_cooking = ingredient_name in ["MEAT", "EGG", "ONIONS"]
            cooked_key = f"{ingredient_name}(COOKED)"
            cooking_key = f"{ingredient_name}(COOKING)"
            chopped_key = f"{ingredient_name}(CHOPPED)"
            raw_key = f"{ingredient_name}(RAW)"

            if needs_cooking:
                if cooked_key in world_items: pass
                elif cooking_key in world_items: pass
                elif chopped_key in world_items:
                    all_ingredients_ready = False
                    ingredients_to_process.append((ingredient_name, "CHOPPED"))
                elif raw_key in world_items:
                    all_ingredients_ready = False
                    ingredients_to_process.append((ingredient_name, "RAW"))
                else:
                    held_by_someone = False
                    for bid in bots:
                        h = check_holding(bid)
                        if isinstance(h, Ingredient) and h.name == ingredient_name:
                            held_by_someone = True
                            if h.state not in [ItemState.CHOPPED, ItemState.COOKED]:
                                all_ingredients_ready = False
                            break
                    if not held_by_someone:
                        all_ingredients_ready = False
                        ingredients_to_process.append((ingredient_name, "NONE"))
            else:
                if ingredient_name not in plate_contents:
                    if raw_key not in world_items:
                        held_by_someone = False
                        for bid in bots:
                            h = check_holding(bid)
                            if h and (isinstance(h, Ingredient) and h.name == ingredient_name):
                                held_by_someone = True
                                break
                        if not held_by_someone:
                            all_ingredients_ready = False
                            ingredients_to_process.append((ingredient_name, "NONE"))

        # --- ASSEMBLY COUNTER MANAGEMENT ---
        occupied_by_food = False
        tile_item = None
        if tile and tile.item:
            if isinstance(tile.item, Food):
                occupied_by_food = True
                tile_item = tile.item
            elif not isinstance(tile.item, Plate):
                occupied_by_food = True
        
        # --- SUBMISSION DETECTION ---
        # If the plate matches the top goal, assign a DELIVER task
        if has_plate and not missing_reqs:
            # Check if any bot is assigned to deliver or already holding it
            assigned_deliver = False
            for t in self.bp.bot_assignments.values():
                if t and t.type == TaskType.DELIVER:
                    assigned_deliver = True
                    break
            
            if not assigned_deliver:
                # Find best bot to deliver (prioritize one already holding plate, else nearest)
                best_bot = None
                for bid in available_bots:
                    if check_holding(bid) == "PLATE":
                        best_bot = bid
                        break
                
                if best_bot is None and available_bots:
                    # Just pick the first available for now
                    best_bot = available_bots[0]
                
                if best_bot is not None:
                    if check_holding(best_bot) == "PLATE":
                        # We have it, deliver it
                        target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                        self.bp.bot_assignments[best_bot] = Task(TaskType.DELIVER, target, item="PLATE")
                        log(f"Bot {best_bot} DELIVERING full plate")
                    else:
                        # Need to pick it up first
                        self.bp.bot_assignments[best_bot] = Task(TaskType.PICKUP, assembly_loc)
                        log(f"Bot {best_bot} FETCHING full plate for delivery")
                    
                    available_bots.remove(best_bot)
        
        # If specifically ready for a plate, assign clear-counter if needed
        is_ready_for_plate = not has_plate and all_ingredients_ready and available_bots
        if is_ready_for_plate and occupied_by_food:
            clearing = False
            for bid, t in self.bp.bot_assignments.items():
                if t and t.type == TaskType.PICKUP and t.target_loc == assembly_loc:
                    clearing = True
                    break
            
            if not clearing:
                bot_id = available_bots[0] 
                self.bp.bot_assignments[bot_id] = Task(TaskType.PICKUP, assembly_loc)
                log(f"Bot {bot_id} CLEARING blocker from assembly counter")
                available_bots.remove(bot_id)

        # Greedy assignment
        for bot_id in available_bots:
            held = check_holding(bot_id)
            if held:
                # 1.1 Priority: If assigned to DELIVER, don't change it
                current_task = self.bp.bot_assignments.get(bot_id)
                if current_task and current_task.type == TaskType.DELIVER:
                    continue

                if str(held) == "PLATE":
                    # If this plate is full (no missing reqs), keep delivering it
                    if not missing_reqs:
                        target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                        self.bp.bot_assignments[bot_id] = Task(TaskType.DELIVER, target, item="PLATE")
                        log(f"Bot {bot_id} holding FULL PLATE, task: DELIVER")
                        continue

                    # Otherwise, place it on assembly if needed
                    # but only if it's not already there
                    self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, assembly_loc, item="PLATE")
                    log(f"Bot {bot_id} holding PLATE, task: PLACE on assembly")
                    continue
                
                # --- PAN HANDLING ---
                if isinstance(held, str) and held == "PAN":
                    # Just ignore for now
                    continue
                    
                if isinstance(held, Ingredient):
                    # We have an ingredient. What to do with it?
                    # Check Recipe Graph forward
                    # Forward planning: held -> next_state
                    
                    # Special case: If cooked, PLATE IT
                    if held.state == ItemState.COOKED:
                        self.bp.bot_assignments[bot_id] = Task(TaskType.PLATE, assembly_loc, item=held)
                        log(f"Bot {bot_id} has COOKED {held.name}, task: PLATE")
                        continue
                        
                    # If chopped, COOK IT (if needed)
                    if held.state == ItemState.CHOPPED:
                        # Check if this ingredient needs cooking
                        # TODO: Check RecipeGraph. For now hardcode: MEAT/EGG/ONIONS need cook
                        if held.name in ["MEAT", "EGG", "ONIONS"]:
                             # Find empty cooker
                             cooker = self.bp.cookers[0] # ToDo: Find empty
                             self.bp.bot_assignments[bot_id] = Task(TaskType.COOK, cooker, item=held)
                             log(f"Bot {bot_id} has CHOPPED {held.name}, task: COOK")
                             continue
                        else:
                             # Just plate it? (e.g. Chopped Salad?)
                             pass
                             
                    # If raw, CHOP IT (if needed)
                    if held.state == ItemState.RAW:
                        # Check if needs chop
                        if held.name in ["MEAT", "ONIONS"]:
                            # Find counter to chop
                            # If Staging Counter exists use it
                            target = self.bp.counters[1] if len(self.bp.counters)>1 else self.bp.counters[0]
                            # Put it down first? execute_task handles CHOP logic?
                            # No, CHOP action implies being at counter.
                            # But we need to PLACE then CHOP.
                            # Actually, controller.chop() works on item on counter.
                            # So we need 2 tasks: PLACE -> CHOP.
                            # Let's assign PLACE first.
                            self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, target, item=held)
                            # Wait, execute_task for MOVE_ITEM needs implementation
                            # Let's simpler: Just use CHOP task which implies Place+Chop sequence?
                            # For now: PLACE
                            # Actually, let's implement CHOP as "Go to counter, Place, Chop"
                            # But execute_task is one-step.
                            
                            # Hack: Assign CHOP task. execute_task will see holding, Place, then next turn Chop.
                            # We need execute_task to handle this.
                            self.bp.bot_assignments[bot_id] = Task(TaskType.CHOP, target, item=held)
                            log(f"Bot {bot_id} has RAW {held.name}, task: CHOP")
                            continue
                            
                        elif held.name in ["NOODLES", "SAUCE"]:
                            # Only plate if plate exists on assembly
                            if has_plate:
                                self.bp.bot_assignments[bot_id] = Task(TaskType.PLATE, assembly_loc, item=held)
                                log(f"Bot {bot_id} has RAW {held.name}, task: PLATE directly")
                            else:
                                # No plate yet - only place on counter if we have multiple counters
                                # On single-counter maps, keep holding it so we don't block the plate
                                if len(self.bp.counters) > 1:
                                    target_counter = None
                                    for cx, cy in self.bp.counters:
                                        if (cx, cy) != assembly_loc:
                                            tile_check = controller.get_tile(controller.get_team(), cx, cy)
                                            if tile_check and not tile_check.item:
                                                target_counter = (cx, cy)
                                                break
                                    
                                    if target_counter:
                                        self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, target_counter, item=held)
                                        log(f"Bot {bot_id} PLACING {held.name} on counter (no plate yet)")
                                    else:
                                        log(f"Bot {bot_id} HOLDING {held.name} (all counters full)")
                                else:
                                    log(f"Bot {bot_id} HOLDING {held.name} (single counter map - don't block plate)")
                            continue
                             
                        elif held.name == "EGG":
                            # Egg goes straight to cook
                            cooker = self.bp.cookers[0]
                            self.bp.bot_assignments[bot_id] = Task(TaskType.COOK, cooker, item=held)
                            log(f"Bot {bot_id} has RAW EGG, task: COOK")
                            continue

            # 2. If idle (not holding anything), check what work remains
            
            # DECISION: Only buy/place plate if all ingredients that need cooking ARE in world (cooked or cooking)
            if not has_plate and all_ingredients_ready:
                # NOW it's safe to buy plate - ingredients won't need the counter
                found_plate = "PLATE" in [str(check_holding(b)) for b in bots]
                if not found_plate:
                    # Budget check for plate
                    team_money = controller.get_team_money(controller.get_team())
                    plate_cost = ShopCosts.PLATE.buy_cost
                    
                    if team_money >= plate_cost:
                        shop = self.bp.shops[0]
                        self.bp.bot_assignments[bot_id] = Task(TaskType.BUY, shop, item="PLATE")
                        log(f"Bot {bot_id} BUYING PLATE (all ingredients ready)")
                        continue
                    else:
                        log(f"Bot {bot_id} CAN'T AFFORD PLATE (${team_money} < ${plate_cost})")
            
            # Process ingredients that need work
            # Sort missing_reqs: cookable ingredients FIRST (they need counter for chopping)
            # This prevents blocking the counter with simple ingredients like NOODLES
            cookable_names = ["MEAT", "EGG", "ONIONS"]
            sorted_missing = sorted(missing_reqs, key=lambda r: 0 if r.name in cookable_names else 1)
            
            if sorted_missing:
                # Try each ingredient in priority order until we find work
                for target_req in sorted_missing:
                    ingredient_name = target_req.name
                    
                    # Check if this ingredient is COOKING (in pan, not ready yet)
                    cooking_key = f"{ingredient_name}(COOKING)"
                    if cooking_key in world_items:
                        # This ingredient is being processed, skip to next
                        continue
                    
                    # Supply chain logic
                    found_task = None
                    
                    # 1. Check for COOKED (ready to plate)
                    cooked_key = f"{ingredient_name}(COOKED)"
                    if cooked_key in world_items:
                        # Only pick up if we have a plate to put it on
                        if has_plate:
                            loc = world_items[cooked_key][0]
                            found_task = Task(TaskType.FETCH_COOKED, loc)
                            log(f"Bot {bot_id} FETCHING cooked {ingredient_name}")
                    
                    # 2. Check for CHOPPED (needs cooking)
                    if not found_task:
                        chopped_key = f"{ingredient_name}(CHOPPED)"
                        if chopped_key in world_items:
                            loc = world_items[chopped_key][0]
                            # Go pickup and cook it
                            found_task = Task(TaskType.PICKUP, loc)  # Generic pickup
                            log(f"Bot {bot_id} PICKING UP chopped {ingredient_name} to cook")
                    
                    # 3. Check for RAW on counter (needs chopping or pickup)
                    if not found_task:
                        raw_key = f"{ingredient_name}(RAW)"
                        if raw_key in world_items:
                            loc = world_items[raw_key][0]
                            # Only CHOP if ingredient is choppable
                            if ingredient_name in ["MEAT", "ONIONS"]:
                                found_task = Task(TaskType.CHOP, loc, item=Ingredient(ingredient_name, ItemState.RAW))
                                log(f"Bot {bot_id} CHOPPING raw {ingredient_name}")
                            else:
                                # NOODLES/SAUCE/EGG - pick up directly to add to plate or cook
                                found_task = Task(TaskType.PICKUP, loc)  # Generic pickup
                                log(f"Bot {bot_id} PICKING UP raw {ingredient_name}")
                    
                    # 4. Buy new raw ingredient (only if not already in world/cooking/holding)
                    if not found_task:
                        # Coordination check: is someone else ALREADY handling this ingredient?
                        is_being_handled = False
                        
                        # Check world
                        if cooking_key in world_items or cooked_key in world_items:
                            is_being_handled = True
                        
                        # Check check holdings
                        if not is_being_handled:
                            for bid in bots:
                                h = check_holding(bid)
                                if isinstance(h, Ingredient) and h.name == ingredient_name:
                                    is_being_handled = True
                                    break
                        
                        # Check assignments
                        if not is_being_handled:
                            for t in self.bp.bot_assignments.values():
                                if t and t.item:
                                    if (isinstance(t.item, Ingredient) and t.item.name == ingredient_name) or \
                                       (isinstance(t.item, str) and t.item == ingredient_name):
                                        is_being_handled = True
                                        break
                        
                        if not is_being_handled:
                            # Budget check
                            team_money = controller.get_team_money(controller.get_team())
                            cost = getattr(FoodType, ingredient_name).buy_cost if hasattr(FoodType, ingredient_name) else 80
                            
                            if team_money >= cost:
                                shop = self.bp.shops[0]
                                found_task = Task(TaskType.BUY, shop, item=Ingredient(ingredient_name, ItemState.RAW))
                                log(f"Bot {bot_id} BUYING raw {ingredient_name}")
                            else:
                                log(f"Bot {bot_id} CAN'T AFFORD {ingredient_name}")
                    
                    if found_task:
                        self.bp.bot_assignments[bot_id] = found_task
                        break  # Found work, stop looking


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
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                t = m.tiles[x][y].tile_name
                p = (x, y)
                if t == "COUNTER": self.counters.append(p)
                elif t == "COOKER": self.cookers.append(p)
                elif t == "SINK": self.sinks.append(p)
                elif t == "SUBMIT": self.submit_locs.append(p)
                elif t == "SHOP": self.shops.append(p)
                elif t == "TRASH": self.trash_locs.append(p)
        
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
                # Need to lookup item enum
                # task.item is Ingredient or str
                name = task.item.name if isinstance(task.item, Ingredient) else str(task.item)
                
                # Check Food types
                if hasattr(FoodType, name):
                    success = controller.buy(bot_id, getattr(FoodType, name), task.target_loc[0], task.target_loc[1])
                # Check Shop items
                elif hasattr(ShopCosts, name):
                    success = controller.buy(bot_id, getattr(ShopCosts, name), task.target_loc[0], task.target_loc[1])
                
                if success: log(f"Bot {bot_id} BOUGHT {name}")

            elif task.type == TaskType.CHOP:
                if state['holding']:
                     # Must place first
                     success = controller.place(bot_id, task.target_loc[0], task.target_loc[1])
                     if success: log(f"Bot {bot_id} PLACED for CHOP")
                     # Task not done yet! Return so we chop next turn.
                     return 
                     
                success = controller.chop(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} CHOPPED")

            elif task.type == TaskType.COOK:
                # Start cooking (Holding food -> Place & Cook at same time)
                success = controller.start_cook(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} STARTED COOKING")

            elif task.type == TaskType.FETCH_COOKED:
                success = controller.take_from_pan(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} TOOK FROM PAN")

            elif task.type == TaskType.PLATE:
                success = controller.add_food_to_plate(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PLATED FOOD")

            elif task.type == TaskType.PICKUP:
                # Pickup from map
                success = controller.pickup(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PICKED UP item")

            elif task.type == TaskType.MOVE_ITEM:
                # Place held item on target location
                success = controller.place(bot_id, task.target_loc[0], task.target_loc[1])
                if success: 
                    log(f"Bot {bot_id} PLACED {task.item}")
                else:
                    # Place failed (likely counter occupied) - clear task so we get reassigned
                    log(f"Bot {bot_id} FAILED to place {task.item}, clearing task")
                    self.bot_assignments[bot_id] = None
                    return

            elif task.type == TaskType.DELIVER:
                success = controller.submit(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} SUBMITTED ORDER")
            
            # If successful, clear assignment (task done)
            if success:
                self.bot_assignments[bot_id] = None
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