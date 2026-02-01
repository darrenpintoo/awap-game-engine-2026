
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

DEBUG = False

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

class JointAStar:
    @staticmethod
    def get_path_pair(controller, bots: List[int], targets: Dict[int, Tuple[int, int]], 
                      stop_dists: Dict[int, int], obstacles: Set[Tuple[int, int]] = None) -> Dict[int, Tuple[int, int]]:
        """
        Returns {bot_id: (dx, dy)} for the next move.
        If only 1 bot, falls back to simple A*.
        """
        if not bots: return {}
        if obstacles is None: obstacles = set()
        
        m = controller.get_map(controller.get_team())
        w, h = m.width, m.height
        
        # Helper: Get moves for a bot
        MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
        
        # 1. Single Bot Case (Optimization)
        if len(bots) == 1:
             bid = bots[0]
             start = controller.get_bot_state(bid)
             spos = (start['x'], start['y'])
             tgt = targets.get(bid, spos)
             dist = stop_dists.get(bid, 0)
             
             # Use simple BFS/A* since no collision check needed
             cx, cy = spos
             if max(abs(cx - tgt[0]), abs(cy - tgt[1])) <= dist: return {bid: (0,0)}
             
             open_set = [(0, 0, cx, cy, [])]
             visited = {(cx, cy): 0}
             
             while open_set:
                 _, g, curr_x, curr_y, path = heapq.heappop(open_set)
                 
                 if max(abs(curr_x - tgt[0]), abs(curr_y - tgt[1])) <= dist:
                     return {bid: path[0]} if path else {bid: (0,0)}

                 if g > 20: continue # Limit depth
                 
                 for dx, dy in MOVES:
                     nx, ny = curr_x + dx, curr_y + dy
                     if m.is_tile_walkable(nx, ny) and (nx, ny) not in obstacles:
                         if (nx, ny) not in visited or visited[(nx,ny)] > g+1:
                             visited[(nx,ny)] = g+1
                             h_score = max(abs(nx - tgt[0]), abs(ny - tgt[1]))
                             heapq.heappush(open_set, (g+1+h_score, g+1, nx, ny, path + [(dx, dy)]))
             return {bid: (0,0)}

        # 2. Joint Case
        b1, b2 = bots[0], bots[1]
        s1 = controller.get_bot_state(b1); p1 = (s1['x'], s1['y'])
        s2 = controller.get_bot_state(b2); p2 = (s2['x'], s2['y'])
        t1 = targets.get(b1, p1); d1 = stop_dists.get(b1, 0)
        t2 = targets.get(b2, p2); d2 = stop_dists.get(b2, 0)
        
        # State: (x1, y1, x2, y2)
        start_state = (*p1, *p2)
        
        # (f, g, state, first_move_moves)
        open_set = [(0, 0, start_state, None)]
        visited = {start_state: 0}
        
        # Heuristic
        def h_val(state):
             d_b1 = max(abs(state[0]-t1[0]), abs(state[1]-t1[1]))
             d_b2 = max(abs(state[2]-t2[0]), abs(state[3]-t2[1]))
             return d_b1 + d_b2

        while open_set:
            f, g, state, first_moves = heapq.heappop(open_set)
            
            x1, y1, x2, y2 = state
            
            # Check goals
            reached1 = max(abs(x1 - t1[0]), abs(y1 - t1[1])) <= d1
            reached2 = max(abs(x2 - t2[0]), abs(y2 - t2[1])) <= d2
            
            if reached1 and reached2:
                return {b1: first_moves[0], b2: first_moves[1]} if first_moves else {b1:(0,0), b2:(0,0)}
            
            if g > 25: # Depth limit increased for complex swaps
                 continue

            # Generate neighbors - Cartesian Product of moves
            # Allow movement even if reached (Yielding logic)
            moves1 = MOVES 
            moves2 = MOVES
            
            for m1 in moves1:
                # Check engine validity for immediate move (handles dynamic entities not in obstacles set)
                if g == 0 and m1 != (0,0):
                     if not controller.can_move(b1, m1[0], m1[1]): continue

                nx1, ny1 = x1 + m1[0], y1 + m1[1]
                if not m.is_tile_walkable(nx1, ny1) or (nx1, ny1) in obstacles: continue
                
                for m2 in moves2:
                    # Check engine validity for immediate move
                    if g == 0 and m2 != (0,0):
                        if len(bots) > 1 and not controller.can_move(b2, m2[0], m2[1]): continue
                    
                    nx2, ny2 = x2 + m2[0], y2 + m2[1]
                    if not m.is_tile_walkable(nx2, ny2) or (nx2, ny2) in obstacles: continue
                    
                    # Collision Checks
                    if nx1 == nx2 and ny1 == ny2: continue # Node collision
                    if nx1 == x2 and ny1 == y2 and nx2 == x1 and ny2 == y1: continue # Swap collision
                    
                    new_state = (nx1, ny1, nx2, ny2)
                    new_g = g + 1
                    
                    if new_state not in visited or visited[new_state] > new_g:
                        visited[new_state] = new_g
                        hv = h_val(new_state)
                        f_score = new_g + hv
                        
                        fm = first_moves if first_moves else (m1, m2)
                        heapq.heappush(open_set, (f_score, new_g, new_state, fm))
                        
        return {b1:(0,0), b2:(0,0)} # Fail


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
                reqs.append(Ingredient(name, ItemState.PLATED))
                
            rem_time = order['expires_turn'] - current_turn
            if rem_time < 20: continue # Skip if expiring soon
            
            goal = Goal(order['order_id'], reqs, order['reward'], rem_time)
            # Heuristic score: Reward / Time remaining
            # Prioritize high reward, urgent orders
            time_factor = max(1, goal.time_left)
            goal.score = (goal.reward ** 2) / time_factor
            
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
                # Valid if tile has pan AND pan is empty AND bot is holding food
                if not tile or not isinstance(tile.item, Pan): valid = False
                elif tile.item.food: valid = False # Pan is occupied
                elif not held or held.get('type') != 'Food': valid = False # Must hold FOOD
            elif task.type == TaskType.PLATE:
                # If holding plate, we are plating FROM the world (target=Food)
                if held and held.get('type') == 'Plate':
                    if not tile or not isinstance(tile.item, (Ingredient, Food)): valid = False
                # If holding food, we are plating ONTO a plate (target=Plate)
                else: 
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
        has_orders = bool(self.bp.order_manager.active_goals)

        # Get world state
        world_items = self.get_available_items(controller)
        available_bots = [bid for bid in bots if not self.bp.bot_assignments.get(bid)]
        
        reserved_locs = set()
        for task in self.bp.bot_assignments.values():
            if task and task.target_loc:
                reserved_locs.add(task.target_loc)
        
        def find_item_loc(ing, bot_pos):
            locs = world_items.get(str(ing), [])
            if not locs: return None
            # Filter reserved
            valid = [l for l in locs if l not in reserved_locs]
            if not valid: return None
            return min(valid, key=lambda p: max(abs(p[0]-bot_pos[0]), abs(p[1]-bot_pos[1])))

        def check_holding(bot_id):
            st = controller.get_bot_state(bot_id)
            if not st['holding']: return None
            h = st['holding']
            if h['type'] == 'Food':
                state = ItemState.RAW
                if h.get('chopped'): state = ItemState.CHOPPED
                if h.get('cooked_stage') == 1: state = ItemState.COOKED
                return Ingredient(h['food_name'], state)
            elif h['type'] == 'Pan': return "PAN"
            elif h['type'] == 'Plate': return "PLATE"
            return None

        assembly_loc = self.bp.counters[0] if self.bp.counters else None
        if not assembly_loc: return
        
        self._validate_tasks(controller, bots, world_items)

        # Snapshot Plates
        global_plates = []
        tile = controller.get_tile(controller.get_team(), assembly_loc[0], assembly_loc[1])
        if tile and tile.item and isinstance(tile.item, Plate):
            global_plates.append(([f.food_name for f in tile.item.food], assembly_loc, None))
        for bid in bots:
            st = controller.get_bot_state(bid)
            holding = st.get('holding')
            if holding and holding.get('type') == 'Plate':
                contents = [f['food_name'] for f in holding.get('food', [])]
                global_plates.append((contents, (st['x'], st['y']), bid))

        # PIPELINE: Iterate through assignments
        for top_goal in self.bp.order_manager.active_goals:
            if not available_bots: break

            # 1. Best Plate for THIS goal
            best_plate = None
            if global_plates:
                # Prioritize plates that have ingredients for THIS goal
                def plate_score(p):
                     matches = len([f for f in p[0] if any(r.name == f for r in top_goal.requirements)])
                     garbage = len(p[0]) - matches
                     return (matches, -garbage)
                best_plate = max(global_plates, key=plate_score)
            
            plate_contents = best_plate[0] if best_plate else []
            has_plate = best_plate is not None
            best_plate_held_by = best_plate[2] if best_plate else None
            missing_reqs = [r for r in top_goal.requirements if r.name not in plate_contents]

            # 2. Prep Check
            all_choppable_ready = True
            for req in top_goal.requirements:
                if req.name in plate_contents: continue
                if req.name in ["MEAT", "ONIONS"]:
                    found = False
                    for k in [f"{req.name}(COOKED)", f"{req.name}(COOKING)", f"{req.name}(CHOPPED)"]:
                        if k in world_items: found = True; break
                    if not found:
                        for bid in bots:
                            h = check_holding(bid)
                            if isinstance(h, Ingredient) and h.name == req.name and h.state != ItemState.RAW:
                                found = True; break
                    if not found: all_choppable_ready = False; break

            # 3. Blocker
            occupied_by_food = False
            if tile and tile.item and isinstance(tile.item, Food): occupied_by_food = True

            # 4. Delivery
            if has_plate and not missing_reqs:
                if best_plate_held_by in available_bots:
                    target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                    self.bp.bot_assignments[best_plate_held_by] = Task(TaskType.DELIVER, target, item="PLATE")
                    available_bots.remove(best_plate_held_by)
                    reserved_locs.add(target)
                elif best_plate_held_by is None and available_bots:
                    bid = available_bots[0]
                    self.bp.bot_assignments[bid] = Task(TaskType.PICKUP, assembly_loc)
                    available_bots.remove(bid)
                    reserved_locs.add(assembly_loc)
                continue

            # 5. Greedy Assign
            for bot_id in available_bots[:]:
                held = check_holding(bot_id)
                if held:
                    if str(held) == "PLATE":
                        if best_plate_held_by == bot_id:
                            if not missing_reqs:
                                target = self.bp.submit_locs[0] if self.bp.submit_locs else assembly_loc
                                self.bp.bot_assignments[bot_id] = Task(TaskType.DELIVER, target, item="PLATE")
                                available_bots.remove(bot_id)
                                reserved_locs.add(target)
                                continue
                            else:
                                plate_task = None
                                st = controller.get_bot_state(bot_id)
                                for req in missing_reqs:
                                    for state in ["COOKED", "RAW"]:
                                        if state == "RAW" and req.name not in ["NOODLES", "SAUCE"]: continue
                                        loc = find_item_loc(f"{req.name}({state})", (st['x'], st['y']))
                                        if loc and loc not in self.bp.cookers: 
                                             plate_task = Task(TaskType.PLATE, loc, item=req)
                                             break
                                    if plate_task: break
                                
                                if plate_task:
                                    self.bp.bot_assignments[bot_id] = plate_task
                                    available_bots.remove(bot_id)
                                    reserved_locs.add(plate_task.target_loc)
                                    continue
                    elif isinstance(held, Ingredient):
                        needed = False
                        if held.name in [r.name for r in missing_reqs]: needed = True
                        elif held.name in ["MEAT", "ONIONS"] and held.state == ItemState.RAW: needed = True 
                        
                        if needed:
                            if held.state == ItemState.COOKED or held.name in ["NOODLES", "SAUCE"]:
                                if has_plate and best_plate_held_by is None and assembly_loc not in reserved_locs:
                                    self.bp.bot_assignments[bot_id] = Task(TaskType.PLATE, assembly_loc, item=held)
                                    available_bots.remove(bot_id)
                                    reserved_locs.add(assembly_loc)
                                    continue
                                else:
                                    # No plate available - buy one!
                                    buying_plate = any(t and t.type == TaskType.BUY and t.item == "PLATE" for t in self.bp.bot_assignments.values())
                                    if not buying_plate and not has_plate:
                                        shop_loc = self.bp.shops.get("PLATE", self.bp.shops.get("GENERAL"))
                                        if shop_loc:
                                            # Drop item first, then will buy plate next turn
                                            drop_loc = None
                                            for c in self.bp.counters:
                                                if c in reserved_locs: continue
                                                tc = controller.get_tile(controller.get_team(), c[0], c[1])
                                                if tc and tc.item is None: drop_loc = c; break
                                            if drop_loc:
                                                self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, drop_loc, item=held)
                                                available_bots.remove(bot_id)
                                                reserved_locs.add(drop_loc)
                                                continue
                            elif held.state == ItemState.RAW and held.name in ["MEAT", "ONIONS"]:
                                target = None
                                for c in self.bp.counters:
                                    if c == assembly_loc and (has_plate or occupied_by_food): continue
                                    if c in reserved_locs: continue
                                    tc = controller.get_tile(controller.get_team(), c[0], c[1])
                                    if tc and tc.item is None: target = c; break
                                if not target: target = assembly_loc
                                self.bp.bot_assignments[bot_id] = Task(TaskType.CHOP, target, item=held)
                                available_bots.remove(bot_id)
                                if target: reserved_locs.add(target)
                                continue
                            elif (held.state == ItemState.CHOPPED) or (held.name == "EGG" and held.state == ItemState.RAW):
                                cooker = self.find_free_cooker(controller)
                                if cooker and cooker not in reserved_locs:
                                    tt = TaskType.COOK if held.name in ["MEAT", "EGG"] else TaskType.MOVE_ITEM
                                    self.bp.bot_assignments[bot_id] = Task(tt, cooker, item=held)
                                    available_bots.remove(bot_id)
                                    reserved_locs.add(cooker)
                                    continue

                if held: continue
                if sorted_missing := sorted(missing_reqs, key=lambda r: 0 if r.name in ["MEAT", "EGG", "ONIONS"] else 1):
                    assigned_fetch = False
                    for req in sorted_missing:
                        is_handled = False
                        for bid in bots:
                            t = self.bp.bot_assignments.get(bid)
                            if t and t.item and isinstance(t.item, Ingredient) and t.item.name == req.name: is_handled = True; break
                        if is_handled: continue

                        st = controller.get_bot_state(bot_id)
                        bot_pos = (st['x'], st['y'])
                        for state, ttype in [("COOKED", TaskType.FETCH_COOKED), ("CHOPPED", TaskType.PICKUP), ("RAW", TaskType.PICKUP)]:
                            loc = find_item_loc(f"{req.name}({state})", bot_pos)
                            if loc:
                                if has_plate and loc in self.bp.counters: continue
                                if loc in reserved_locs: continue
                                
                                final_type = ttype
                                if state == "COOKED" and loc not in self.bp.cookers: final_type = TaskType.PICKUP
                                # Only MEAT and ONIONS need CHOP, others go straight to plate/cook
                                if state == "RAW" and req.name in ["MEAT", "ONIONS"]: final_type = TaskType.CHOP
                                
                                self.bp.bot_assignments[bot_id] = Task(final_type, loc, item=req)
                                available_bots.remove(bot_id)
                                reserved_locs.add(loc)
                                assigned_fetch = True
                                break
                        if assigned_fetch: break
                    
                    if assigned_fetch: continue
                    
                    # BUY
                    cost = getattr(FoodType, req.name).buy_cost if hasattr(FoodType, req.name) else 80
                    if controller.get_team_money(controller.get_team()) >= cost:
                        shop_loc = self.bp.shops.get(req.name, self.bp.shops.get("GENERAL"))
                        if shop_loc and shop_loc not in reserved_locs:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.BUY, shop_loc, item=req)
                            available_bots.remove(bot_id)
                            continue
                
                # Buy Plate
                if not has_plate and all_choppable_ready and not self.bp.bot_assignments.get(bot_id):
                    buying_plate = any(t and t.type == TaskType.BUY and t.item == "PLATE" for t in self.bp.bot_assignments.values())
                    if not buying_plate:
                        shop_loc = self.bp.shops.get("PLATE", self.bp.shops.get("GENERAL"))
                        if shop_loc:
                            self.bp.bot_assignments[bot_id] = Task(TaskType.BUY, shop_loc, item="PLATE")
                            available_bots.remove(bot_id)
                            continue

        # FALLBACK: Drop ONLY if item is NOT needed by any order
        all_needed_items = set()
        for goal in self.bp.order_manager.active_goals:
            for req in goal.requirements:
                all_needed_items.add(req.name)
        
        for bot_id in available_bots:
            st = controller.get_bot_state(bot_id)
            held = st.get('holding')
            if held:
                # Check if this item is useful
                held_name = held.get('food_name') if held.get('type') == 'Food' else held.get('type')
                if held_name in all_needed_items:
                    # Don't drop useful items - just wait
                    continue
                if held.get('type') == 'Plate':
                    # Don't drop plates either
                    continue
                    
                drop_loc = None
                t_ass = controller.get_tile(controller.get_team(), assembly_loc[0], assembly_loc[1])
                if not t_ass.item and assembly_loc not in reserved_locs:
                     drop_loc = assembly_loc
                else:
                    candidates = [self.bp.counters, self.bp.all_placeable]
                    best_drop = None; min_dist = 999
                    curr_pos = (st['x'], st['y'])
                    for lst in candidates:
                        if best_drop: break
                        for p in lst:
                            if p in reserved_locs: continue
                            t = controller.get_tile(controller.get_team(), p[0], p[1])
                            if not t.item:
                                d = abs(p[0]-curr_pos[0]) + abs(p[1]-curr_pos[1])
                                if d < min_dist: min_dist = d; best_drop = p
                    drop_loc = best_drop
                
                if drop_loc:
                    self.bp.bot_assignments[bot_id] = Task(TaskType.MOVE_ITEM, drop_loc, item=held)


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
        self.order_manager = OrderManager()
        self.allocator = TaskAllocator(self)
        
        self.bot_assignments: Dict[int, Optional[Task]] = {}
        
    def _init_map(self, controller):
        # Scan map for key features
        self.counters = []
        self.cookers = []
        self.sinks = [] 
        self.submit_locs = []
        self.shops = {} 
        self.trash_locs = [] 
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
                    if hasattr(tile, 'item_name'):
                        self.shops[tile.item_name] = p
                elif tile.tile_name == "TRASH":
                    self.trash_locs.append(p)
                
                if getattr(tile, 'is_placeable', False) or tile.tile_name in ["COUNTER", "BOX", "SINK", "COOKER", "TRASH", "SUBMIT"]:
                    self.all_placeable.append(p)
        
        if not self.counters and self.all_placeable:
            self.counters = [self.all_placeable[0]]
        
        self.initialized = True
        log(f"Init complete: {len(self.counters)} counters")

    def try_interact(self, controller, bot_id, task):
        """Attempts to perform the action if in range. Returns True if acted."""
        state = controller.get_bot_state(bot_id)
        pos = (state['x'], state['y'])
        
        # Check proximity
        dist = max(abs(pos[0] - task.target_loc[0]), abs(pos[1] - task.target_loc[1]))
        
        if dist <= 1:
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
                     else: self.bot_assignments[bot_id] = None  # Clear if place fails
                     return True
                success = controller.chop(bot_id, task.target_loc[0], task.target_loc[1])
                if success: 
                    log(f"Bot {bot_id} CHOPPED")
                else:
                    log(f"Bot {bot_id} FAILED CHOP, clearing task")
                    self.bot_assignments[bot_id] = None

            elif task.type == TaskType.COOK:
                success = controller.start_cook(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} STARTED COOKING")
                else: 
                     log(f"Bot {bot_id} FAILED action {task}, clearing task")
                     self.bot_assignments[bot_id] = None

            elif task.type == TaskType.FETCH_COOKED:
                success = controller.take_from_pan(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} TOOK FROM PAN")
                else: self.bot_assignments[bot_id] = None

            elif task.type == TaskType.PLATE:
                success = controller.add_food_to_plate(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PLATED FOOD")
                else: self.bot_assignments[bot_id] = None

            elif task.type == TaskType.PICKUP:
                success = controller.pickup(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PICKED UP item")
                else: self.bot_assignments[bot_id] = None

            elif task.type == TaskType.MOVE_ITEM:
                success = controller.place(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} PLACED {task.item}")
                else: self.bot_assignments[bot_id] = None

            elif task.type == TaskType.DELIVER:
                success = controller.submit(bot_id, task.target_loc[0], task.target_loc[1])
                if success: log(f"Bot {bot_id} SUBMITTED ORDER")
                else: self.bot_assignments[bot_id] = None
            
            if success:
                self.bot_assignments[bot_id] = None # Task done
            
            # Whether success or fail, we attempted an action.
            # If we failed but were in range, we basically wasted an action (or need to wait).
            # Return True to indicate "we are at target".
            return True
            
        return False

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map(controller)
            
        current_turn = controller.get_turn()
        bots = controller.get_team_bot_ids(controller.get_team())
        
        # 1. Update Goals
        self.order_manager.update_goals(controller, current_turn)
        
        # 2. Assign Tasks
        self.allocator.assign_tasks(controller, bots)
        
        # 3. Execution & Movement Phase
        
        # Gather targets for movement
        move_targets = {}
        stop_dists = {}
        bots_needing_move = []
        
        for bot_id in bots:
            task = self.bot_assignments.get(bot_id)
            if task:
                # Try interacting first
                acted = self.try_interact(controller, bot_id, task)
                if not acted:
                    # If not acted (too far), needs move
                    move_targets[bot_id] = task.target_loc
                    stop_dists[bot_id] = 1
                    bots_needing_move.append(bot_id)
            else:
                # Idle handling? Just stay put.
                pass
                
        # 4. Joint Pathfinding
        if bots_needing_move:
            moves = JointAStar.get_path_pair(controller, bots_needing_move, move_targets, stop_dists)
            
            pending_moves = []
            for bid, (dx, dy) in moves.items():
                if dx == 0 and dy == 0: continue
                pending_moves.append((bid, dx, dy))
            
            passes = 0
            while pending_moves and passes < 3:
                passes += 1
                next_pending = []
                progress = False
                
                for bid, dx, dy in pending_moves:
                    if controller.can_move(bid, dx, dy):
                         if controller.move(bid, dx, dy):
                             progress = True
                         else:
                             # Move technically valid but failed? Weird. Keep it?
                             # Or assume transient failure and drop? 
                             # If move() returns False, it failed.
                             log(f"Bot {bid} move {dx,dy} failed execution")
                    else:
                        next_pending.append((bid, dx, dy))
                
                if not progress:
                    # No moves succeeded this pass. Deadlock or blocked by external.
                    break
                pending_moves = next_pending
            
            for bid, dx, dy in pending_moves:
                 log(f"Bot {bid} Move {dx,dy} rejected by engine (Dependancy/Block)")