import random
from collections import deque, defaultdict
import heapq
from typing import Tuple, Optional, List, Dict, Set, Any
import math

# Try imports
try:
    from game_constants import Team, TileType, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food, Item
except ImportError:
    pass

# --- Configuration ---
DEBUG_MODE = True
MAX_PLAN_DEPTH = 20
STUCK_THRESHOLD = 5

def log(msg):
    if DEBUG_MODE:
        print(f"[SmartWinBot] {msg}")

# --- Primitives ---

class Action:
    def execute(self, bot_id: int, controller: RobotController) -> bool:
        raise NotImplementedError

    def is_done(self, bot_id: int, controller: RobotController) -> bool:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__

class MoveAction(Action):
    def __init__(self, target: Tuple[int, int], stop_dist=0):
        self.target = target
        self.stop_dist = stop_dist
        self.path = [] 

    def execute(self, bot_id: int, controller: RobotController) -> bool:
        # Check completion first
        if self.is_done(bot_id, controller): return True
        return False # logic handled in Coordinator wrapper

    def is_done(self, bot_id: int, controller: RobotController) -> bool:
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        
        # Manhattan distance
        dist = abs(bx - self.target[0]) + abs(by - self.target[1])
        return dist <= self.stop_dist

class InteractAction(Action):
    def __init__(self, type: str, target: Tuple[int, int], item_to_buy=None):
        self.type = type 
        self.target = target
        self.item_to_buy = item_to_buy

    def execute(self, bot_id: int, controller: RobotController) -> bool:
        tx, ty = self.target
        if self.type == "PICKUP": return controller.pickup(bot_id, tx, ty)
        if self.type == "PLACE": return controller.place(bot_id, tx, ty)
        if self.type == "CHOP": return controller.chop(bot_id, tx, ty)
        if self.type == "BUY": return controller.buy(bot_id, self.item_to_buy, tx, ty)
        if self.type == "SUBMIT": return controller.submit(bot_id, tx, ty)
        if self.type == "ADD_TO_PLATE": return controller.add_food_to_plate(bot_id, tx, ty)
        if self.type == "TAKE_FROM_PAN": return controller.take_from_pan(bot_id, tx, ty)
        if self.type == "TRASH": return controller.trash(bot_id, tx, ty)
        return False

    def is_done(self, bot_id: int, controller: RobotController) -> bool:
        return True
        
    def __repr__(self):
        return f"{self.type}@{self.target}"

# --- Reservation System ---

class ReservationSystem:
    def __init__(self):
        self.reserved_tiles = {} # (x,y) -> bot_id
    
    def reserve(self, x, y, bot_id):
        self.reserved_tiles[(x, y)] = bot_id
        
    def release(self, x, y, bot_id):
        if self.reserved_tiles.get((x, y)) == bot_id:
            del self.reserved_tiles[(x, y)]
            
    def is_reserved(self, x, y, bot_id=None):
        owner = self.reserved_tiles.get((x, y))
        return owner is not None and owner != bot_id

    def clear_bot(self, bot_id):
        to_del = [k for k, v in self.reserved_tiles.items() if v == bot_id]
        for k in to_del: del self.reserved_tiles[k]

# --- Pathfinding ---
class Pathfinding:
    @staticmethod
    def get_path(controller: RobotController, start: Tuple[int, int], target: Tuple[int, int], 
                 stop_dist: int, avoid: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        m = controller.get_map()
        w, h = m.width, m.height
        
        # BFS
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            
            # Check Goal
            dist = abs(curr[0] - target[0]) + abs(curr[1] - target[1])
            if dist <= stop_dist:
                return path
            
            cx, cy = curr
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < w and 0 <= ny < h:
                    if neighbor not in visited:
                        # Must be walkable
                        if m.is_tile_walkable(nx, ny):
                            # Must not be dynamic obstacle (unless it's the target? No, target might be unwalkable)
                            # But here we are pathing THOUGH accessible tiles.
                            if neighbor not in avoid:
                                visited.add(neighbor)
                                queue.append((neighbor, path + [(dx, dy)]))
                        else:
                            # Not walkable. But is it our target?
                            # If target is unwalkable (Counter) and stop_dist > 0, we can't enter it anyway.
                            # We just need to stand NEXT to it.
                            # So we don't add unwalkable tiles to consideration.
                            pass
        return None

# --- Main Logic ---

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.reservations = ReservationSystem()
        self.actions = {} # bot_id -> deque[Action]
        self.stuck_counters = {} # bot_id -> int
        self.last_positions = {} # bot_id -> (x,y)
        self.initialized = False
        
        # Cache
        self.counters = []
        self.cookers = []
        self.sinks = []
        self.submit_locs = []
        self.shops = []
        self.trash_locs = []
        
    def _init(self, controller: RobotController):
        m = controller.get_map()
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

    def find_nearest_empty_counter(self, controller, pos, bot_id):
         m = controller.get_map()
         candidates = []
         for cx, cy in self.counters:
             if not self.reservations.is_reserved(cx, cy, bot_id):
                 tile = m.tiles[cx][cy]
                 if tile.item is None: # Physically empty
                     candidates.append((cx, cy))
         
         if not candidates: return None
         return min(candidates, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))

    def get_closest(self, pos, locs):
        if not locs: return None
        return min(locs, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))

    def plan_bot(self, controller, bot_id):
        # PLANNING CORE
        
        state = controller.get_bot_state(bot_id)
        bx, by = state['x'], state['y']
        holding = state['holding']
        
        plan = deque()
        
        # 1. IF HOLDING SOMETHING
        if holding:
             itype = holding.get('type')
             
             # --> PLATE
             if itype == 'Plate':
                 foods = holding.get('food', [])
                 if foods:
                     # Non-empty plate: Assume done? 
                     target = self.get_closest((bx, by), self.submit_locs)
                     if target:
                         plan.append(MoveAction(target, 1))
                         plan.append(InteractAction("SUBMIT", target))
                 else:
                     # Empty plate: Put on counter to fill
                     target = self.find_nearest_empty_counter(controller, (bx, by), bot_id)
                     if target:
                         self.reservations.reserve(target[0], target[1], bot_id)
                         plan.append(MoveAction(target, 1))
                         plan.append(InteractAction("PLACE", target))
             
             # --> FOOD (Ingredient)
             elif itype == 'Food':
                 name = holding.get('food_name')
                 chopped = holding.get('chopped')
                 cooked_stage = holding.get('cooked_stage')
                 
                 is_choppable = name in ["Onion", "Meat"]
                 is_cookable = name in ["Egg", "Meat"]
                 
                 if is_choppable and not chopped:
                     target = self.find_nearest_empty_counter(controller, (bx, by), bot_id)
                     if target:
                         self.reservations.reserve(target[0], target[1], bot_id)
                         plan.append(MoveAction(target, 1))
                         plan.append(InteractAction("PLACE", target))
                         plan.append(InteractAction("CHOP", target))
                         plan.append(InteractAction("PICKUP", target))
                 
                 elif is_cookable and cooked_stage == 0:
                     best = None
                     m = controller.get_map()
                     for kx, ky in self.cookers:
                         tile = m.tiles[kx][ky]
                         if tile.item and isinstance(tile.item, Pan):
                             if not tile.item.food: # Empty pan
                                 if not self.reservations.is_reserved(kx, ky, bot_id):
                                     best = (kx, ky)
                                     break
                     if best:
                         self.reservations.reserve(best[0], best[1], bot_id)
                         plan.append(MoveAction(best, 1))
                         plan.append(InteractAction("PLACE", best)) 
                     else:
                         target = self.find_nearest_empty_counter(controller, (bx, by), bot_id)
                         if target:
                             self.reservations.reserve(target[0], target[1], bot_id)
                             plan.append(MoveAction(target, 1))
                             plan.append(InteractAction("PLACE", target))
                 else:
                     best_plate = None
                     m = controller.get_map()
                     for cx, cy in self.counters:
                         if not self.reservations.is_reserved(cx, cy, bot_id):
                             tile = m.tiles[cx][cy]
                             if tile.item and isinstance(tile.item, Plate):
                                 best_plate = (cx, cy)
                                 break
                     
                     if best_plate:
                         self.reservations.reserve(best_plate[0], best_plate[1], bot_id)
                         plan.append(MoveAction(best_plate, 1))
                         plan.append(InteractAction("ADD_TO_PLATE", best_plate))
                     else:
                         # No plate? Place on counter
                         target = self.find_nearest_empty_counter(controller, (bx, by), bot_id)
                         if target:
                             self.reservations.reserve(target[0], target[1], bot_id)
                             plan.append(MoveAction(target, 1))
                             plan.append(InteractAction("PLACE", target))

        # 2. IF EMPTY HANDED
        else:
             m = controller.get_map()
             
             # A. Cooked Food
             for kx, ky in self.cookers:
                 if self.reservations.is_reserved(kx, ky, bot_id): continue
                 tile = m.tiles[kx][ky]
                 if tile.item and isinstance(tile.item, Pan) and tile.item.food:
                     f = tile.item.food
                     if isinstance(f, dict) and f.get('cooked_stage') == 1:
                         self.reservations.reserve(kx, ky, bot_id)
                         plan.append(MoveAction((kx, ky), 1))
                         plan.append(InteractAction("TAKE_FROM_PAN", (kx, ky)))
                         return plan

             # B. Processed on Counter
             for cx, cy in self.counters:
                 if self.reservations.is_reserved(cx, cy, bot_id): continue
                 tile = m.tiles[cx][cy]
                 if tile.item and isinstance(tile.item, Food):
                     f = tile.item
                     if f.chopped:
                         self.reservations.reserve(cx, cy, bot_id)
                         plan.append(MoveAction((cx, cy), 1))
                         plan.append(InteractAction("PICKUP", (cx, cy)))
                         return plan
                         
             # C. Full Plate
             for cx, cy in self.counters:
                 if self.reservations.is_reserved(cx, cy, bot_id): continue
                 tile = m.tiles[cx][cy]
                 if tile.item and isinstance(tile.item, Plate) and hasattr(tile.item, 'food') and len(tile.item.food) > 0:
                     self.reservations.reserve(cx, cy, bot_id)
                     plan.append(MoveAction((cx, cy), 1))
                     plan.append(InteractAction("PICKUP", (cx, cy)))
                     return plan

             # D. Fetch Ingredient (Naive: Just find first active order requirement)
             needed = None
             orders = controller.get_orders()
             for o in orders:
                 if o['is_active']:
                     needed = o['required'][0] # Take first
                     break
             
             if needed:
                 # Check plates
                 plates_on_map = 0
                 for cx, cy in self.counters:
                     if m.tiles[cx][cy].item and isinstance(m.tiles[cx][cy].item, Plate):
                         plates_on_map += 1
                 
                 if plates_on_map < 1:
                      shop = self.get_closest((bx, by), self.shops)
                      if shop:
                          plan.append(MoveAction(shop, 1))
                          plan.append(InteractAction("BUY", shop, item_to_buy=ShopCosts.PLATE))
                          return plan

                 # Buy Ingredient
                 item_const = None
                 if needed == "Onion": item_const = FoodType.ONION
                 if needed == "Meat": item_const = FoodType.MEAT
                 if needed == "Egg": item_const = FoodType.EGG
                 if needed == "Noodles": item_const = FoodType.NOODLES
                 if needed == "Sauce": item_const = FoodType.SAUCE
                 
                 if item_const:
                     shop = self.get_closest((bx, by), self.shops)
                     if shop:
                          plan.append(MoveAction(shop, 1))
                          plan.append(InteractAction("BUY", shop, item_to_buy=item_const))
                          return plan

        return plan

    def play_turn(self, controller: RobotController):
        if not self.initialized: self._init(controller)
        
        my_bots = controller.get_team_bot_ids()
        
        for bot_id in my_bots:
            # Init state tracking
            if bot_id not in self.stuck_counters: self.stuck_counters[bot_id] = 0
            if bot_id not in self.last_positions: self.last_positions[bot_id] = (0,0)
            
            # Stuck check
            cur_pos = (controller.get_bot_state(bot_id)['x'], controller.get_bot_state(bot_id)['y'])
            if cur_pos == self.last_positions[bot_id]:
                self.stuck_counters[bot_id] += 1
            else:
                self.stuck_counters[bot_id] = 0
            self.last_positions[bot_id] = cur_pos
            
            if self.stuck_counters[bot_id] > STUCK_THRESHOLD:
                # Reset
                log(f"Bot {bot_id} STUCK. Resetting plan.")
                self.actions[bot_id] = deque()
                self.reservations.clear_bot(bot_id)
                self.stuck_counters[bot_id] = 0

            # Get Plan
            if bot_id not in self.actions: self.actions[bot_id] = deque()
            queue = self.actions[bot_id]
            
            if not queue:
                self.reservations.clear_bot(bot_id)
                new_plan = self.plan_bot(controller, bot_id)
                if new_plan:
                    self.actions[bot_id] = new_plan
                    queue = self.actions[bot_id]
            
            # Execute
            if queue:
                action = queue[0]
                
                # Move Logic
                if isinstance(action, MoveAction):
                    # Check if done
                    if action.is_done(bot_id, controller):
                        queue.popleft()
                        if queue:
                             next_act = queue[0]
                             if isinstance(next_act, InteractAction):
                                 next_act.execute(bot_id, controller)
                                 queue.popleft()
                    else:
                        # Pathfind
                        start = cur_pos
                        target = action.target
                        
                        avoid = set()
                        for b in my_bots:
                            if b != bot_id:
                                bs = controller.get_bot_state(b)
                                avoid.add((bs['x'], bs['y']))
                                
                        # Use updated pathfinder with stop_dist
                        path = Pathfinding.get_path(controller, start, target, action.stop_dist, avoid)
                        
                        if path:
                            dx, dy = path[0]
                            controller.move(bot_id, dx, dy)
                        else:
                            # Random Wiggle
                            valid_moves = []
                            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                                if controller.get_map().is_tile_walkable(cur_pos[0]+dx, cur_pos[1]+dy):
                                    valid_moves.append((dx, dy))
                            if valid_moves:
                                dx, dy = random.choice(valid_moves)
                                controller.move(bot_id, dx, dy)

                elif isinstance(action, InteractAction):
                    action.execute(bot_id, controller)
                    queue.popleft()
