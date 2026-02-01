"""
OptimalBot - Fully deterministic order scheduling
Uses dynamic programming to find optimal order sequence based on:
- Order availability windows (start_turn, expires_turn)
- Estimated completion time per order
- Reward/penalty values

With get_orders() now returning ALL orders (past/present/future), we can
plan the entire game optimally from the start.
"""

from collections import deque
from typing import List, Tuple, Dict, Optional, Set
from itertools import permutations

try:
    from game_constants import Team, TileType, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

DEBUG = False

def log(msg):
    if DEBUG:
        print(f"[OptimalBot] {msg}")


# Estimated turns to complete each ingredient type
INGREDIENT_TIMES = {
    "MEAT": 25,      # Buy + chop + place + cook (15 turns) + take = ~25
    "ONIONS": 25,    # Buy + chop + place + cook (15 turns) + take = ~25
    "EGG": 20,       # Buy + place + cook (15 turns) + take = ~20
    "NOODLES": 5,    # Just buy
    "SAUCE": 5,      # Just buy
}

# Base time for order overhead (buy plate, deliver)
ORDER_OVERHEAD = 15


def estimate_order_time(order):
    """Estimate turns needed to complete an order."""
    total = ORDER_OVERHEAD
    for item in order['required']:
        total += INGREDIENT_TIMES.get(item, 10)
    return total


def can_complete_order(order, current_turn, completion_time):
    """Check if order can be completed before expiration."""
    start = order.get('start_turn', 0)
    expires = order['expires_turn']
    
    # Can't start before order is available
    if current_turn < start:
        current_turn = start
    
    finish_turn = current_turn + completion_time
    return finish_turn <= expires


def calculate_order_value(order, completion_turn):
    """Calculate the value of completing an order at a given turn."""
    expires = order['expires_turn']
    if completion_turn > expires:
        return -order.get('penalty', 0)  # Penalty for missing
    return order['reward']


def find_optimal_schedule(orders, current_turn, max_turns):
    """
    Find the optimal order sequence that maximizes total earnings.
    Uses dynamic programming with memoization.
    
    Returns: list of order_ids in optimal execution order
    """
    if not orders:
        return []
    
    # Filter to orders that are possible to complete
    feasible = []
    for o in orders:
        est_time = estimate_order_time(o)
        start = max(current_turn, o.get('start_turn', 0))
        if start + est_time <= o['expires_turn'] and start + est_time <= max_turns:
            feasible.append(o)
    
    if not feasible:
        return []
    
    # For small number of orders, try all permutations
    # For larger sets, use greedy heuristic
    if len(feasible) <= 8:
        return _exhaustive_search(feasible, current_turn, max_turns)
    else:
        return _greedy_schedule(feasible, current_turn, max_turns)


def _exhaustive_search(orders, current_turn, max_turns):
    """Try all permutations and find the best one."""
    best_value = 0
    best_schedule = []
    
    for perm in permutations(range(len(orders))):
        turn = current_turn
        value = 0
        schedule = []
        
        for idx in perm:
            order = orders[idx]
            est_time = estimate_order_time(order)
            start = max(turn, order.get('start_turn', 0))
            finish = start + est_time
            
            if finish <= order['expires_turn'] and finish <= max_turns:
                value += order['reward']
                schedule.append(order['order_id'])
                turn = finish
        
        if value > best_value:
            best_value = value
            best_schedule = schedule
    
    log(f"Optimal schedule found: {best_schedule} with value ${best_value}")
    return best_schedule


def _greedy_schedule(orders, current_turn, max_turns):
    """Greedy approach: prioritize by reward-to-time ratio with deadline awareness."""
    remaining = list(orders)
    schedule = []
    turn = current_turn
    
    while remaining:
        # Score each order by (reward / time) * deadline_factor
        best_score = -1
        best_order = None
        
        for o in remaining:
            est_time = estimate_order_time(o)
            start = max(turn, o.get('start_turn', 0))
            finish = start + est_time
            
            if finish > o['expires_turn'] or finish > max_turns:
                continue  # Can't complete in time
            
            # Deadline urgency factor (higher for orders expiring soon)
            slack = o['expires_turn'] - finish
            urgency = 1.0 / (1 + slack * 0.1)
            
            score = (o['reward'] / est_time) * (1 + urgency)
            
            if score > best_score:
                best_score = score
                best_order = o
        
        if best_order is None:
            break
        
        schedule.append(best_order['order_id'])
        remaining.remove(best_order)
        turn += estimate_order_time(best_order)
    
    log(f"Greedy schedule: {schedule}")
    return schedule


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map features
        self.counters = []
        self.cookers = []
        self.submit_locs = []
        self.shop_loc = None
        self.corner = None
        
        # Bot state
        self.worker_id = None
        self.idler_id = None
        
        # Planning state
        self.all_orders = None
        self.schedule = []
        self.current_order_idx = 0
        self.current_order = None
        self.current_req_idx = 0
        
        # Execution state
        self.plate_loc = None
        
    def _init_map(self, controller):
        m = controller.get_map(controller.get_team())
        corners = []
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COUNTER":
                    self.counters.append((x, y))
                elif tile.tile_name == "COOKER":
                    self.cookers.append((x, y))
                elif tile.tile_name == "SUBMIT":
                    self.submit_locs.append((x, y))
                elif tile.tile_name == "SHOP":
                    self.shop_loc = (x, y)
                
                # Track corners for idler
                if m.is_tile_walkable(x, y):
                    neighbors = sum(1 for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] 
                                   if 0 <= x+dx < m.width and 0 <= y+dy < m.height 
                                   and not m.is_tile_walkable(x+dx, y+dy))
                    if neighbors >= 2:
                        corners.append((x, y))
        
        # Pick corner farthest from shop for idler
        if corners and self.shop_loc:
            self.corner = max(corners, key=lambda c: abs(c[0]-self.shop_loc[0]) + abs(c[1]-self.shop_loc[1]))
        elif corners:
            self.corner = corners[0]
        
        self.initialized = True
        log(f"Init: {len(self.counters)} counters, {len(self.cookers)} cookers, shop={self.shop_loc}")
        log(f"Counters: {self.counters}")

    def _init_schedule(self, controller):
        """Initialize the optimal schedule at game start."""
        team = controller.get_team()
        current_turn = controller.get_turn()
        
        # Get ALL orders (past/present/future) - new API feature!
        all_orders = controller.get_orders(team)
        self.all_orders = all_orders
        
        # Find optimal schedule
        max_turns = 1000  # Adjust based on game settings
        self.schedule = find_optimal_schedule(all_orders, current_turn, max_turns)
        
        log(f"Planned schedule for {len(self.schedule)} orders: {self.schedule}")
        
        if self.schedule:
            self._load_next_order(controller)
    
    def _load_next_order(self, controller):
        """Load the next order from the schedule."""
        if self.current_order_idx >= len(self.schedule):
            self.current_order = None
            return
        
        target_id = self.schedule[self.current_order_idx]
        team = controller.get_team()
        
        # Find the order by ID
        for o in self.all_orders:
            if o['order_id'] == target_id:
                self.current_order = o
                self.current_req_idx = 0
                self.plate_loc = None
                log(f"Loading order {target_id}: {o['required']}")
                return
        
        # Order not found, skip
        self.current_order_idx += 1
        self._load_next_order(controller)
    
    def bfs_path(self, controller, start, target, stop_dist=0):
        """BFS pathfinding with Chebyshev distance."""
        if max(abs(start[0] - target[0]), abs(start[1] - target[1])) <= stop_dist:
            return (0, 0)
        
        m = controller.get_map(controller.get_team())
        queue = deque([(start[0], start[1], None)])
        visited = {start}
        
        MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        while queue:
            x, y, first_move = queue.popleft()
            
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not m.is_tile_walkable(nx, ny):
                    continue
                
                visited.add((nx, ny))
                fm = first_move if first_move else (dx, dy)
                
                if max(abs(nx - target[0]), abs(ny - target[1])) <= stop_dist:
                    return fm
                
                queue.append((nx, ny, fm))
        
        return (0, 0)

    def find_free_counter(self, controller):
        for c in self.counters:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and tile.item is None:
                return c
            else:
                log(f"Counter {c} has item: {type(tile.item).__name__ if tile and tile.item else 'None'}")
        return None

    def find_free_cooker(self, controller):
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item'):
                if tile.item is None:
                    return c
                if isinstance(tile.item, Pan) and tile.item.food is None:
                    return c
        return None

    def find_ready_pan(self, controller, food_name=None):
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                if tile.item.food and tile.item.food.cooked_stage == 1:
                    if food_name is None or tile.item.food.food_name == food_name:
                        return c
        return None

    def find_cooking_pan(self, controller, food_name=None):
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                if tile.item.food and tile.item.food.cooked_stage == 0:
                    if food_name is None or tile.item.food.food_name == food_name:
                        return c
        return None

    def find_food_on_counter(self, controller, food_name):
        for c in self.counters:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and isinstance(tile.item, Food):
                if tile.item.food_name == food_name:
                    return c
        return None

    def get_requirement(self, item_name):
        """Get (needs_chop, needs_cook) for an ingredient."""
        if item_name in ["MEAT", "ONIONS"]:
            return (True, True)  # chop + cook
        elif item_name == "EGG":
            return (False, True)  # just cook
        else:
            return (False, False)  # direct plate

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map(controller)
        
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        
        # Assign worker and idler
        if self.worker_id is None:
            self.worker_id = bots[0]
            self.idler_id = bots[1] if len(bots) > 1 else None
        
        # Initialize schedule on first turn
        if self.all_orders is None:
            self._init_schedule(controller)
        
        # Move idler to corner
        if self.idler_id and self.corner:
            idler_state = controller.get_bot_state(self.idler_id)
            idler_pos = (idler_state['x'], idler_state['y'])
            if idler_pos != self.corner:
                move = self.bfs_path(controller, idler_pos, self.corner)
                if move != (0, 0) and controller.can_move(self.idler_id, move[0], move[1]):
                    controller.move(self.idler_id, move[0], move[1])
        
        # Execute current order
        if not self.current_order:
            return
        
        # Check if order is still valid (not expired)
        current_turn = controller.get_turn()
        if current_turn > self.current_order['expires_turn']:
            log(f"Order {self.current_order['order_id']} expired, moving to next")
            self.current_order_idx += 1
            self._load_next_order(controller)
            return
        
        # Wait for order to start
        if current_turn < self.current_order.get('start_turn', 0):
            return
        
        self._execute_order(controller)
    
    def _execute_order(self, controller):
        """
        Execute the current order with ingredients-first workflow.
        Process all ingredients (chop/cook) BEFORE buying plate to keep counter free.
        """
        team = controller.get_team()
        worker_state = controller.get_bot_state(self.worker_id)
        worker_pos = (worker_state['x'], worker_state['y'])
        holding = worker_state.get('holding')
        
        required = self.current_order['required']
        
        # Track plate location
        if not self.plate_loc:
            for c in self.counters:
                tile = controller.get_tile(team, c[0], c[1])
                if tile and isinstance(tile.item, Plate):
                    self.plate_loc = c
                    break
        
        # Get plate contents
        plate_contents = []
        if holding and holding.get('type') == 'Plate':
            plate_contents = [f['food_name'] for f in holding.get('food', [])]
        elif self.plate_loc:
            tile = controller.get_tile(team, self.plate_loc[0], self.plate_loc[1])
            if tile and isinstance(tile.item, Plate):
                plate_contents = [f.food_name for f in tile.item.food]
        
        # Find missing ingredients
        missing = [r for r in required if r not in plate_contents]
        
        # ==== DELIVERY PHASE ====
        # Holding completed plate - deliver it
        if holding and holding.get('type') == 'Plate':
            if not missing:
                target = self.submit_locs[0] if self.submit_locs else None
                if target:
                    dist = max(abs(worker_pos[0] - target[0]), abs(worker_pos[1] - target[1]))
                    if dist <= 1:
                        if controller.submit(self.worker_id, target[0], target[1]):
                            log(f"SUBMITTED order {self.current_order['order_id']}")
                            self.current_order_idx += 1
                            self._load_next_order(controller)
                    else:
                        move = self.bfs_path(controller, worker_pos, target, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
            else:
                # Still missing ingredients - put plate down
                counter = self.find_free_counter(controller)
                if counter:
                    dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                    if dist <= 1:
                        if controller.place(self.worker_id, counter[0], counter[1]):
                            self.plate_loc = counter
                    else:
                        move = self.bfs_path(controller, worker_pos, counter, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
        
        # Plate is down and all ingredients are on it - pick up and deliver
        if not missing and self.plate_loc:
            dist = max(abs(worker_pos[0] - self.plate_loc[0]), abs(worker_pos[1] - self.plate_loc[1]))
            if dist <= 1:
                controller.pickup(self.worker_id, self.plate_loc[0], self.plate_loc[1])
            else:
                move = self.bfs_path(controller, worker_pos, self.plate_loc, 1)
                if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                    controller.move(self.worker_id, move[0], move[1])
            return
        
        # ==== PREP PHASE: Check what ingredients are ready ====
        # Check for cooked ingredients waiting in pans or on counters
        ready_ingredients = []  # List of (ingredient_name, location_type, location)
        
        for item in missing:
            needs_chop, needs_cook = self.get_requirement(item)
            
            if needs_cook:
                # Check for cooked version in pan
                ready_pan = self.find_ready_pan(controller, item)
                if ready_pan:
                    ready_ingredients.append((item, 'pan', ready_pan))
                    continue
                
                # Check for cooking version in pan
                cooking_pan = self.find_cooking_pan(controller, item)
                if cooking_pan:
                    ready_ingredients.append((item, 'cooking', cooking_pan))
                    continue
            
            # Check for ready (chopped if needed) version on counter
            counter_loc = self.find_food_on_counter(controller, item)
            if counter_loc:
                tile = controller.get_tile(team, counter_loc[0], counter_loc[1])
                food = tile.item
                is_chopped = food.chopped if hasattr(food, 'chopped') else False
                
                if not needs_chop or is_chopped:
                    ready_ingredients.append((item, 'counter_ready', counter_loc))
                else:
                    ready_ingredients.append((item, 'counter_need_chop', counter_loc))
        
        # Count ingredients that NEED processing vs ready
        needs_processing = [i for i in missing if i not in [r[0] for r in ready_ingredients]]
        already_cooking = [r for r in ready_ingredients if r[1] == 'cooking']
        fully_ready = [r for r in ready_ingredients if r[1] in ['pan', 'counter_ready']]
        need_chop = [r for r in ready_ingredients if r[1] == 'counter_need_chop']
        
        # ==== PLATING PHASE: If we have plate and cooked food ready, add to plate ====
        if self.plate_loc and holding and holding.get('type') == 'Food':
            food = holding
            food_name = food.get('food_name')
            needs_chop, needs_cook = self.get_requirement(food_name)
            is_chopped = food.get('chopped', False)
            is_cooked = food.get('cooked_stage', 0) >= 1
            
            # Check if food is ready to plate
            ready_to_plate = (not needs_chop or is_chopped) and (not needs_cook or is_cooked)
            
            if ready_to_plate and food_name in missing:
                dist = max(abs(worker_pos[0] - self.plate_loc[0]), abs(worker_pos[1] - self.plate_loc[1]))
                if dist <= 1:
                    controller.add_food_to_plate(self.worker_id, self.plate_loc[0], self.plate_loc[1])
                else:
                    move = self.bfs_path(controller, worker_pos, self.plate_loc, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
                return
        
        # ==== INGREDIENT ACQUISITION PHASE ====
        # If holding food, process it appropriately  
        if holding and holding.get('type') == 'Food':
            food = holding
            food_name = food.get('food_name')
            needs_chop, needs_cook = self.get_requirement(food_name)
            is_chopped = food.get('chopped', False)
            is_cooked = food.get('cooked_stage', 0) >= 1
            
            # Is this food for this order?
            if food_name not in required:
                # Wrong food - try to drop it
                counter = self.find_free_counter(controller)
                if counter:
                    dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                    if dist <= 1:
                        controller.place(self.worker_id, counter[0], counter[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, counter, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
            
            # Food needs chopping
            if needs_chop and not is_chopped:
                counter = self.find_free_counter(controller)
                if counter:
                    dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                    if dist <= 1:
                        if controller.place(self.worker_id, counter[0], counter[1]):
                            controller.chop(self.worker_id, counter[0], counter[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, counter, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                else:
                    log(f"No free counter for chopping {food_name}")
                return
            
            # Food needs cooking
            if needs_cook and not is_cooked:
                cooker = self.find_free_cooker(controller)
                if cooker:
                    dist = max(abs(worker_pos[0] - cooker[0]), abs(worker_pos[1] - cooker[1]))
                    if dist <= 1:
                        controller.start_cook(self.worker_id, cooker[0], cooker[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, cooker, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
            
            # Food is ready but no plate yet - means we should get the plate now
            # (food is processed, safe to place plate)
            if not self.plate_loc:
                # Put food on counter temporarily, go get plate
                counter = self.find_free_counter(controller)
                if counter:
                    dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                    if dist <= 1:
                        controller.place(self.worker_id, counter[0], counter[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, counter, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
        
        # ==== NOT HOLDING ANYTHING ====
        # Priority: 1) Get cooked food from pan, 2) Chop food on counter, 3) Pick up ready food
        #           4) Wait for cooking, 5) Buy next ingredient, 6) Buy plate when all ready
        
        # Check if we should wait for cooking
        if already_cooking:
            # Something is cooking, wait for it
            return
        
        # Get cooked food from pan
        for item in missing:
            needs_chop, needs_cook = self.get_requirement(item)
            if needs_cook:
                ready_pan = self.find_ready_pan(controller, item)
                if ready_pan:
                    dist = max(abs(worker_pos[0] - ready_pan[0]), abs(worker_pos[1] - ready_pan[1]))
                    if dist <= 1:
                        controller.take_from_pan(self.worker_id, ready_pan[0], ready_pan[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, ready_pan, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                    return
        
        # Chop food on counter
        if need_chop:
            item, _, loc = need_chop[0]
            dist = max(abs(worker_pos[0] - loc[0]), abs(worker_pos[1] - loc[1]))
            if dist <= 1:
                controller.chop(self.worker_id, loc[0], loc[1])
            else:
                move = self.bfs_path(controller, worker_pos, loc, 1)
                if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                    controller.move(self.worker_id, move[0], move[1])
            return
        
        # Pick up ready food from counter (only if we have plate to put it on!)
        if fully_ready and self.plate_loc:
            for item, ltype, loc in fully_ready:
                if ltype == 'counter_ready':
                    dist = max(abs(worker_pos[0] - loc[0]), abs(worker_pos[1] - loc[1]))
                    if dist <= 1:
                        controller.pickup(self.worker_id, loc[0], loc[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, loc, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                    return
        
        # Buy next ingredient that needs processing
        # PRIORITY: Items needing chopping first (they need the counter), then cooking-only, then simple items
        if needs_processing and self.shop_loc:
            # Sort: items needing chop first
            def ingredient_priority(item):
                needs_chop, needs_cook = self.get_requirement(item)
                if needs_chop:
                    return 0  # Highest priority - needs counter for chopping
                elif needs_cook:
                    return 1  # Needs cooker but not counter
                else:
                    return 2  # Simple item, no counter needed
            
            sorted_items = sorted(needs_processing, key=ingredient_priority)
            next_item = sorted_items[0]
            
            dist = max(abs(worker_pos[0] - self.shop_loc[0]), abs(worker_pos[1] - self.shop_loc[1]))
            if dist <= 1:
                if hasattr(FoodType, next_item):
                    controller.buy(self.worker_id, getattr(FoodType, next_item), self.shop_loc[0], self.shop_loc[1])
            else:
                move = self.bfs_path(controller, worker_pos, self.shop_loc, 1)
                if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                    controller.move(self.worker_id, move[0], move[1])
            return
        
        # All processing done OR no processing needed - get the plate!
        if not self.plate_loc and self.shop_loc:
            dist = max(abs(worker_pos[0] - self.shop_loc[0]), abs(worker_pos[1] - self.shop_loc[1]))
            if dist <= 1:
                controller.buy(self.worker_id, ShopCosts.PLATE, self.shop_loc[0], self.shop_loc[1])
            else:
                move = self.bfs_path(controller, worker_pos, self.shop_loc, 1)
                if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                    controller.move(self.worker_id, move[0], move[1])
            return
        
        # If holding plate but no plate_loc set, place it
        if holding and holding.get('type') == 'Plate' and not self.plate_loc:
            counter = self.find_free_counter(controller)
            if counter:
                dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                if dist <= 1:
                    if controller.place(self.worker_id, counter[0], counter[1]):
                        self.plate_loc = counter
                        log(f"Placed plate at {counter}")
                else:
                    move = self.bfs_path(controller, worker_pos, counter, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
            return
