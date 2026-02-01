"""
SoloBot - Simple single-bot strategy
One bot does all the work, the other stays in a corner to avoid interference.
"""

from collections import deque
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Set

try:
    from game_constants import Team, TileType, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

DEBUG = False

def log(msg):
    if DEBUG:
        print(f"[SoloBot] {msg}")

class Phase(Enum):
    IDLE = auto()
    BUY_INGREDIENT = auto()
    CHOP = auto()
    COOK = auto()
    WAIT_COOK = auto()
    FETCH_COOKED = auto()
    BUY_PLATE = auto()
    PICKUP_PLATE = auto()
    PLATE_FOOD = auto()
    DELIVER = auto()


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
        
        # State
        self.worker_id = None
        self.idler_id = None
        self.phase = Phase.IDLE
        self.current_order = None
        self.current_req_idx = 0
        self.plate_loc = None
        self.assembly_loc = None
        
    def _init_map(self, controller):
        m = controller.get_map(controller.get_team())
        
        # Find corners (walkable tiles at edges)
        corners = []
        for x in range(m.width):
            for y in range(m.height):
                if m.is_tile_walkable(x, y):
                    # Check if it's a corner-ish position
                    if (x <= 1 or x >= m.width - 2) and (y <= 1 or y >= m.height - 2):
                        corners.append((x, y))
        
        # Find map features
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
        
        # Pick a corner away from action
        if corners:
            # Prefer bottom-right corner
            self.corner = max(corners, key=lambda p: p[0] + p[1])
        else:
            # Just use any walkable tile far from shop
            for x in range(m.width - 1, -1, -1):
                for y in range(m.height - 1, -1, -1):
                    if m.is_tile_walkable(x, y):
                        self.corner = (x, y)
                        break
                if self.corner:
                    break
        
        # Use first counter as assembly area
        self.assembly_loc = self.counters[0] if self.counters else None
        
        self.initialized = True
        log(f"Init: {len(self.counters)} counters, corner={self.corner}")

    def bfs_path(self, controller, start, target, stop_dist=0):
        """Simple BFS to find path. Returns next move (dx, dy)."""
        if max(abs(start[0] - target[0]), abs(start[1] - target[1])) <= stop_dist:
            return (0, 0)
        
        m = controller.get_map(controller.get_team())
        queue = deque([(start[0], start[1], None)])  # x, y, first_move
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
        
        return (0, 0)  # No path found

    def get_order_requirements(self, order):
        """Parse order requirements into a list of (food_name, needs_chop, needs_cook)."""
        reqs = []
        for item_name in order['required']:
            if item_name in ["MEAT", "ONIONS"]:
                reqs.append((item_name, True, True))  # Needs chop + cook
            elif item_name == "EGG":
                reqs.append((item_name, False, True))  # Just cook
            else:  # NOODLES, SAUCE
                reqs.append((item_name, False, False))  # Just plate directly
        return reqs

    def find_free_counter(self, controller):
        for c in self.counters:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and tile.item is None:
                return c
        return self.assembly_loc

    def find_free_cooker(self, controller):
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item'):
                if tile.item is None:
                    return c
                if isinstance(tile.item, Pan) and tile.item.food is None:
                    return c
        return None

    def find_ready_pan(self, controller):
        """Find a pan with fully cooked food."""
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                if tile.item.food and tile.item.food.cooked_stage == 1:
                    return c
        return None

    def find_cooking_pan(self, controller, food_name=None):
        """Find a pan with food that's still cooking."""
        for c in self.cookers:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                if tile.item.food and tile.item.food.cooked_stage == 0:
                    if food_name is None or tile.item.food.food_name == food_name:
                        return c
        return None

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map(controller)
        
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        
        # Assign roles on first turn
        if self.worker_id is None:
            self.worker_id = bots[0]
            self.idler_id = bots[1] if len(bots) > 1 else None
        
        # Move idler to corner
        if self.idler_id and self.corner:
            idler_state = controller.get_bot_state(self.idler_id)
            idler_pos = (idler_state['x'], idler_state['y'])
            if idler_pos != self.corner:
                move = self.bfs_path(controller, idler_pos, self.corner, stop_dist=0)
                if move != (0, 0) and controller.can_move(self.idler_id, move[0], move[1]):
                    controller.move(self.idler_id, move[0], move[1])
        
        # Worker logic
        worker_state = controller.get_bot_state(self.worker_id)
        worker_pos = (worker_state['x'], worker_state['y'])
        holding = worker_state.get('holding')
        
        # Get current order
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        valid_orders = [o for o in orders if o['expires_turn'] - current_turn > 30]
        
        if not valid_orders:
            log("No valid orders")
            return
        
        # Pick best order (simplest first)
        if not self.current_order or self.current_order not in [o['order_id'] for o in valid_orders]:
            # Pick order with fewest requirements
            best = min(valid_orders, key=lambda o: len(o['required']))
            self.current_order = best['order_id']
            self.current_req_idx = 0
            self.phase = Phase.IDLE
            log(f"New order: {best['required']}")
        
        order = next((o for o in valid_orders if o['order_id'] == self.current_order), None)
        if not order:
            self.current_order = None
            return
        
        reqs = self.get_order_requirements(order)
        
        # State machine
        log(f"Phase: {self.phase}, req_idx: {self.current_req_idx}, holding: {holding}")
        
        # Check if we need a plate
        need_plate = self.plate_loc is None
        for c in self.counters:
            tile = controller.get_tile(team, c[0], c[1])
            if tile and isinstance(tile.item, Plate):
                self.plate_loc = c
                need_plate = False
                break
        
        if holding and holding.get('type') == 'Plate':
            # Check if plate is complete
            plate_foods = [f['food_name'] for f in holding.get('food', [])]
            all_present = all(r[0] in plate_foods for r in reqs)
            
            if all_present:
                # Deliver!
                target = self.submit_locs[0] if self.submit_locs else self.assembly_loc
                dist = max(abs(worker_pos[0] - target[0]), abs(worker_pos[1] - target[1]))
                if dist <= 1:
                    if controller.submit(self.worker_id, target[0], target[1]):
                        log("SUBMITTED!")
                        self.current_order = None
                        self.plate_loc = None
                else:
                    move = self.bfs_path(controller, worker_pos, target, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
                return
            else:
                # Need to add more food to plate - put it down first
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
        
        # Check what's on the plate already
        plate_contents = []
        if self.plate_loc:
            tile = controller.get_tile(team, self.plate_loc[0], self.plate_loc[1])
            if tile and isinstance(tile.item, Plate):
                plate_contents = [f.food_name for f in tile.item.food]
        
        # Find what we still need
        missing = [r for r in reqs if r[0] not in plate_contents]
        
        if not missing:
            # All ingredients on plate! Pick it up and deliver
            if self.plate_loc:
                dist = max(abs(worker_pos[0] - self.plate_loc[0]), abs(worker_pos[1] - self.plate_loc[1]))
                if dist <= 1:
                    controller.pickup(self.worker_id, self.plate_loc[0], self.plate_loc[1])
                else:
                    move = self.bfs_path(controller, worker_pos, self.plate_loc, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
            return
        
        # Check if we have a plate; if not, buy one
        if need_plate and not holding:
            if self.shop_loc:
                dist = max(abs(worker_pos[0] - self.shop_loc[0]), abs(worker_pos[1] - self.shop_loc[1]))
                if dist <= 1:
                    controller.buy(self.worker_id, ShopCosts.PLATE, self.shop_loc[0], self.shop_loc[1])
                else:
                    move = self.bfs_path(controller, worker_pos, self.shop_loc, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
            return
        
        # Process next missing ingredient
        next_req = missing[0]
        food_name, needs_chop, needs_cook = next_req
        
        # If holding the right food
        if holding and holding.get('type') == 'Food' and holding.get('food_name') == food_name:
            food = holding
            
            if needs_chop and not food.get('chopped'):
                # Need to chop - place item first, will chop next turn
                counter = self.find_free_counter(controller)
                if counter:
                    dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                    if dist <= 1:
                        controller.place(self.worker_id, counter[0], counter[1])
                        # Will chop next turn when we see it on counter
                    else:
                        move = self.bfs_path(controller, worker_pos, counter, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                return
            
            if needs_cook and food.get('cooked_stage', 0) < 1:
                # Need to cook
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
            
            # Ready to plate!
            if self.plate_loc:
                dist = max(abs(worker_pos[0] - self.plate_loc[0]), abs(worker_pos[1] - self.plate_loc[1]))
                if dist <= 1:
                    controller.add_food_to_plate(self.worker_id, self.plate_loc[0], self.plate_loc[1])
                else:
                    move = self.bfs_path(controller, worker_pos, self.plate_loc, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
            return
        
        # If holding wrong food or plate, put it down
        if holding:
            counter = self.find_free_counter(controller)
            if counter:
                dist = max(abs(worker_pos[0] - counter[0]), abs(worker_pos[1] - counter[1]))
                if dist <= 1:
                    controller.place(self.worker_id, counter[0], counter[1])
                    if holding.get('type') == 'Plate':
                        self.plate_loc = counter
                else:
                    move = self.bfs_path(controller, worker_pos, counter, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
            return
        
        # Not holding anything - need to get the ingredient
        # First check if cooked version is in a pan
        if needs_cook:
            ready_pan = self.find_ready_pan(controller)
            if ready_pan:
                tile = controller.get_tile(team, ready_pan[0], ready_pan[1])
                if tile.item.food.food_name == food_name:
                    dist = max(abs(worker_pos[0] - ready_pan[0]), abs(worker_pos[1] - ready_pan[1]))
                    if dist <= 1:
                        controller.take_from_pan(self.worker_id, ready_pan[0], ready_pan[1])
                    else:
                        move = self.bfs_path(controller, worker_pos, ready_pan, 1)
                        if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                            controller.move(self.worker_id, move[0], move[1])
                    return
            
            # Check if food is currently cooking - just wait
            cooking_pan = self.find_cooking_pan(controller, food_name)
            if cooking_pan:
                log(f"Waiting for {food_name} to cook...")
                return
        
        # Check counters for the ingredient in any state
        for c in self.counters:
            tile = controller.get_tile(team, c[0], c[1])
            if tile and isinstance(tile.item, Food) and tile.item.food_name == food_name:
                dist = max(abs(worker_pos[0] - c[0]), abs(worker_pos[1] - c[1]))
                if dist <= 1:
                    # Check if needs chopping
                    if needs_chop and not tile.item.chopped:
                        controller.chop(self.worker_id, c[0], c[1])
                    else:
                        controller.pickup(self.worker_id, c[0], c[1])
                else:
                    move = self.bfs_path(controller, worker_pos, c, 1)
                    if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                        controller.move(self.worker_id, move[0], move[1])
                return
        
        # Need to buy the ingredient
        if self.shop_loc:
            dist = max(abs(worker_pos[0] - self.shop_loc[0]), abs(worker_pos[1] - self.shop_loc[1]))
            if dist <= 1:
                if hasattr(FoodType, food_name):
                    controller.buy(self.worker_id, getattr(FoodType, food_name), self.shop_loc[0], self.shop_loc[1])
            else:
                move = self.bfs_path(controller, worker_pos, self.shop_loc, 1)
                if move != (0, 0) and controller.can_move(self.worker_id, move[0], move[1]):
                    controller.move(self.worker_id, move[0], move[1])
