"""
TeamBot - Master-Slave coordination
Master does main work (cooking, plating, delivery)
Slave helps by fetching ingredients and placing on transfer counters
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
        print(f"[TeamBot] {msg}")


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map features
        self.counters = []
        self.cookers = []
        self.submit_locs = []
        self.shops = []  # Store all shops
        
        # State
        self.master_id = None
        self.slave_id = None
        self.plate_loc = None
        self.current_order = None
        
        # Slave task queue
        self.slave_task = None  # (action, food_name, target_loc)
        
    def _init_map(self, controller):
        m = controller.get_map(controller.get_team())
        
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
                    self.shops.append((x, y))
        
        self.initialized = True
        log(f"Init: {len(self.counters)} counters, {len(self.cookers)} cookers")

    def bfs_path(self, controller, start, target, stop_dist=0):
        """Simple BFS to find path. Returns next move (dx, dy)."""
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

    def can_reach(self, controller, start, target):
        """Check if bot at start can interact with target (Chebyshev distance 1)."""
        m = controller.get_map(controller.get_team())
        queue = deque([start])
        visited = {start}
        
        # Use 8-directional BFS for Chebyshev distance
        MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        while queue:
            x, y = queue.popleft()
            
            # Check if current position is within Chebyshev distance 1 of target
            if max(abs(x - target[0]), abs(y - target[1])) <= 1:
                return True
            
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not m.is_tile_walkable(nx, ny):
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        
        return False

    def find_reachable_shop(self, controller, pos):
        """Find a shop that this position can reach."""
        for shop in self.shops:
            if self.can_reach(controller, pos, shop):
                return shop
        return None

    def get_order_requirements(self, order):
        """Parse order requirements."""
        reqs = []
        for item_name in order['required']:
            if item_name in ["MEAT", "ONIONS"]:
                reqs.append((item_name, True, True))  # chop + cook
            elif item_name == "EGG":
                reqs.append((item_name, False, True))  # just cook
            else:
                reqs.append((item_name, False, False))  # direct plate
        return reqs

    def find_free_counter(self, controller, near_pos=None):
        valid = []
        for c in self.counters:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and tile.item is None:
                valid.append(c)
        if not valid:
            return None
        if near_pos:
            return min(valid, key=lambda p: abs(p[0]-near_pos[0]) + abs(p[1]-near_pos[1]))
        return valid[0]

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

    def find_food_on_counter(self, controller, food_name, states=None):
        """Find food on a counter. states is list of (chopped, cooked_stage) tuples."""
        for c in self.counters:
            tile = controller.get_tile(controller.get_team(), c[0], c[1])
            if tile and isinstance(tile.item, Food):
                if tile.item.food_name == food_name:
                    if states is None:
                        return c
                    for chopped, cooked in states:
                        if tile.item.chopped == chopped and tile.item.cooked_stage == cooked:
                            return c
        return None

    def move_or_act(self, controller, bot_id, target, action_fn, stop_dist=1):
        """Move towards target or perform action if in range."""
        state = controller.get_bot_state(bot_id)
        pos = (state['x'], state['y'])
        dist = max(abs(pos[0] - target[0]), abs(pos[1] - target[1]))
        
        if dist <= stop_dist:
            return action_fn()
        else:
            move = self.bfs_path(controller, pos, target, stop_dist)
            if move != (0, 0) and controller.can_move(bot_id, move[0], move[1]):
                controller.move(bot_id, move[0], move[1])
            return False

    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init_map(controller)
        
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        
        # Assign roles - smart assignment based on shop accessibility
        if self.master_id is None:
            # Find which bot can reach a shop
            bot0_state = controller.get_bot_state(bots[0])
            bot0_pos = (bot0_state['x'], bot0_state['y'])
            bot0_shop = self.find_reachable_shop(controller, bot0_pos)
            
            if len(bots) > 1:
                bot1_state = controller.get_bot_state(bots[1])
                bot1_pos = (bot1_state['x'], bot1_state['y'])
                bot1_shop = self.find_reachable_shop(controller, bot1_pos)
                
                # Prefer the bot that can reach a shop as master
                if bot0_shop and not bot1_shop:
                    self.master_id = bots[0]
                    self.slave_id = bots[1]
                elif bot1_shop and not bot0_shop:
                    self.master_id = bots[1]
                    self.slave_id = bots[0]
                else:
                    # Both can or can't reach shop - default assignment
                    self.master_id = bots[0]
                    self.slave_id = bots[1]
            else:
                self.master_id = bots[0]
                self.slave_id = None
        
        # Get orders
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        valid_orders = [o for o in orders if o['expires_turn'] - current_turn > 40]
        
        if not valid_orders:
            return
        
        # Pick order
        if not self.current_order or self.current_order not in [o['order_id'] for o in valid_orders]:
            best = min(valid_orders, key=lambda o: len(o['required']))
            self.current_order = best['order_id']
            log(f"New order: {best['required']}")
        
        order = next((o for o in valid_orders if o['order_id'] == self.current_order), None)
        if not order:
            self.current_order = None
            return
        
        reqs = self.get_order_requirements(order)
        
        # Find/track plate
        self.plate_loc = None
        for c in self.counters:
            tile = controller.get_tile(team, c[0], c[1])
            if tile and isinstance(tile.item, Plate):
                self.plate_loc = c
                break
        
        # Get plate contents
        plate_contents = []
        if self.plate_loc:
            tile = controller.get_tile(team, self.plate_loc[0], self.plate_loc[1])
            if tile and isinstance(tile.item, Plate):
                plate_contents = [f.food_name for f in tile.item.food]
        
        # Check what's held by each bot
        master_state = controller.get_bot_state(self.master_id)
        master_pos = (master_state['x'], master_state['y'])
        master_holding = master_state.get('holding')
        
        slave_state = controller.get_bot_state(self.slave_id) if self.slave_id else None
        slave_pos = (slave_state['x'], slave_state['y']) if slave_state else None
        slave_holding = slave_state.get('holding') if slave_state else None
        
        # Also check if master is holding plate
        if master_holding and master_holding.get('type') == 'Plate':
            plate_contents = [f['food_name'] for f in master_holding.get('food', [])]
        
        # Find missing ingredients
        missing = [r for r in reqs if r[0] not in plate_contents]
        
        log(f"Missing: {[m[0] for m in missing]}, plate_loc: {self.plate_loc}")
        
        # ========== MASTER LOGIC ==========
        self.run_master(controller, team, master_pos, master_holding, missing, reqs, plate_contents)
        
        # ========== SLAVE LOGIC ==========
        if self.slave_id:
            self.run_slave(controller, team, slave_pos, slave_holding, missing, master_pos)

    def run_master(self, controller, team, master_pos, master_holding, missing, reqs, plate_contents):
        """Master: cooking, plating, delivery"""
        
        # If holding plate with all ingredients -> deliver
        if master_holding and master_holding.get('type') == 'Plate':
            plate_foods = [f['food_name'] for f in master_holding.get('food', [])]
            if all(r[0] in plate_foods for r in reqs):
                target = self.submit_locs[0] if self.submit_locs else self.counters[0]
                def act():
                    if controller.submit(self.master_id, target[0], target[1]):
                        log("SUBMITTED!")
                        self.current_order = None
                        self.plate_loc = None
                        return True
                    return False
                self.move_or_act(controller, self.master_id, target, act)
                return
            else:
                # Put plate down to add more ingredients
                counter = self.find_free_counter(controller, master_pos)
                if counter:
                    def act():
                        if controller.place(self.master_id, counter[0], counter[1]):
                            self.plate_loc = counter
                            return True
                        return False
                    self.move_or_act(controller, self.master_id, counter, act)
                return
        
        # If plate is ready with all ingredients -> pick it up
        if not missing and self.plate_loc:
            def act():
                return controller.pickup(self.master_id, self.plate_loc[0], self.plate_loc[1])
            self.move_or_act(controller, self.master_id, self.plate_loc, act)
            return
        
        # Need plate first
        if not self.plate_loc and not master_holding:
            master_shop = self.find_reachable_shop(controller, master_pos)
            if master_shop:
                def act():
                    return controller.buy(self.master_id, ShopCosts.PLATE, master_shop[0], master_shop[1])
                self.move_or_act(controller, self.master_id, master_shop, act)
                return
        
        if not missing:
            return
        
        # Process next missing ingredient
        next_req = missing[0]
        food_name, needs_chop, needs_cook = next_req
        
        # If holding the needed food
        if master_holding and master_holding.get('type') == 'Food':
            held_name = master_holding.get('food_name')
            held_chopped = master_holding.get('chopped', False)
            held_cooked = master_holding.get('cooked_stage', 0)
            
            if held_name == food_name:
                # Needs chopping?
                if needs_chop and not held_chopped:
                    counter = self.find_free_counter(controller, master_pos)
                    if counter:
                        def act():
                            return controller.place(self.master_id, counter[0], counter[1])
                        self.move_or_act(controller, self.master_id, counter, act)
                    return
                
                # Needs cooking?
                if needs_cook and held_cooked < 1:
                    cooker = self.find_free_cooker(controller)
                    if cooker:
                        def act():
                            return controller.start_cook(self.master_id, cooker[0], cooker[1])
                        self.move_or_act(controller, self.master_id, cooker, act)
                    return
                
                # Ready to plate!
                if self.plate_loc:
                    def act():
                        return controller.add_food_to_plate(self.master_id, self.plate_loc[0], self.plate_loc[1])
                    self.move_or_act(controller, self.master_id, self.plate_loc, act)
                return
            else:
                # Wrong food - put it down
                counter = self.find_free_counter(controller, master_pos)
                if counter:
                    def act():
                        return controller.place(self.master_id, counter[0], counter[1])
                    self.move_or_act(controller, self.master_id, counter, act)
                return
        
        # Not holding - need to get ingredient
        # Check for cooked version in pan
        if needs_cook:
            ready_pan = self.find_ready_pan(controller, food_name)
            if ready_pan:
                def act():
                    return controller.take_from_pan(self.master_id, ready_pan[0], ready_pan[1])
                self.move_or_act(controller, self.master_id, ready_pan, act)
                return
            
            cooking_pan = self.find_cooking_pan(controller, food_name)
            if cooking_pan:
                log(f"Waiting for {food_name} to cook")
                return
        
        # Check counters for ingredient
        if needs_chop:
            # Look for chopped version first
            loc = self.find_food_on_counter(controller, food_name, [(True, 0)])
            if loc:
                def act():
                    return controller.pickup(self.master_id, loc[0], loc[1])
                self.move_or_act(controller, self.master_id, loc, act)
                return
            
            # Look for raw version to chop
            loc = self.find_food_on_counter(controller, food_name, [(False, 0)])
            if loc:
                def act():
                    return controller.chop(self.master_id, loc[0], loc[1])
                self.move_or_act(controller, self.master_id, loc, act)
                return
        else:
            # Just need raw/cooked
            if needs_cook:
                loc = self.find_food_on_counter(controller, food_name)
            else:
                loc = self.find_food_on_counter(controller, food_name, [(False, 0)])
            if loc:
                def act():
                    return controller.pickup(self.master_id, loc[0], loc[1])
                self.move_or_act(controller, self.master_id, loc, act)
                return
        
        # Need to buy - give task to slave if master can't reach any shop
        master_shop = self.find_reachable_shop(controller, master_pos)
        if master_shop:
            def act():
                if hasattr(FoodType, food_name):
                    return controller.buy(self.master_id, getattr(FoodType, food_name), master_shop[0], master_shop[1])
                return False
            self.move_or_act(controller, self.master_id, master_shop, act)
        else:
            # Master can't reach shop - assign to slave (who should have their own shop)
            slave_shop = self.find_reachable_shop(controller, (0, 0))  # Will be updated in run_slave
            if slave_shop:
                log(f"Master can't reach shop, assigning {food_name} to slave")
                self.slave_task = ("BUY", food_name, slave_shop)

    def run_slave(self, controller, team, slave_pos, slave_holding, missing, master_pos):
        """Slave: fetch ingredients and place on transfer counters"""
        
        # If no task and nothing to do, stay idle
        if not self.slave_task and not missing:
            return
        
        # If holding something, put it on a counter that BOTH bots can access
        if slave_holding:
            log(f"Slave holding {slave_holding}, looking for transfer counter. Master at {master_pos}, Slave at {slave_pos}")
            # Find transfer counter (one that BOTH master and slave can reach)
            transfer_counter = None
            best_dist = 999
            
            for c in self.counters:
                tile = controller.get_tile(controller.get_team(), c[0], c[1])
                if tile and tile.item is None:
                    # Both bots must be able to interact with this counter
                    master_can = self.can_reach(controller, master_pos, c)
                    slave_can = self.can_reach(controller, slave_pos, c)
                    log(f"  Counter {c}: master_can={master_can}, slave_can={slave_can}")
                    
                    if master_can and slave_can:
                        # Prefer closest to slave
                        dist = max(abs(slave_pos[0] - c[0]), abs(slave_pos[1] - c[1]))
                        if dist < best_dist:
                            best_dist = dist
                            transfer_counter = c
                            log(f"    -> Best so far (dist={dist})")
            
            if transfer_counter:
                def act():
                    if controller.place(self.slave_id, transfer_counter[0], transfer_counter[1]):
                        log(f"Slave placed item at {transfer_counter}")
                        self.slave_task = None
                        return True
                    return False
                self.move_or_act(controller, self.slave_id, transfer_counter, act)
            else:
                log(f"No shared transfer counter found!")
            return
        
        # Check for assigned task
        if self.slave_task:
            action, food_name, target = self.slave_task
            
            if action == "BUY":
                def act():
                    if hasattr(FoodType, food_name):
                        if controller.buy(self.slave_id, getattr(FoodType, food_name), target[0], target[1]):
                            log(f"Slave bought {food_name}")
                            return True
                    return False
                self.move_or_act(controller, self.slave_id, target, act)
                return
        
        # Get slave's reachable shop
        slave_shop = self.find_reachable_shop(controller, slave_pos)
        master_state = controller.get_bot_state(self.master_id)
        master_pos = (master_state['x'], master_state['y'])
        master_shop = self.find_reachable_shop(controller, master_pos)
        
        # COOPERATIVE MODE: If both can reach shop, slave helps with different ingredients
        if master_shop and slave_shop and len(missing) > 1:
            # Work on the SECOND missing ingredient (master handles first)
            for food_name, needs_chop, needs_cook in missing[1:]:
                # Check if this ingredient is already in progress
                if self.find_food_on_counter(controller, food_name):
                    continue
                if needs_cook and (self.find_ready_pan(controller, food_name) or self.find_cooking_pan(controller, food_name)):
                    continue
                
                # Slave buys it
                def act():
                    if hasattr(FoodType, food_name):
                        if controller.buy(self.slave_id, getattr(FoodType, food_name), slave_shop[0], slave_shop[1]):
                            log(f"Slave cooperatively bought {food_name}")
                            return True
                    return False
                self.move_or_act(controller, self.slave_id, slave_shop, act)
                return
        
        # SPLIT MAP MODE: Master can't reach shop - slave helps by buying from their shop
        if not master_shop and slave_shop and missing:
            for food_name, needs_chop, needs_cook in missing:
                # Check if this ingredient is already in progress
                if self.find_food_on_counter(controller, food_name):
                    continue
                if needs_cook and (self.find_ready_pan(controller, food_name) or self.find_cooking_pan(controller, food_name)):
                    continue
                
                # Slave buys it from their shop
                def act():
                    if hasattr(FoodType, food_name):
                        if controller.buy(self.slave_id, getattr(FoodType, food_name), slave_shop[0], slave_shop[1]):
                            log(f"Slave proactively bought {food_name} from {slave_shop}")
                            return True
                    return False
                self.move_or_act(controller, self.slave_id, slave_shop, act)
                return
