"""
Greedy Order Picker Bot - Maximum Reward Chaser
=================================================

STRATEGY:
- Always targets the HIGHEST REWARD order regardless of complexity
- May abandon partially completed orders if a better one appears
- Doesn't consider time-to-complete, only raw reward value
- Accepts high-risk, high-reward orders others might skip

KEY DECISION LOGIC:
1. Every N turns, re-evaluate all orders and pick highest reward
2. If new order reward > current * 1.5, ABANDON current order
3. Both bots work on same order (maximizes chance of completion)
4. No sabotage - all resources toward big orders

WEAKNESSES THIS EXPLOITS:
- Bots that play conservatively with simple orders
- Maps where one big order win dominates many small wins
- Scenarios where reward variance is high

PERFORMANCE PROFILE:
- EXCELS ON: Maps with high-reward complex orders, long duration maps
- STRUGGLES ON: Short games, maps where consistency beats variance
"""

from collections import deque
from typing import Optional, List
from enum import Enum, auto

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass


INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True},
}


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map features
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sink_tables = []
        
        # Order tracking
        self.current_order = None
        self.last_check_turn = 0
        self.check_interval = 15  # Re-evaluate orders every 15 turns
        
        # Pipeline state
        self.state = 0
        self.plate_counter = None
        self.work_counter = None
        self.ingredients_done = []
        self.current_ingredient = None
        self.cooking_cooker = None
        
    def _init_map(self, controller, team):
        m = controller.get_map(team)
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                pos = (x, y)
                name = tile.tile_name
                if name == "SHOP": self.shops.append(pos)
                elif name == "COOKER": self.cookers.append(pos)
                elif name == "COUNTER": self.counters.append(pos)
                elif name == "SUBMIT": self.submits.append(pos)
                elif name == "TRASH": self.trashes.append(pos)
                elif name == "SINKTABLE": self.sink_tables.append(pos)
        self.initialized = True
    
    def bfs_move(self, controller, bot_id, target, team):
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        
        if max(abs(bx - target[0]), abs(by - target[1])) <= 1:
            return True
        
        m = controller.get_map(team)
        queue = deque([((bx, by), None)])
        visited = {(bx, by)}
        
        while queue:
            (x, y), first_move = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited or not m.is_tile_walkable(nx, ny):
                        continue
                    
                    visited.add((nx, ny))
                    fm = first_move or (dx, dy)
                    
                    if max(abs(nx - target[0]), abs(ny - target[1])) <= 1:
                        if controller.can_move(bot_id, fm[0], fm[1]):
                            controller.move(bot_id, fm[0], fm[1])
                        return False
                    queue.append(((nx, ny), fm))
        return False
    
    def get_nearest(self, pos, locations):
        if not locations:
            return None
        return min(locations, key=lambda p: max(abs(p[0]-pos[0]), abs(p[1]-pos[1])))
    
    def get_free_counter(self, controller, team, pos, exclude=None):
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
        return self.get_nearest(pos, free)
    
    def select_highest_reward_order(self, controller, team):
        """GREEDY: Select order with highest raw reward"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_reward = -1
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 20:  # Need minimum time
                continue
            
            # PURE GREED - highest reward wins
            if order['reward'] > best_reward:
                best_reward = order['reward']
                best = order
        
        return best
    
    def should_abandon_order(self, controller, team):
        """Check if we should abandon current order for a better one"""
        if not self.current_order:
            return True
        
        current_turn = controller.get_turn()
        
        # Only check periodically
        if current_turn - self.last_check_turn < self.check_interval:
            return False
        
        self.last_check_turn = current_turn
        
        best = self.select_highest_reward_order(controller, team)
        if not best:
            return False
        
        # ABANDON if new order is 1.5x better
        if best['reward'] > self.current_order['reward'] * 1.5:
            return True
        
        return False
    
    def reset_state(self):
        """Reset pipeline state (for abandoning orders)"""
        self.state = 0
        self.plate_counter = None
        self.work_counter = None
        self.ingredients_done = []
        self.current_ingredient = None
        self.cooking_cooker = None
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        turn = controller.get_turn()
        
        if not self.initialized:
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        bot_id = bots[0]
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        shop = self.get_nearest((bx, by), self.shops)
        submit = self.get_nearest((bx, by), self.submits)
        trash = self.get_nearest((bx, by), self.trashes)
        cooker = self.cookers[0] if self.cookers else None
        
        # Check for order abandonment
        if self.should_abandon_order(controller, team):
            new_order = self.select_highest_reward_order(controller, team)
            if new_order and (not self.current_order or 
                             new_order['order_id'] != self.current_order.get('order_id')):
                # ABANDON! Trash anything we're holding
                if holding:
                    if trash and self.bfs_move(controller, bot_id, trash, team):
                        controller.trash(bot_id, trash[0], trash[1])
                    return
                self.current_order = new_order
                self.reset_state()
        
        if not self.current_order:
            self.current_order = self.select_highest_reward_order(controller, team)
            if not self.current_order:
                return
        
        order = self.current_order
        
        # ========== PIPELINE ==========
        
        # State 0: Ensure pan
        if self.state == 0:
            needs_cook = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in order['required'])
            if needs_cook and cooker:
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if not isinstance(getattr(tile, 'item', None), Pan):
                    # Buy pan
                    if holding and holding.get('type') == 'Pan':
                        if self.bfs_move(controller, bot_id, cooker, team):
                            controller.place(bot_id, cooker[0], cooker[1])
                        return
                    elif shop and self.bfs_move(controller, bot_id, shop, team):
                        if money >= ShopCosts.PAN.buy_cost:
                            controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                        return
            self.state = 1
        
        # State 1: Get plate
        if self.state == 1:
            if holding and holding.get('type') == 'Plate':
                self.state = 2
            else:
                # Try sink table
                if self.sink_tables:
                    st = self.get_nearest((bx, by), self.sink_tables)
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self.bfs_move(controller, bot_id, st, team):
                            controller.take_clean_plate(bot_id, st[0], st[1])
                        return
                # Buy plate
                if shop and self.bfs_move(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
        
        # State 2: Place plate
        elif self.state == 2:
            counter = self.get_free_counter(controller, team, (bx, by))
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.plate_counter = counter
                self.state = 3
        
        # State 3: Process ingredients
        elif self.state == 3:
            # Check plate contents
            plate_contents = []
            if self.plate_counter:
                tile = controller.get_tile(team, self.plate_counter[0], self.plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            missing = [i for i in order['required'] if i not in plate_contents]
            if not missing:
                self.state = 10  # Pickup and submit
                return
            
            # Get next ingredient (prioritize cooking items)
            next_ing = None
            for ing in missing:
                if INGREDIENT_INFO.get(ing, {}).get('cook'):
                    next_ing = ing
                    break
            if not next_ing:
                next_ing = missing[0]
            
            self.current_ingredient = next_ing
            info = INGREDIENT_INFO.get(next_ing, {})
            
            if holding and holding.get('type') == 'Food':
                if info.get('chop') and not holding.get('chopped'):
                    self.state = 4  # Chop
                elif info.get('cook') and holding.get('cooked_stage', 0) < 1:
                    self.state = 6  # Cook
                else:
                    self.state = 9  # Add to plate
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                food_type = getattr(FoodType, next_ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        # State 4: Place for chop
        elif self.state == 4:
            exclude = {self.plate_counter}
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.work_counter = counter
                self.state = 5
        
        # State 5: Chop and pickup
        elif self.state == 5:
            if self.work_counter and self.bfs_move(controller, bot_id, self.work_counter, team):
                tile = controller.get_tile(team, self.work_counter[0], self.work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    if tile.item.chopped:
                        if controller.pickup(bot_id, self.work_counter[0], self.work_counter[1]):
                            info = INGREDIENT_INFO.get(self.current_ingredient, {})
                            if info.get('cook'):
                                self.state = 6
                            else:
                                self.state = 9
                    else:
                        controller.chop(bot_id, self.work_counter[0], self.work_counter[1])
        
        # State 6: Start cook
        elif self.state == 6:
            if cooker and self.bfs_move(controller, bot_id, cooker, team):
                controller.place(bot_id, cooker[0], cooker[1])
                self.cooking_cooker = cooker
                self.state = 7
        
        # State 7: Wait for cook
        elif self.state == 7:
            if self.cooking_cooker:
                tile = controller.get_tile(team, self.cooking_cooker[0], self.cooking_cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage >= 1:
                        self.state = 8
        
        # State 8: Take from pan
        elif self.state == 8:
            if self.cooking_cooker and self.bfs_move(controller, bot_id, self.cooking_cooker, team):
                if controller.take_from_pan(bot_id, self.cooking_cooker[0], self.cooking_cooker[1]):
                    if holding and holding.get('cooked_stage') == 2:
                        # Burnt - trash and retry
                        if trash and self.bfs_move(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                        self.state = 3
                    else:
                        self.state = 9
        
        # State 9: Add to plate
        elif self.state == 9:
            if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                if controller.add_food_to_plate(bot_id, self.plate_counter[0], self.plate_counter[1]):
                    self.state = 3  # Back to process more
        
        # State 10: Pickup plate
        elif self.state == 10:
            if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                controller.pickup(bot_id, self.plate_counter[0], self.plate_counter[1])
                self.state = 11
        
        # State 11: Submit
        elif self.state == 11:
            if submit and self.bfs_move(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.current_order = None
                    self.reset_state()
        
        # Helper bot: assist or wash dishes
        if len(bots) > 1:
            helper_id = bots[1]
            helper = controller.get_bot_state(helper_id)
            hx, hy = helper['x'], helper['y']
            # Just wash dishes if any
            for sink in self.sinks if hasattr(self, 'sinks') else []:
                tile = controller.get_tile(team, sink[0], sink[1])
                if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                    if self.bfs_move(controller, helper_id, sink, team):
                        controller.wash_sink(helper_id, sink[0], sink[1])
                    break
