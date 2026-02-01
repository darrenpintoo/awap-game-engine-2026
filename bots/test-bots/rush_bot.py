"""
Rush Bot - Speed Over Optimization
====================================

STRATEGY:
- Completes MANY simple orders quickly
- Never does cooking orders (too slow)
- Both bots work independently on separate simple orders
- Sacrifices per-order profit for volume
- Ideal for maps with many simple orders

KEY DECISION LOGIC:
1. ONLY accept orders with 1-2 non-cooking ingredients
2. Both bots run independent parallel pipelines
3. No waiting, no complex coordination
4. Skip any order that requires cooking

WEAKNESSES THIS EXPLOITS:
- Bots that overthink order selection
- Maps with many simple high-frequency orders
- Short games where speed beats optimization

PERFORMANCE PROFILE:
- EXCELS ON: Small/compact maps, maps with SAUCE/NOODLES orders, pressure_cooker style maps
- STRUGGLES ON: Maps with only complex orders, maps where cooking is required
"""

from collections import deque
from typing import Optional, List

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass


SIMPLE_INGREDIENTS = ['SAUCE', 'NOODLES', 'ONIONS']  # No cooking required


class BotPipeline:
    """Independent pipeline for each bot"""
    def __init__(self):
        self.state = 0
        self.order = None
        self.plate_counter = None
        self.work_counter = None
        self.ingredients_done = []
        self.current_ingredient = None


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map features
        self.shops = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sink_tables = []
        
        # Independent bot pipelines
        self.pipelines = {}
        
    def _init_map(self, controller, team):
        m = controller.get_map(team)
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                pos = (x, y)
                name = tile.tile_name
                if name == "SHOP": self.shops.append(pos)
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
    
    def select_rush_order(self, controller, team, exclude_ids=None):
        """Select simplest order (no cooking, 1-2 ingredients)"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        candidates = []
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            if exclude_ids and order['order_id'] in exclude_ids:
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 15:
                continue
            
            required = order['required']
            
            # RUSH RULE: Max 2 ingredients, NO COOKING
            if len(required) > 2:
                continue
            
            has_cooking = any(i in ['EGG', 'MEAT'] for i in required)
            if has_cooking:
                continue
            
            # Prefer fewest ingredients
            candidates.append((len(required), order))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return None
    
    def execute_bot(self, controller, bot_id, team, other_counter=None, other_order_id=None):
        """Execute rush pipeline for single bot"""
        if bot_id not in self.pipelines:
            self.pipelines[bot_id] = BotPipeline()
        
        p = self.pipelines[bot_id]
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        shop = self.get_nearest((bx, by), self.shops)
        submit = self.get_nearest((bx, by), self.submits)
        trash = self.get_nearest((bx, by), self.trashes)
        
        # State 0: Select order
        if p.state == 0:
            exclude = {other_order_id} if other_order_id else None
            p.order = self.select_rush_order(controller, team, exclude)
            if not p.order:
                return None
            p.ingredients_done = []
            p.state = 1
        
        if not p.order:
            return None
        
        # State 1: Get plate
        if p.state == 1:
            if holding and holding.get('type') == 'Plate':
                p.state = 2
            else:
                # Try sink table first (free plates!)
                if self.sink_tables:
                    st = self.get_nearest((bx, by), self.sink_tables)
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self.bfs_move(controller, bot_id, st, team):
                            controller.take_clean_plate(bot_id, st[0], st[1])
                        return p.order['order_id']
                # Buy plate
                if shop and self.bfs_move(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
            return p.order['order_id']
        
        # State 2: Place plate
        if p.state == 2:
            exclude = {other_counter} if other_counter else None
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                p.plate_counter = counter
                p.state = 3
            return p.order['order_id']
        
        # State 3: Process ingredients
        if p.state == 3:
            plate_contents = []
            if p.plate_counter:
                tile = controller.get_tile(team, p.plate_counter[0], p.plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            missing = [i for i in p.order['required'] if i not in plate_contents]
            if not missing:
                p.state = 6
                return p.order['order_id']
            
            p.current_ingredient = missing[0]
            
            if holding and holding.get('type') == 'Food':
                # Check if needs chop
                if p.current_ingredient == 'ONIONS' and not holding.get('chopped'):
                    p.state = 4
                else:
                    p.state = 5
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                food_type = getattr(FoodType, p.current_ingredient, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
            return p.order['order_id']
        
        # State 4: Chop (only for ONIONS)
        if p.state == 4:
            exclude = {other_counter, p.plate_counter}
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if holding:
                if counter and self.bfs_move(controller, bot_id, counter, team):
                    controller.place(bot_id, counter[0], counter[1])
                    p.work_counter = counter
            elif p.work_counter:
                if self.bfs_move(controller, bot_id, p.work_counter, team):
                    tile = controller.get_tile(team, p.work_counter[0], p.work_counter[1])
                    if tile and isinstance(getattr(tile, 'item', None), Food):
                        if tile.item.chopped:
                            if controller.pickup(bot_id, p.work_counter[0], p.work_counter[1]):
                                p.state = 5
                        else:
                            controller.chop(bot_id, p.work_counter[0], p.work_counter[1])
            return p.order['order_id']
        
        # State 5: Add to plate
        if p.state == 5:
            if p.plate_counter and self.bfs_move(controller, bot_id, p.plate_counter, team):
                if controller.add_food_to_plate(bot_id, p.plate_counter[0], p.plate_counter[1]):
                    p.state = 3  # Back to process more
            return p.order['order_id']
        
        # State 6: Pickup plate
        if p.state == 6:
            if p.plate_counter and self.bfs_move(controller, bot_id, p.plate_counter, team):
                controller.pickup(bot_id, p.plate_counter[0], p.plate_counter[1])
                p.state = 7
            return p.order['order_id']
        
        # State 7: Submit
        if p.state == 7:
            if submit and self.bfs_move(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    p.order = None
                    p.plate_counter = None
                    p.state = 0
            return None if p.order is None else p.order['order_id']
        
        return p.order['order_id'] if p.order else None
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        
        if not self.initialized:
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Execute both bots independently for MAXIMUM SPEED
        order_id_0 = None
        counter_0 = None
        
        # Bot 0
        order_id_0 = self.execute_bot(controller, bots[0], team)
        if bots[0] in self.pipelines:
            counter_0 = self.pipelines[bots[0]].plate_counter
        
        # Bot 1: Work on different order
        if len(bots) > 1:
            self.execute_bot(controller, bots[1], team, 
                           other_counter=counter_0, 
                           other_order_id=order_id_0)
