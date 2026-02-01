"""
Efficiency Maximizer Bot - Optimal Reward-to-Effort Ratio
============================================================

STRATEGY:
- Calculates reward-to-effort ratio for each order
- Considers: ingredient cost, processing time, travel distance
- AVOIDS complex 5-ingredient orders (high variance, risky)
- Prefers 2-3 ingredient orders with no cooking
- Optimizes profit per turn, not total profit

KEY DECISION LOGIC:
1. Score = (reward - cost) / estimated_turns
2. Penalty for cooking items (adds variance/burn risk)
3. Bonus for orders matching current map layout
4. Hard cap: Reject any order with >3 ingredients

WEAKNESSES THIS EXPLOITS:
- Bots that take risky complex orders
- Maps where simple orders dominate
- Scenarios with tight time limits

PERFORMANCE PROFILE:
- EXCELS ON: Fast-paced maps, short duration orders, simple layouts
- STRUGGLES ON: Maps with only complex orders, high-reward scenarios
"""

from collections import deque
from typing import Optional, List, Tuple
from enum import Enum, auto
from dataclasses import dataclass

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass


INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'turns': 3},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'turns': 25},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'turns': 8},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'turns': 3},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'turns': 30},
}


@dataclass
class OrderEfficiency:
    order_id: int
    required: List[str]
    reward: int
    cost: int
    turns_needed: int
    efficiency: float  # (reward - cost) / turns
    
    @classmethod
    def calculate(cls, order: dict, current_turn: int):
        required = order['required']
        reward = order['reward']
        
        # Calculate cost
        cost = ShopCosts.PLATE.buy_cost
        for ing in required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50})
            cost += info['cost']
        
        # Calculate turns needed (base overhead + ingredient processing)
        turns = 15  # Base overhead
        cook_count = 0
        
        for ing in required:
            info = INGREDIENT_INFO.get(ing, {'turns': 5})
            turns += info['turns']
            if info.get('cook'):
                cook_count += 1
        
        # Cooking must be sequential
        if cook_count > 1:
            turns += (cook_count - 1) * 20
        
        profit = reward - cost
        efficiency = profit / max(turns, 1)
        
        return cls(
            order_id=order['order_id'],
            required=required,
            reward=reward,
            cost=cost,
            turns_needed=turns,
            efficiency=efficiency
        )


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
        
        # State
        self.current_order = None
        self.state = 0
        self.plate_counter = None
        self.work_counter = None
        self.current_ingredient = None
        self.cooking_cooker = None
        
        # Efficiency tracking
        self.orders_completed = 0
        self.total_profit = 0
        
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
    
    def select_most_efficient_order(self, controller, team):
        """Select order with best efficiency (profit/turn), avoiding complex orders"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_efficiency = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 25:
                continue
            
            # HARD RULE: Reject orders with >3 ingredients
            if len(order['required']) > 3:
                continue
            
            # SOFT PENALTY: Avoid orders with multiple cooking items
            cook_count = sum(1 for i in order['required'] 
                           if INGREDIENT_INFO.get(i, {}).get('cook'))
            if cook_count > 1:
                continue  # Too risky
            
            eff = OrderEfficiency.calculate(order, current_turn)
            
            # Bonus for no-cook orders (faster, no burn risk)
            if cook_count == 0:
                eff.efficiency *= 1.3
            
            # Bonus for single ingredient (super fast)
            if len(order['required']) == 1:
                eff.efficiency *= 1.2
            
            if eff.efficiency > best_efficiency:
                best_efficiency = eff.efficiency
                best = order
        
        return best
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        
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
        
        # Select efficient order
        if not self.current_order:
            self.current_order = self.select_most_efficient_order(controller, team)
            self.state = 0
            self.plate_counter = None
            self.work_counter = None
            self.current_ingredient = None
            
        if not self.current_order:
            return
        
        order = self.current_order
        
        # ========== EFFICIENT PIPELINE ==========
        
        # State 0: Check if cooking needed, setup pan
        if self.state == 0:
            needs_cook = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in order['required'])
            if needs_cook and cooker:
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if not isinstance(getattr(tile, 'item', None), Pan):
                    if holding and holding.get('type') == 'Pan':
                        if self.bfs_move(controller, bot_id, cooker, team):
                            controller.place(bot_id, cooker[0], cooker[1])
                        return
                    elif shop and self.bfs_move(controller, bot_id, shop, team):
                        if money >= ShopCosts.PAN.buy_cost:
                            controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                        return
            self.state = 1
        
        # State 1: Get plate (prefer recycling)
        if self.state == 1:
            if holding and holding.get('type') == 'Plate':
                self.state = 2
            else:
                # Recycle first - more efficient!
                if self.sink_tables:
                    st = self.get_nearest((bx, by), self.sink_tables)
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self.bfs_move(controller, bot_id, st, team):
                            controller.take_clean_plate(bot_id, st[0], st[1])
                        return
                # Buy if must
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
        
        # State 3: Process ingredients (optimized order: simple first, then cook)
        elif self.state == 3:
            plate_contents = []
            if self.plate_counter:
                tile = controller.get_tile(team, self.plate_counter[0], self.plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            missing = [i for i in order['required'] if i not in plate_contents]
            if not missing:
                self.state = 10
                return
            
            # Efficient ordering: simple (SAUCE, NOODLES) first, then chop, then cook
            next_ing = None
            for ing in missing:
                info = INGREDIENT_INFO.get(ing, {})
                if not info.get('chop') and not info.get('cook'):
                    next_ing = ing
                    break
            if not next_ing:
                for ing in missing:
                    info = INGREDIENT_INFO.get(ing, {})
                    if info.get('chop') and not info.get('cook'):
                        next_ing = ing
                        break
            if not next_ing:
                next_ing = missing[0]
            
            self.current_ingredient = next_ing
            info = INGREDIENT_INFO.get(next_ing, {})
            
            if holding and holding.get('type') == 'Food':
                if info.get('chop') and not holding.get('chopped'):
                    self.state = 4
                elif info.get('cook') and holding.get('cooked_stage', 0) < 1:
                    self.state = 6
                else:
                    self.state = 9
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                food_type = getattr(FoodType, next_ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        # State 4-5: Chop
        elif self.state == 4:
            exclude = {self.plate_counter}
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.work_counter = counter
                self.state = 5
        
        elif self.state == 5:
            if self.work_counter and self.bfs_move(controller, bot_id, self.work_counter, team):
                tile = controller.get_tile(team, self.work_counter[0], self.work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    if tile.item.chopped:
                        if controller.pickup(bot_id, self.work_counter[0], self.work_counter[1]):
                            info = INGREDIENT_INFO.get(self.current_ingredient, {})
                            self.state = 6 if info.get('cook') else 9
                    else:
                        controller.chop(bot_id, self.work_counter[0], self.work_counter[1])
        
        # State 6-8: Cook
        elif self.state == 6:
            if cooker and self.bfs_move(controller, bot_id, cooker, team):
                controller.place(bot_id, cooker[0], cooker[1])
                self.cooking_cooker = cooker
                self.state = 7
        
        elif self.state == 7:
            if self.cooking_cooker:
                tile = controller.get_tile(team, self.cooking_cooker[0], self.cooking_cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage >= 1:
                        self.state = 8
        
        elif self.state == 8:
            if self.cooking_cooker and self.bfs_move(controller, bot_id, self.cooking_cooker, team):
                if controller.take_from_pan(bot_id, self.cooking_cooker[0], self.cooking_cooker[1]):
                    self.state = 9
        
        # State 9: Add to plate
        elif self.state == 9:
            if holding and holding.get('cooked_stage') == 2:
                # Burnt - trash
                if trash and self.bfs_move(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
                    self.state = 3
                return
            
            if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                if controller.add_food_to_plate(bot_id, self.plate_counter[0], self.plate_counter[1]):
                    self.state = 3
        
        # State 10-11: Submit
        elif self.state == 10:
            if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                controller.pickup(bot_id, self.plate_counter[0], self.plate_counter[1])
                self.state = 11
        
        elif self.state == 11:
            if submit and self.bfs_move(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.orders_completed += 1
                    self.current_order = None
                    self.state = 0
        
        # Helper bot: wash dishes for efficiency
        if len(bots) > 1:
            helper = bots[1]
            hbot = controller.get_bot_state(helper)
            hx, hy = hbot['x'], hbot['y']
            
            # Find sink with dirty plates
            for x in range(self.map.width):
                for y in range(self.map.height):
                    tile = controller.get_tile(team, x, y)
                    if tile and tile.tile_name == "SINK":
                        if getattr(tile, 'num_dirty_plates', 0) > 0:
                            if self.bfs_move(controller, helper, (x, y), team):
                                controller.wash_sink(helper, x, y)
                            return
