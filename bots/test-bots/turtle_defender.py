"""
Turtle Defender Bot - Pure Throughput Strategy
================================================

STRATEGY:
- NEVER sabotages enemy (wastes time/resources)
- Uses BOTH bots for parallel order completion
- Bot 1 handles cooking items, Bot 2 handles non-cooking items
- Maximizes throughput on maps with abundant resources
- Ignores enemy sabotage - focuses purely on own production

KEY DECISION LOGIC:
1. Select simplest orders first (1-2 ingredients, no cooking preferred)
2. Split work: Bot 1 takes cooking, Bot 2 takes non-cooking
3. Both bots can submit simultaneously to maximize submissions
4. Recycles plates from sink tables aggressively

WEAKNESSES THIS EXPLOITS:
- Bots that waste time on sabotage
- Maps where pure speed wins over disruption
- Late-game scenarios where throughput matters

PERFORMANCE PROFILE:
- EXCELS ON: Large maps (more space to work), resource-rich maps, long games
- STRUGGLES ON: Small maps (collision issues), maps where sabotage is powerful
"""

from collections import deque
from typing import Tuple, Optional, List
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


class BotRole(Enum):
    COOKER = auto()    # Handles items that need cooking
    RUNNER = auto()    # Handles simple items, fast submissions


class BotState(Enum):
    IDLE = auto()
    BUY_PAN = auto()
    PLACE_PAN = auto()
    BUY_PLATE = auto()
    PLACE_PLATE = auto()
    GET_PLATE = auto()
    BUY_INGREDIENT = auto()
    PLACE_CHOP = auto()
    CHOP = auto()
    PICKUP = auto()
    START_COOK = auto()
    WAIT_COOK = auto()
    TAKE_PAN = auto()
    ADD_PLATE = auto()
    PICKUP_PLATE = auto()
    SUBMIT = auto()
    TRASH = auto()


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
        self.sinks = []
        self.sink_tables = []
        
        # Bot states (independent pipelines)
        self.bot_states = {}
        self.bot_targets = {}
        self.bot_items = {}
        self.bot_plates = {}  # Track plate locations per bot
        self.bot_orders = {}  # Current order per bot
        
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
                elif name == "SINK": self.sinks.append(pos)
                elif name == "SINKTABLE": self.sink_tables.append(pos)
        self.initialized = True
    
    def bfs_move(self, controller, bot_id, target, team):
        """Move toward target. Returns True if adjacent."""
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
        """Find nearest free counter"""
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
        return self.get_nearest(pos, free)
    
    def select_order_for_bot(self, controller, team, prefer_cooking=False):
        """Select best order for bot type"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 30:
                continue
            
            required = order['required']
            has_cooking = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in required)
            
            # Score based on simplicity
            score = 100 - len(required) * 20
            score += order['reward'] / 10
            
            # Cooker bot prefers orders with cooking
            if prefer_cooking and has_cooking:
                score += 30
            elif not prefer_cooking and not has_cooking:
                score += 30
            
            if score > best_score:
                best_score = score
                best = order
        
        return best
    
    def execute_bot(self, controller, bot_id, team, role: BotRole, other_bot_plate=None):
        """Execute bot pipeline with role specialization"""
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        state = self.bot_states.get(bot_id, BotState.IDLE)
        target = self.bot_targets.get(bot_id)
        item = self.bot_items.get(bot_id)
        plate_loc = self.bot_plates.get(bot_id)
        order = self.bot_orders.get(bot_id)
        
        shop = self.get_nearest((bx, by), self.shops)
        submit = self.get_nearest((bx, by), self.submits)
        trash = self.get_nearest((bx, by), self.trashes)
        
        prefer_cooking = (role == BotRole.COOKER)
        
        # ========== STATE MACHINE ==========
        
        if state == BotState.IDLE:
            # Select order
            order = self.select_order_for_bot(controller, team, prefer_cooking)
            if not order:
                return
            
            self.bot_orders[bot_id] = order
            
            # Check if we need a pan (cooker bot only)
            if prefer_cooking and self.cookers:
                cooker = self.cookers[0]
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if not isinstance(getattr(tile, 'item', None), Pan):
                    self.bot_states[bot_id] = BotState.BUY_PAN
                    return
            
            # Get plate
            self.bot_states[bot_id] = BotState.GET_PLATE
        
        elif state == BotState.BUY_PAN:
            if holding and holding.get('type') == 'Pan':
                self.bot_states[bot_id] = BotState.PLACE_PAN
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                if money >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
        
        elif state == BotState.PLACE_PAN:
            cooker = self.cookers[0] if self.cookers else None
            if cooker and self.bfs_move(controller, bot_id, cooker, team):
                controller.place(bot_id, cooker[0], cooker[1])
                self.bot_states[bot_id] = BotState.GET_PLATE
        
        elif state == BotState.GET_PLATE:
            if holding and holding.get('type') == 'Plate':
                self.bot_states[bot_id] = BotState.PLACE_PLATE
            else:
                # Try sink table first
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
        
        elif state == BotState.PLACE_PLATE:
            exclude = {other_bot_plate} if other_bot_plate else None
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.bot_plates[bot_id] = counter
                self.bot_states[bot_id] = BotState.BUY_INGREDIENT
        
        elif state == BotState.BUY_INGREDIENT:
            order = self.bot_orders.get(bot_id)
            if not order:
                self.bot_states[bot_id] = BotState.IDLE
                return
            
            # Check what's on plate
            plate_loc = self.bot_plates.get(bot_id)
            plate_contents = []
            if plate_loc:
                tile = controller.get_tile(team, plate_loc[0], plate_loc[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            # Find missing ingredient
            missing = [i for i in order['required'] if i not in plate_contents]
            if not missing:
                self.bot_states[bot_id] = BotState.PICKUP_PLATE
                return
            
            # Pick next ingredient based on role
            next_ing = None
            for ing in missing:
                info = INGREDIENT_INFO.get(ing, {})
                if prefer_cooking and info.get('cook'):
                    next_ing = ing
                    break
                elif not prefer_cooking and not info.get('cook'):
                    next_ing = ing
                    break
            if not next_ing:
                next_ing = missing[0]
            
            self.bot_items[bot_id] = next_ing
            
            if holding:
                info = INGREDIENT_INFO.get(next_ing, {})
                if info.get('chop'):
                    self.bot_states[bot_id] = BotState.PLACE_CHOP
                elif info.get('cook'):
                    self.bot_states[bot_id] = BotState.START_COOK
                else:
                    self.bot_states[bot_id] = BotState.ADD_PLATE
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                food_type = getattr(FoodType, next_ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        elif state == BotState.PLACE_CHOP:
            exclude = {other_bot_plate, self.bot_plates.get(bot_id)}
            counter = self.get_free_counter(controller, team, (bx, by), exclude)
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.bot_targets[bot_id] = counter
                self.bot_states[bot_id] = BotState.CHOP
        
        elif state == BotState.CHOP:
            if target and self.bfs_move(controller, bot_id, target, team):
                controller.chop(bot_id, target[0], target[1])
                self.bot_states[bot_id] = BotState.PICKUP
        
        elif state == BotState.PICKUP:
            if target and self.bfs_move(controller, bot_id, target, team):
                controller.pickup(bot_id, target[0], target[1])
                info = INGREDIENT_INFO.get(item, {})
                if info.get('cook'):
                    self.bot_states[bot_id] = BotState.START_COOK
                else:
                    self.bot_states[bot_id] = BotState.ADD_PLATE
        
        elif state == BotState.START_COOK:
            cooker = self.cookers[0] if self.cookers else None
            if cooker and self.bfs_move(controller, bot_id, cooker, team):
                controller.place(bot_id, cooker[0], cooker[1])
                self.bot_targets[bot_id] = cooker
                self.bot_states[bot_id] = BotState.WAIT_COOK
        
        elif state == BotState.WAIT_COOK:
            cooker = target or (self.cookers[0] if self.cookers else None)
            if cooker:
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage == 1:
                        self.bot_states[bot_id] = BotState.TAKE_PAN
                    elif pan.food and pan.food.cooked_stage == 2:
                        self.bot_states[bot_id] = BotState.TAKE_PAN  # Burnt - take and trash
        
        elif state == BotState.TAKE_PAN:
            cooker = target or (self.cookers[0] if self.cookers else None)
            if holding:
                if holding.get('cooked_stage') == 2:
                    self.bot_states[bot_id] = BotState.TRASH
                else:
                    self.bot_states[bot_id] = BotState.ADD_PLATE
            elif cooker and self.bfs_move(controller, bot_id, cooker, team):
                controller.take_from_pan(bot_id, cooker[0], cooker[1])
        
        elif state == BotState.ADD_PLATE:
            plate_loc = self.bot_plates.get(bot_id)
            if plate_loc and self.bfs_move(controller, bot_id, plate_loc, team):
                if controller.add_food_to_plate(bot_id, plate_loc[0], plate_loc[1]):
                    self.bot_states[bot_id] = BotState.BUY_INGREDIENT
        
        elif state == BotState.PICKUP_PLATE:
            plate_loc = self.bot_plates.get(bot_id)
            if plate_loc and self.bfs_move(controller, bot_id, plate_loc, team):
                controller.pickup(bot_id, plate_loc[0], plate_loc[1])
                self.bot_states[bot_id] = BotState.SUBMIT
        
        elif state == BotState.SUBMIT:
            if submit and self.bfs_move(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.bot_orders[bot_id] = None
                    self.bot_plates[bot_id] = None
                    self.bot_states[bot_id] = BotState.IDLE
        
        elif state == BotState.TRASH:
            if trash and self.bfs_move(controller, bot_id, trash, team):
                controller.trash(bot_id, trash[0], trash[1])
                self.bot_states[bot_id] = BotState.BUY_INGREDIENT
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        
        if not self.initialized:
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Bot 0: COOKER role (handles cooking items)
        other_plate = self.bot_plates.get(bots[1]) if len(bots) > 1 else None
        self.execute_bot(controller, bots[0], team, BotRole.COOKER, other_plate)
        
        # Bot 1: RUNNER role (handles simple items)
        if len(bots) > 1:
            other_plate = self.bot_plates.get(bots[0])
            self.execute_bot(controller, bots[1], team, BotRole.RUNNER, other_plate)
