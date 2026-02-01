"""
Junior Champion Bot - AWAP 2026
===============================
A competitive bot that uses:
- Fast BFS precomputation for O(1) distance lookups
- Position-aware greedy order selection
- Parallel bot execution (both bots work on different orders)
- Priority on simple orders (quick wins)
- Smart cooking state machine

Designed to score ~7-10k on orbit map.
"""

from collections import deque
from typing import Tuple, Optional, List, Dict, Set, Any
from enum import Enum, auto

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food


# =============================================================================
# CONSTANTS
# =============================================================================

FOOD_INFO = {
    'EGG':     {'type': FoodType.EGG,     'cost': 20, 'chop': False, 'cook': True,  'base_time': 25},
    'ONIONS':  {'type': FoodType.ONIONS,  'cost': 30, 'chop': True,  'cook': False, 'base_time': 8},
    'MEAT':    {'type': FoodType.MEAT,    'cost': 80, 'chop': True,  'cook': True,  'base_time': 35},
    'NOODLES': {'type': FoodType.NOODLES, 'cost': 40, 'chop': False, 'cook': False, 'base_time': 3},
    'SAUCE':   {'type': FoodType.SAUCE,   'cost': 10, 'chop': False, 'cook': False, 'base_time': 3},
}

COOK_TIME = 20  # Turns to cook


class BotState(Enum):
    """State machine states"""
    IDLE = auto()
    SELECT_ORDER = auto()
    BUY_INGREDIENT = auto()
    CHOP_PLACE = auto()
    CHOP_ACTION = auto()
    CHOP_PICKUP = auto()
    COOK_PLACE = auto()
    COOK_WAIT = auto()
    COOK_TAKE = auto()
    STORE_FOOD = auto()
    BUY_PLATE = auto()
    PLACE_PLATE = auto()
    ADD_FOOD = auto()
    PICKUP_PLATE = auto()
    SUBMIT = auto()
    TRASH = auto()


# =============================================================================
# MAP ANALYZER - Precompute BFS distances
# =============================================================================

class MapAnalyzer:
    """Precomputes map analysis and BFS distances"""
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        
        # Tile locations
        self.shops: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.walkable: Set[Tuple[int, int]] = set()
        
        # BFS distance cache
        self.dist_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        
        self._analyze_tiles(map_obj)
        self._precompute_distances()
    
    def _analyze_tiles(self, map_obj):
        """Scan map and categorize tiles"""
        for x in range(self.width):
            for y in range(self.height):
                tile = map_obj.tiles[x][y]
                name = tile.tile_name
                pos = (x, y)
                
                if name == 'SHOP':
                    self.shops.append(pos)
                elif name == 'COUNTER':
                    self.counters.append(pos)
                elif name == 'COOKER':
                    self.cookers.append(pos)
                elif name == 'SINK':
                    self.sinks.append(pos)
                elif name == 'SINKTABLE':
                    self.sink_tables.append(pos)
                elif name == 'SUBMIT':
                    self.submits.append(pos)
                elif name == 'TRASH':
                    self.trashes.append(pos)
                
                if tile.is_walkable:
                    self.walkable.add(pos)
    
    def _precompute_distances(self):
        """BFS from every walkable tile"""
        for start in self.walkable:
            self.dist_cache[start] = self._bfs_from(start)
    
    def _bfs_from(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """BFS from start to all reachable tiles"""
        distances = {start: 0}
        queue = deque([start])
        
        while queue:
            x, y = queue.popleft()
            curr_dist = distances[(x, y)]
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in distances:
                        continue
                    if (nx, ny) not in self.walkable:
                        continue
                    distances[(nx, ny)] = curr_dist + 1
                    queue.append((nx, ny))
        
        return distances
    
    def get_distance(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Get distance between two positions"""
        if from_pos not in self.dist_cache:
            return 9999
        return self.dist_cache[from_pos].get(to_pos, 9999)
    
    def get_distance_to_tile(self, from_pos: Tuple[int, int], tile_pos: Tuple[int, int]) -> int:
        """Get distance to reach adjacent to a tile"""
        if from_pos not in self.dist_cache:
            return 9999
        
        tx, ty = tile_pos
        best_dist = 9999
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                adj = (tx + dx, ty + dy)
                if adj in self.dist_cache[from_pos]:
                    best_dist = min(best_dist, self.dist_cache[from_pos][adj])
        
        return best_dist
    
    def get_nearest(self, from_pos: Tuple[int, int], tile_list: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find nearest tile from a list"""
        best_dist = 9999
        best_tile = None
        
        for tile_pos in tile_list:
            dist = self.get_distance_to_tile(from_pos, tile_pos)
            if dist < best_dist:
                best_dist = dist
                best_tile = tile_pos
        
        return best_tile
    
    def get_next_step(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                      occupied: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Get next step (dx, dy) towards a target"""
        if from_pos == to_pos:
            return (0, 0)
        
        occupied = occupied or set()
        fx, fy = from_pos
        best_step = None
        best_dist = 9999
        
        # Find adjacent walkable tiles for non-walkable targets
        targets = []
        if to_pos in self.walkable:
            targets.append(to_pos)
        else:
            tx, ty = to_pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    adj = (tx + dx, ty + dy)
                    if adj in self.walkable:
                        targets.append(adj)
        
        if not targets:
            return None
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = fx + dx, fy + dy
                if (nx, ny) not in self.walkable:
                    continue
                if (nx, ny) in occupied:
                    continue
                
                if (nx, ny) not in self.dist_cache:
                    continue
                for t in targets:
                    d = self.dist_cache[(nx, ny)].get(t, 9999)
                    if d < best_dist:
                        best_dist = d
                        best_step = (dx, dy)
        
        return best_step
    
    def can_reach(self, from_pos: Tuple[int, int], tile_pos: Tuple[int, int]) -> bool:
        """Check if we can reach (adjacent to) a tile"""
        return self.get_distance_to_tile(from_pos, tile_pos) < 9999


# =============================================================================
# BOT TASK - Per-bot state tracking
# =============================================================================

class BotTask:
    """Tracks state for a single bot"""
    
    def __init__(self):
        self.state = BotState.IDLE
        self.order = None
        self.order_id = None
        
        # Ingredient processing
        self.ingredients_queue: List[str] = []
        self.ingredients_done: List[str] = []
        self.current_ingredient: Optional[str] = None
        self.need_chop = False
        self.need_cook = False
        
        # Locations
        self.plate_counter: Optional[Tuple[int, int]] = None
        self.work_counter: Optional[Tuple[int, int]] = None
        self.cooker_pos: Optional[Tuple[int, int]] = None
        self.food_locations: Dict[str, Tuple[int, int]] = {}
        
        # Tracking
        self.stuck_count = 0
        self.last_pos = None
        self.idle_turns = 0  # How long bot has been idle/selecting
    
    def reset(self):
        """Reset to idle state"""
        self.__init__()


# =============================================================================
# MAIN BOT PLAYER
# =============================================================================

class BotPlayer:
    """Main bot controller"""
    
    def __init__(self, map_copy):
        self.map_analyzer = MapAnalyzer(map_copy)
        self.bot_tasks: Dict[int, BotTask] = {}
        self.assigned_orders: Set[int] = set()
        self.allocated_counters: Set[Tuple[int, int]] = set()
        self.allocated_cookers: Set[Tuple[int, int]] = set()
        self.turn = 0
        
        # Bot roles: assign at first turn based on position
        self.cook_bot: Optional[int] = None  # Bot that handles cooking orders
        self.runner_bot: Optional[int] = None  # Bot that handles non-cooking orders
        self.roles_assigned = False
        
        # Map type detection
        self.is_split_map = False  # Bots can't reach each other
    
    def _assign_roles(self, controller: RobotController):
        """Assign roles to bots based on their position relative to cookers"""
        if self.roles_assigned:
            return
        
        team = controller.get_team()
        bot_ids = controller.get_team_bot_ids(team)
        
        if len(bot_ids) < 2:
            # Single bot - it does everything
            self.roles_assigned = True
            return
        
        # Detect split map: check if bots can reach each other
        positions = []
        for bid in bot_ids:
            state = controller.get_bot_state(bid)
            if state:
                positions.append((state['x'], state['y']))
        
        if len(positions) >= 2:
            dist = self.map_analyzer.get_distance(positions[0], positions[1])
            if dist >= 9999:
                self.is_split_map = True
                self.roles_assigned = True
                return  # Don't assign roles on split maps
        
        # Note: all_orders_need_cooking is checked dynamically in _select_order
        
        # Find which bot is better suited for each role
        # Runner bot should be close to shop (for buying)
        # Cook bot should be close to cooker
        if not self.map_analyzer.cookers or not self.map_analyzer.shops:
            self.roles_assigned = True
            return
        
        cooker = self.map_analyzer.cookers[0]
        shop = self.map_analyzer.shops[0]
        
        bot_info = []
        for bid in bot_ids:
            state = controller.get_bot_state(bid)
            if state:
                pos = (state['x'], state['y'])
                dist_cooker = self.map_analyzer.get_distance_to_tile(pos, cooker)
                dist_shop = self.map_analyzer.get_distance_to_tile(pos, shop)
                bot_info.append((bid, dist_cooker, dist_shop))
        
        if len(bot_info) >= 2:
            # Assign based on relative advantages
            # Bot closer to shop should be runner, bot closer to cooker should be cook
            b0, d0_cook, d0_shop = bot_info[0]
            b1, d1_cook, d1_shop = bot_info[1]
            
            # Calculate advantage scores
            # Runner advantage = how much closer to shop vs other bot
            # Cook advantage = how much closer to cooker vs other bot
            b0_runner_adv = d1_shop - d0_shop  # positive if b0 is better runner
            b0_cook_adv = d1_cook - d0_cook    # positive if b0 is better cook
            
            # Assign roles based on relative strengths
            if b0_runner_adv > b0_cook_adv:
                # b0 has bigger advantage at running
                self.runner_bot = b0
                self.cook_bot = b1
            else:
                # b0 has bigger advantage at cooking (or equal)
                self.cook_bot = b0
                self.runner_bot = b1
        
        self.roles_assigned = True
    
    def play_turn(self, controller: RobotController):
        """Main entry point"""
        self.turn = controller.get_turn()
        team = controller.get_team()
        bot_ids = controller.get_team_bot_ids(team)
        
        # Assign roles on first turn
        if not self.roles_assigned:
            self._assign_roles(controller)
        
        # Get occupied positions
        occupied = set()
        for bid in bot_ids:
            state = controller.get_bot_state(bid)
            if state:
                occupied.add((state['x'], state['y']))
        
        # Initialize tasks for new bots
        for bot_id in bot_ids:
            if bot_id not in self.bot_tasks:
                self.bot_tasks[bot_id] = BotTask()
        
        # Execute each bot
        for bot_id in bot_ids:
            other_occupied = occupied - {self._get_bot_pos(controller, bot_id)}
            self._execute_bot(controller, bot_id, other_occupied)
    
    def _get_bot_pos(self, controller: RobotController, bot_id: int) -> Optional[Tuple[int, int]]:
        """Get bot position"""
        state = controller.get_bot_state(bot_id)
        if state:
            return (state['x'], state['y'])
        return None
    
    def _execute_bot(self, controller: RobotController, bot_id: int, occupied: Set[Tuple[int, int]]):
        """Execute state machine for one bot"""
        task = self.bot_tasks[bot_id]
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        team = controller.get_team()
        money = controller.get_team_money(team)
        
        # Stuck detection
        if pos == task.last_pos:
            task.stuck_count += 1
        else:
            task.stuck_count = 0
        task.last_pos = pos
        
        # Stuck recovery
        if task.stuck_count > 30:
            self._handle_stuck(controller, bot_id, task, pos, holding, occupied)
            return
        
        # Check if order expired
        if task.order and task.order_id:
            orders = controller.get_orders(team)
            order_active = any(o['order_id'] == task.order_id and o['is_active'] for o in orders)
            if not order_active:
                self._cleanup_order(task)
                task.reset()
        
        # State machine dispatch
        state = task.state
        
        if state == BotState.IDLE:
            task.state = BotState.SELECT_ORDER
            task.idle_turns = 0
            return
        
        if state == BotState.SELECT_ORDER:
            task.idle_turns += 1
            self._select_order(controller, bot_id, task, pos, money)
            return
        
        if state == BotState.BUY_INGREDIENT:
            self._buy_ingredient(controller, bot_id, task, pos, holding, money, occupied)
            return
        
        if state == BotState.CHOP_PLACE:
            self._chop_place(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.CHOP_ACTION:
            self._chop_action(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.CHOP_PICKUP:
            self._chop_pickup(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.COOK_PLACE:
            self._cook_place(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.COOK_WAIT:
            self._cook_wait(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.COOK_TAKE:
            self._cook_take(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.STORE_FOOD:
            self._store_food(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.BUY_PLATE:
            self._buy_plate(controller, bot_id, task, pos, holding, money, occupied)
            return
        
        if state == BotState.PLACE_PLATE:
            self._place_plate(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.ADD_FOOD:
            self._add_food_to_plate(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.PICKUP_PLATE:
            self._pickup_plate(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.SUBMIT:
            self._submit_order(controller, bot_id, task, pos, holding, occupied)
            return
        
        if state == BotState.TRASH:
            self._trash_item(controller, bot_id, task, pos, holding, occupied)
            return
    
    # =========================================================================
    # ORDER SELECTION - Position-aware, prioritize simple orders
    # =========================================================================
    
    def _select_order(self, controller: RobotController, bot_id: int, task: BotTask,
                      pos: Tuple[int, int], money: int):
        """Select best order based on bot role and position"""
        team = controller.get_team()
        orders = controller.get_orders(team)
        
        # Find nearest facilities for this bot
        if not self.map_analyzer.shops or not self.map_analyzer.submits:
            return
            
        nearest_shop = min(self.map_analyzer.shops, 
                          key=lambda s: self.map_analyzer.get_distance_to_tile(pos, s))
        nearest_submit = min(self.map_analyzer.submits,
                            key=lambda s: self.map_analyzer.get_distance_to_tile(pos, s))
        nearest_cooker = min(self.map_analyzer.cookers,
                            key=lambda c: self.map_analyzer.get_distance_to_tile(pos, c)) if self.map_analyzer.cookers else None
        
        # Check bot capabilities using nearest facilities
        can_cook = self.map_analyzer.can_reach(pos, nearest_cooker) if nearest_cooker else False
        can_chop = self.map_analyzer.can_reach(pos, self.map_analyzer.counters[0]) if self.map_analyzer.counters else False
        can_submit = self.map_analyzer.can_reach(pos, nearest_submit)
        can_shop = self.map_analyzer.can_reach(pos, nearest_shop)
        
        if not can_shop or not can_submit:
            return
        
        # Get distances to nearest facilities
        dist_to_shop = self.map_analyzer.get_distance_to_tile(pos, nearest_shop)
        dist_to_cooker = self.map_analyzer.get_distance_to_tile(pos, nearest_cooker) if nearest_cooker else 999
        dist_to_submit = self.map_analyzer.get_distance_to_tile(pos, nearest_submit)
        
        # Skip order selection if bot is too far from shop (inefficient)
        if dist_to_shop > 20:
            return
        
        # Determine this bot's role
        is_cook_bot = (bot_id == self.cook_bot)
        is_runner_bot = (bot_id == self.runner_bot)
        
        best_order = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active', False):
                continue
            if order['order_id'] in self.assigned_orders:
                continue
            
            required = order['required']
            needs_cook = any(FOOD_INFO.get(f, {}).get('cook', False) for f in required)
            needs_chop = any(FOOD_INFO.get(f, {}).get('chop', False) for f in required)
            
            # Role-based filtering:
            # - On split maps: no filtering (each bot works independently)
            # - If idle too long: take any order
            # - Otherwise: runner bot prefers non-cooking orders
            runner_desperate = task.idle_turns > 15
            if (is_runner_bot and needs_cook and not runner_desperate 
                and not self.is_split_map):
                continue  # Runner bot skips cooking orders unless desperate
            
            if needs_cook and not can_cook:
                continue
            
            # Calculate cost
            cost = 2
            for food in required:
                cost += FOOD_INFO.get(food, {}).get('cost', 100)
            
            if cost > money + 10:
                continue
            
            # Estimate time - account for per-item overhead
            n_items = len(required)
            base_time = 15 + n_items * 5  # Minimum 5 turns per item for handling
            travel_time = dist_to_shop + dist_to_submit
            
            for food in required:
                info = FOOD_INFO.get(food, {})
                base_time += info.get('base_time', 10)
                if info.get('cook', False):
                    travel_time += dist_to_cooker * 2
            
            time_est = base_time + travel_time
            
            time_left = order['expires_turn'] - self.turn
            
            # Time feasibility check - be more aggressive for simple orders
            n_cook = sum(1 for f in required if FOOD_INFO.get(f, {}).get('cook', False))
            if n_cook == 0:
                # Simple orders (no cooking) - be more aggressive
                if time_est > time_left * 0.9:
                    continue
            else:
                # Cooking orders need more buffer due to variability
                if time_left < 60:
                    if time_est > time_left * 0.7:
                        continue
                else:
                    if time_est > time_left - 20:
                        continue
            
            # Score based on simplicity
            complexity = len(required)
            cook_count = sum(1 for f in required if FOOD_INFO.get(f, {}).get('cook', False))
            chop_count = sum(1 for f in required if FOOD_INFO.get(f, {}).get('chop', False))
            reward = order['reward']
            
            # Score formula: HEAVILY favor simple orders
            if len(required) == 1:
                if not needs_cook and not needs_chop:
                    score = 1000  # SAUCE - instant
                elif not needs_cook:
                    score = 800   # ONIONS - fast
                else:
                    score = 200   # Single cook item
            elif len(required) == 2 and not needs_cook:
                score = 600       # 2 no-cook
            elif len(required) == 2:
                score = 150       # 2 with cook
            elif len(required) == 3 and not needs_cook:
                score = 400       # 3 no-cook
            else:
                score = reward / max((complexity + cook_count * 3) * 10 + time_est, 1)
            
            # Role-based bonuses (only on non-split maps)
            if not self.is_split_map:
                # Cook bot gets bonus for cooking orders
                if is_cook_bot and needs_cook:
                    score += 50
                
                # Runner bot gets bonus for non-cooking orders
                if is_runner_bot and not needs_cook:
                    score += 50
            
            # Urgency bonus
            if time_left < 100:
                score += 20
            
            if score > best_score:
                best_score = score
                best_order = order
        
        if best_order:
            task.order = best_order
            task.order_id = best_order['order_id']
            task.ingredients_queue = self._order_ingredients(best_order['required'])
            task.ingredients_done = []
            task.current_ingredient = None
            task.food_locations = {}
            task.plate_counter = None
            task.work_counter = None
            task.cooker_pos = None
            self.assigned_orders.add(best_order['order_id'])
            task.state = BotState.BUY_INGREDIENT
    
    def _order_ingredients(self, required: List[str]) -> List[str]:
        """Order ingredients - put cooking items LAST so food cooks while we prep"""
        simple = []
        chop_only = []
        cook_items = []
        
        for food in required:
            info = FOOD_INFO.get(food, {})
            if info.get('cook', False):
                cook_items.append(food)
            elif info.get('chop', False):
                chop_only.append(food)
            else:
                simple.append(food)
        
        # Simple first (fastest), then chop, then cook last
        return simple + chop_only + cook_items
    
    # =========================================================================
    # INGREDIENT PROCESSING
    # =========================================================================
    
    def _buy_ingredient(self, controller: RobotController, bot_id: int, task: BotTask,
                        pos: Tuple[int, int], holding: Any, money: int, occupied: Set):
        """Buy the next ingredient"""
        # If done with all ingredients, go to plating
        if not task.ingredients_queue:
            task.state = BotState.BUY_PLATE
            return
        
        # Handle held items
        if holding:
            h_type = holding.get('type') if isinstance(holding, dict) else None
            if h_type == 'Food':
                food_name = holding.get('food_name')
                task.current_ingredient = food_name
                info = FOOD_INFO.get(food_name, {})
                task.need_chop = info.get('chop', False) and not holding.get('chopped', False)
                task.need_cook = info.get('cook', False) and holding.get('cooked_stage', 0) == 0
                
                if task.need_chop:
                    task.state = BotState.CHOP_PLACE
                elif task.need_cook:
                    task.state = BotState.COOK_PLACE
                else:
                    task.state = BotState.STORE_FOOD
                return
            elif h_type == 'Plate':
                task.state = BotState.PLACE_PLATE
                return
            else:
                task.state = BotState.TRASH
                return
        
        # Get next ingredient
        task.current_ingredient = task.ingredients_queue[0]
        info = FOOD_INFO.get(task.current_ingredient, {})
        food_type = info.get('type')
        cost = info.get('cost', 100)
        
        if money < cost:
            return  # Wait for money
        
        # Move to shop and buy
        shop = self.map_analyzer.get_nearest(pos, self.map_analyzer.shops)
        if not shop:
            return
        
        if self._move_to_tile(controller, bot_id, pos, shop, occupied):
            if controller.buy(bot_id, food_type, shop[0], shop[1]):
                task.need_chop = info.get('chop', False)
                task.need_cook = info.get('cook', False)
                
                if task.need_chop:
                    task.state = BotState.CHOP_PLACE
                elif task.need_cook:
                    task.state = BotState.COOK_PLACE
                else:
                    task.state = BotState.STORE_FOOD
    
    def _chop_place(self, controller: RobotController, bot_id: int, task: BotTask,
                    pos: Tuple[int, int], holding: Any, occupied: Set):
        """Place food on counter for chopping"""
        if not holding or holding.get('type') != 'Food':
            task.state = BotState.BUY_INGREDIENT
            return
        
        # Find nearest free counter
        counter = self._get_free_counter(controller, pos, exclude={task.plate_counter})
        if not counter:
            return
        
        if self._move_to_tile(controller, bot_id, pos, counter, occupied):
            if controller.place(bot_id, counter[0], counter[1]):
                task.work_counter = counter
                self.allocated_counters.add(counter)
                task.state = BotState.CHOP_ACTION
    
    def _chop_action(self, controller: RobotController, bot_id: int, task: BotTask,
                     pos: Tuple[int, int], holding: Any, occupied: Set):
        """Chop food on counter"""
        if not task.work_counter:
            task.state = BotState.BUY_INGREDIENT
            return
        
        team = controller.get_team()
        tile = controller.get_tile(team, task.work_counter[0], task.work_counter[1])
        if not tile:
            task.state = BotState.BUY_INGREDIENT
            return
        
        item = getattr(tile, 'item', None)
        if not isinstance(item, Food):
            task.state = BotState.BUY_INGREDIENT
            return
        
        if item.chopped:
            task.state = BotState.CHOP_PICKUP
            return
        
        if self._move_to_tile(controller, bot_id, pos, task.work_counter, occupied):
            controller.chop(bot_id, task.work_counter[0], task.work_counter[1])
    
    def _chop_pickup(self, controller: RobotController, bot_id: int, task: BotTask,
                     pos: Tuple[int, int], holding: Any, occupied: Set):
        """Pick up chopped food"""
        if holding:
            if holding.get('type') == 'Food' and holding.get('chopped', False):
                self.allocated_counters.discard(task.work_counter)
                task.work_counter = None
                if task.need_cook:
                    task.state = BotState.COOK_PLACE
                else:
                    task.state = BotState.STORE_FOOD
                return
            else:
                task.state = BotState.TRASH
                return
        
        if not task.work_counter:
            task.state = BotState.BUY_INGREDIENT
            return
        
        if self._move_to_tile(controller, bot_id, pos, task.work_counter, occupied):
            if controller.pickup(bot_id, task.work_counter[0], task.work_counter[1]):
                self.allocated_counters.discard(task.work_counter)
                task.work_counter = None
                if task.need_cook:
                    task.state = BotState.COOK_PLACE
                else:
                    task.state = BotState.STORE_FOOD
    
    def _cook_place(self, controller: RobotController, bot_id: int, task: BotTask,
                    pos: Tuple[int, int], holding: Any, occupied: Set):
        """Place food in cooker pan"""
        if not holding or holding.get('type') != 'Food':
            task.state = BotState.BUY_INGREDIENT
            return
        
        team = controller.get_team()
        cooker = self._get_available_cooker(controller, team, pos)
        if not cooker:
            return  # Wait for cooker
        
        if self._move_to_tile(controller, bot_id, pos, cooker, occupied):
            if controller.place(bot_id, cooker[0], cooker[1]):
                task.cooker_pos = cooker
                self.allocated_cookers.add(cooker)
                task.state = BotState.COOK_WAIT
    
    def _cook_wait(self, controller: RobotController, bot_id: int, task: BotTask,
                   pos: Tuple[int, int], holding: Any, occupied: Set):
        """Wait for food to cook"""
        if not task.cooker_pos:
            task.state = BotState.BUY_INGREDIENT
            return
        
        team = controller.get_team()
        tile = controller.get_tile(team, task.cooker_pos[0], task.cooker_pos[1])
        if not tile:
            task.state = BotState.BUY_INGREDIENT
            return
        
        pan = getattr(tile, 'item', None)
        if not isinstance(pan, Pan) or not pan.food:
            self.allocated_cookers.discard(task.cooker_pos)
            task.cooker_pos = None
            task.state = BotState.BUY_INGREDIENT
            return
        
        if pan.food.cooked_stage == 1:
            task.state = BotState.COOK_TAKE
        elif pan.food.cooked_stage == 2:
            task.state = BotState.COOK_TAKE  # Take and trash
    
    def _cook_take(self, controller: RobotController, bot_id: int, task: BotTask,
                   pos: Tuple[int, int], holding: Any, occupied: Set):
        """Take food from pan"""
        if holding:
            h_type = holding.get('type') if isinstance(holding, dict) else None
            if h_type == 'Food':
                cooked_stage = holding.get('cooked_stage', 0)
                self.allocated_cookers.discard(task.cooker_pos)
                task.cooker_pos = None
                
                if cooked_stage == 2:
                    task.state = BotState.TRASH
                else:
                    task.state = BotState.STORE_FOOD
                return
            else:
                task.state = BotState.TRASH
                return
        
        if not task.cooker_pos:
            task.state = BotState.BUY_INGREDIENT
            return
        
        if self._move_to_tile(controller, bot_id, pos, task.cooker_pos, occupied):
            if controller.take_from_pan(bot_id, task.cooker_pos[0], task.cooker_pos[1]):
                self.allocated_cookers.discard(task.cooker_pos)
                task.cooker_pos = None
    
    def _store_food(self, controller: RobotController, bot_id: int, task: BotTask,
                    pos: Tuple[int, int], holding: Any, occupied: Set):
        """Store processed food on counter"""
        if not holding or holding.get('type') != 'Food':
            task.state = BotState.BUY_INGREDIENT
            return
        
        if holding.get('cooked_stage', 0) == 2:
            task.state = BotState.TRASH
            return
        
        # Find free counter near submit (for faster plating later)
        counter = self._get_free_counter(controller, pos, 
                                         exclude={task.plate_counter},
                                         prefer_near=self.map_analyzer.submits[0] if self.map_analyzer.submits else None)
        if not counter:
            return
        
        if self._move_to_tile(controller, bot_id, pos, counter, occupied):
            if controller.place(bot_id, counter[0], counter[1]):
                food_name = holding.get('food_name')
                task.food_locations[food_name] = counter
                self.allocated_counters.add(counter)
                
                if food_name in task.ingredients_queue:
                    task.ingredients_queue.remove(food_name)
                    task.ingredients_done.append(food_name)
                
                task.current_ingredient = None
                task.need_chop = False
                task.need_cook = False
                
                if task.ingredients_queue:
                    task.state = BotState.BUY_INGREDIENT
                else:
                    task.state = BotState.BUY_PLATE
    
    # =========================================================================
    # PLATING
    # =========================================================================
    
    def _buy_plate(self, controller: RobotController, bot_id: int, task: BotTask,
                   pos: Tuple[int, int], holding: Any, money: int, occupied: Set):
        """Get a plate"""
        if holding:
            h_type = holding.get('type') if isinstance(holding, dict) else None
            if h_type == 'Plate' and not holding.get('dirty', True):
                task.state = BotState.PLACE_PLATE
                return
            else:
                task.state = BotState.TRASH
                return
        
        team = controller.get_team()
        
        # Try sink table first (free plates)
        if self.map_analyzer.sink_tables:
            for st in self.map_analyzer.sink_tables:
                tile = controller.get_tile(team, st[0], st[1])
                if tile and hasattr(tile, 'num_clean_plates') and tile.num_clean_plates > 0:
                    if self._move_to_tile(controller, bot_id, pos, st, occupied):
                        controller.take_clean_plate(bot_id, st[0], st[1])
                    return
        
        # Buy from shop
        if money < 2:
            return
        
        shop = self.map_analyzer.get_nearest(pos, self.map_analyzer.shops)
        if shop and self._move_to_tile(controller, bot_id, pos, shop, occupied):
            controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
    
    def _place_plate(self, controller: RobotController, bot_id: int, task: BotTask,
                     pos: Tuple[int, int], holding: Any, occupied: Set):
        """Place plate on counter"""
        if not holding or holding.get('type') != 'Plate':
            task.state = BotState.BUY_PLATE
            return
        
        # Find counter near submit
        counter = self._get_free_counter(controller, pos, 
                                         exclude=set(task.food_locations.values()),
                                         prefer_near=self.map_analyzer.submits[0] if self.map_analyzer.submits else None)
        if not counter:
            return
        
        if self._move_to_tile(controller, bot_id, pos, counter, occupied):
            if controller.place(bot_id, counter[0], counter[1]):
                task.plate_counter = counter
                self.allocated_counters.add(counter)
                task.state = BotState.ADD_FOOD
    
    def _add_food_to_plate(self, controller: RobotController, bot_id: int, task: BotTask,
                           pos: Tuple[int, int], holding: Any, occupied: Set):
        """Add all food items to plate"""
        # If holding food, add it
        if holding and holding.get('type') == 'Food':
            if not task.plate_counter:
                task.state = BotState.BUY_PLATE
                return
            
            if self._move_to_tile(controller, bot_id, pos, task.plate_counter, occupied):
                controller.add_food_to_plate(bot_id, task.plate_counter[0], task.plate_counter[1])
            return
        
        # Check if plate is complete
        team = controller.get_team()
        if task.plate_counter:
            tile = controller.get_tile(team, task.plate_counter[0], task.plate_counter[1])
            if tile and isinstance(getattr(tile, 'item', None), Plate):
                plate = tile.item
                plated_foods = [f.food_name for f in plate.food if isinstance(f, Food)]
                
                if set(plated_foods) >= set(task.ingredients_done):
                    task.state = BotState.PICKUP_PLATE
                    return
        
        # Get next food to add
        if not task.food_locations:
            task.state = BotState.PICKUP_PLATE
            return
        
        food_name, food_loc = next(iter(task.food_locations.items()))
        
        if self._move_to_tile(controller, bot_id, pos, food_loc, occupied):
            if controller.pickup(bot_id, food_loc[0], food_loc[1]):
                del task.food_locations[food_name]
                self.allocated_counters.discard(food_loc)
    
    def _pickup_plate(self, controller: RobotController, bot_id: int, task: BotTask,
                      pos: Tuple[int, int], holding: Any, occupied: Set):
        """Pick up completed plate"""
        if holding and holding.get('type') == 'Plate':
            self.allocated_counters.discard(task.plate_counter)
            task.plate_counter = None
            task.state = BotState.SUBMIT
            return
        
        if holding:
            if holding.get('type') == 'Food':
                task.state = BotState.ADD_FOOD
                return
            else:
                task.state = BotState.TRASH
                return
        
        if not task.plate_counter:
            task.state = BotState.BUY_PLATE
            return
        
        if self._move_to_tile(controller, bot_id, pos, task.plate_counter, occupied):
            if controller.pickup(bot_id, task.plate_counter[0], task.plate_counter[1]):
                self.allocated_counters.discard(task.plate_counter)
                task.plate_counter = None
                task.state = BotState.SUBMIT
    
    def _submit_order(self, controller: RobotController, bot_id: int, task: BotTask,
                      pos: Tuple[int, int], holding: Any, occupied: Set):
        """Submit the completed order"""
        if not holding or holding.get('type') != 'Plate':
            task.state = BotState.BUY_PLATE
            return
        
        submit = self.map_analyzer.get_nearest(pos, self.map_analyzer.submits)
        if not submit:
            return
        
        if self._move_to_tile(controller, bot_id, pos, submit, occupied):
            if controller.submit(bot_id, submit[0], submit[1]):
                self._cleanup_order(task)
                task.reset()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _trash_item(self, controller: RobotController, bot_id: int, task: BotTask,
                    pos: Tuple[int, int], holding: Any, occupied: Set):
        """Trash held item"""
        if not holding:
            if task.current_ingredient:
                task.state = BotState.BUY_INGREDIENT
            else:
                task.state = BotState.IDLE
            return
        
        trash = self.map_analyzer.get_nearest(pos, self.map_analyzer.trashes)
        if not trash:
            return
        
        if self._move_to_tile(controller, bot_id, pos, trash, occupied):
            if controller.trash(bot_id, trash[0], trash[1]):
                if holding.get('type') == 'Food' and holding.get('cooked_stage', 0) == 2:
                    food_name = holding.get('food_name')
                    if food_name and food_name in task.ingredients_done:
                        task.ingredients_done.remove(food_name)
                        task.ingredients_queue.insert(0, food_name)
                
                task.state = BotState.BUY_INGREDIENT
    
    def _handle_stuck(self, controller: RobotController, bot_id: int, task: BotTask,
                      pos: Tuple[int, int], holding: Any, occupied: Set):
        """Handle stuck bot"""
        task.stuck_count = 0
        
        if holding:
            trash = self.map_analyzer.get_nearest(pos, self.map_analyzer.trashes)
            if trash and self._move_to_tile(controller, bot_id, pos, trash, occupied):
                controller.trash(bot_id, trash[0], trash[1])
            return
        
        self._cleanup_order(task)
        task.reset()
    
    def _cleanup_order(self, task: BotTask):
        """Clean up order state"""
        if task.order_id:
            self.assigned_orders.discard(task.order_id)
        if task.plate_counter:
            self.allocated_counters.discard(task.plate_counter)
        if task.work_counter:
            self.allocated_counters.discard(task.work_counter)
        if task.cooker_pos:
            self.allocated_cookers.discard(task.cooker_pos)
        for loc in task.food_locations.values():
            self.allocated_counters.discard(loc)
    
    def _move_to_tile(self, controller: RobotController, bot_id: int,
                      pos: Tuple[int, int], target: Tuple[int, int], 
                      occupied: Set) -> bool:
        """Move towards a target, returns True if adjacent"""
        px, py = pos
        tx, ty = target
        
        if max(abs(px - tx), abs(py - ty)) <= 1:
            return True
        
        step = self.map_analyzer.get_next_step(pos, target, occupied)
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
        
        return False
    
    def _get_free_counter(self, controller: RobotController, pos: Tuple[int, int],
                          exclude: Set[Tuple[int, int]] = None,
                          prefer_near: Tuple[int, int] = None) -> Optional[Tuple[int, int]]:
        """Find nearest free counter"""
        exclude = exclude or set()
        team = controller.get_team()
        available = []
        
        for c in self.map_analyzer.counters:
            if c in exclude:
                continue
            if c in self.allocated_counters:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is not None:
                continue
            available.append(c)
        
        if not available:
            return None
        
        if prefer_near:
            available.sort(key=lambda c: (
                self.map_analyzer.get_distance_to_tile(prefer_near, c),
                self.map_analyzer.get_distance_to_tile(pos, c)
            ))
        else:
            available.sort(key=lambda c: self.map_analyzer.get_distance_to_tile(pos, c))
        
        return available[0]
    
    def _get_available_cooker(self, controller: RobotController, team: Team,
                              pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find cooker with empty pan"""
        available = []
        
        for cooker in self.map_analyzer.cookers:
            if cooker in self.allocated_cookers:
                continue
            tile = controller.get_tile(team, cooker[0], cooker[1])
            if not tile:
                continue
            pan = getattr(tile, 'item', None)
            if isinstance(pan, Pan) and pan.food is None:
                available.append(cooker)
        
        if not available:
            return None
        
        available.sort(key=lambda c: self.map_analyzer.get_distance_to_tile(pos, c))
        return available[0]
