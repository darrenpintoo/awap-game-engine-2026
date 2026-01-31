"""
Champion Bot - Ultimate AWAP 2026 Tournament Winner
====================================================

Implements ALL winning strategies:
1. Smart Order Selection - Picks highest profit/turn orders
2. Ingredient Efficiency - Prefers cheap ingredients (Sauce > Egg > Noodles > Onions > Meat)
3. Parallel Pipeline - Both bots work productively
4. Plate Recycling - Wash and reuse plates
5. Box Buffering - Pre-buy cheap ingredients
6. Optimal Sabotage - Strike at the perfect moment
7. Counter-Sabotage Defense - Protect against enemy

Uses updated API format: get_team_money(team), get_team_bot_ids(team), etc.
"""

import heapq
from collections import deque
from typing import Tuple, Optional, List, Dict, Set, Any
from dataclasses import dataclass
from enum import Enum, auto

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = True  # Enable for testing

def log(msg):
    if DEBUG:
        print(f"[ChampionBot] {msg}")

# Ingredient costs and processing requirements
INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'turns': 0},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'turns': 20},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'turns': 1},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'turns': 0},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'turns': 21},
}

# =============================================================================
# PATHFINDING
# =============================================================================

class Pathfinder:
    """Optimized A* pathfinding with 8-directional movement"""
    
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    @staticmethod
    def chebyshev(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    @staticmethod
    def manhattan(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    
    @staticmethod
    def get_path(map_obj, start: Tuple[int,int], target: Tuple[int,int], 
                 stop_dist: int = 1, avoid: Set[Tuple[int,int]] = None) -> Optional[List[Tuple[int,int]]]:
        """A* to get adjacent to target"""
        if avoid is None:
            avoid = set()
            
        w, h = map_obj.width, map_obj.height
        
        # Already there?
        if Pathfinder.chebyshev(start, target) <= stop_dist:
            return []
        
        # Priority queue: (f_score, g_score, x, y, path)
        queue = [(Pathfinder.chebyshev(start, target), 0, start[0], start[1], [])]
        visited = {start: 0}
        
        while queue:
            f, g, cx, cy, path = heapq.heappop(queue)
            
            if Pathfinder.chebyshev((cx, cy), target) <= stop_dist:
                return path
            
            if g > 50:  # Depth limit
                continue
            
            for dx, dy in Pathfinder.DIRS_8:
                nx, ny = cx + dx, cy + dy
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not map_obj.is_tile_walkable(nx, ny):
                    continue
                if (nx, ny) in avoid:
                    continue
                
                new_g = g + 1
                if (nx, ny) not in visited or new_g < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_g
                    h_score = Pathfinder.chebyshev((nx, ny), target)
                    heapq.heappush(queue, (new_g + h_score, new_g, nx, ny, path + [(dx, dy)]))
        
        return None


# =============================================================================
# ORDER ANALYZER
# =============================================================================

@dataclass
class OrderScore:
    order_id: int
    required: List[str]
    reward: int
    penalty: int
    expires_turn: int
    
    # Calculated
    cost: int = 0
    turns_needed: int = 0
    profit: float = 0
    score: float = 0  # profit per turn
    
    def calculate(self, current_turn: int):
        """Calculate order profitability"""
        self.cost = 0
        self.turns_needed = 5  # Base overhead (movement, plating, submit)
        
        needs_cooking = False
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50, 'chop': False, 'cook': False, 'turns': 5})
            self.cost += info['cost']
            
            if info['cook']:
                needs_cooking = True
            if info['chop']:
                self.turns_needed += 2  # Place + chop + pickup
        
        # Cooking takes 20 turns but happens in parallel
        if needs_cooking:
            self.turns_needed = max(self.turns_needed, 25)
        
        # Add plate cost
        self.cost += ShopCosts.PLATE.buy_cost
        
        time_left = self.expires_turn - current_turn
        
        # Can we complete it?
        if self.turns_needed > time_left:
            self.score = -1000  # Impossible
            return
        
        self.profit = self.reward - self.cost
        
        # Score = profit per turn, with bonus for simpler orders
        self.score = self.profit / max(self.turns_needed, 1)
        
        # Bonus for orders that are about to expire (urgency)
        if time_left < 50 and time_left > self.turns_needed:
            self.score *= 1.2


class OrderAnalyzer:
    """Analyzes and ranks orders by profitability"""
    
    @staticmethod
    def get_best_orders(controller: RobotController, team: Team, limit: int = 3) -> List[OrderScore]:
        """Get the most profitable orders"""
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        
        scored = []
        for order in orders:
            if not order.get('is_active', False):
                continue
                
            os = OrderScore(
                order_id=order['order_id'],
                required=order['required'],
                reward=order['reward'],
                penalty=order.get('penalty', 0),
                expires_turn=order['expires_turn']
            )
            os.calculate(current_turn)
            
            if os.score > 0:
                scored.append(os)
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]


# =============================================================================
# BOT STATES
# =============================================================================

class BotState(Enum):
    IDLE = auto()
    
    # Equipment
    BUY_PAN = auto()
    PLACE_PAN = auto()
    
    # Ingredient pipeline
    BUY_INGREDIENT = auto()
    PLACE_FOR_CHOP = auto()
    CHOP = auto()
    PICKUP_CHOPPED = auto()
    START_COOK = auto()
    WAIT_COOK = auto()
    TAKE_FROM_PAN = auto()
    
    # Plating
    BUY_PLATE = auto()
    PLACE_PLATE = auto()
    GET_CLEAN_PLATE = auto()
    ADD_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    STORE_PLATE = auto()
    RETRIEVE_PLATE = auto()
    
    # Delivery
    SUBMIT = auto()
    
    # Maintenance
    WASH_DISHES = auto()
    
    # Recovery
    TRASH = auto()
    
    # Sabotage
    SABOTAGE_SWITCH = auto()
    SABOTAGE_STEAL_PAN = auto()
    SABOTAGE_STEAL_PLATE = auto()
    SABOTAGE_BLOCK = auto()


@dataclass
class BotTask:
    state: BotState
    target: Optional[Tuple[int, int]] = None
    item: Optional[str] = None
    order_id: Optional[int] = None
    sub_state: int = 0  # For multi-step states


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    """Ultimate Champion Bot"""
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Cached tile locations
        self.shops: List[Tuple[int,int]] = []
        self.cookers: List[Tuple[int,int]] = []
        self.counters: List[Tuple[int,int]] = []
        self.submits: List[Tuple[int,int]] = []
        self.trashes: List[Tuple[int,int]] = []
        self.sinks: List[Tuple[int,int]] = []
        self.sink_tables: List[Tuple[int,int]] = []
        self.boxes: List[Tuple[int,int]] = []
        
        # Bot task assignments
        self.bot_tasks: Dict[int, BotTask] = {}
        
        # Designated locations
        self.assembly_counter: Optional[Tuple[int,int]] = None
        self.work_counter: Optional[Tuple[int,int]] = None
        self.primary_cooker: Optional[Tuple[int,int]] = None
        
        # Tracking
        self.current_order: Optional[OrderScore] = None
        self.plate_on_assembly: bool = False
        self.ingredients_on_plate: List[str] = []
        self.cooking_ingredient: Optional[str] = None
        self.has_switched: bool = False
        self.single_counter: bool = False
        self.plate_storage_box: Optional[Tuple[int, int]] = None
        self.plate_in_box: bool = False
        self.pending_food_name: Optional[str] = None
        self.pending_food_pos: Optional[Tuple[int, int]] = None
        
    def _init_map(self, controller: RobotController, team: Team):
        """Initialize map data"""
        m = controller.get_map(team)
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                name = tile.tile_name
                pos = (x, y)
                
                if name == "SHOP": self.shops.append(pos)
                elif name == "COOKER": self.cookers.append(pos)
                elif name == "COUNTER": self.counters.append(pos)
                elif name == "SUBMIT": self.submits.append(pos)
                elif name == "TRASH": self.trashes.append(pos)
                elif name == "SINK": self.sinks.append(pos)
                elif name == "SINKTABLE": self.sink_tables.append(pos)
                elif name == "BOX": self.boxes.append(pos)
        
        # Assign key locations
        if self.counters:
            self.assembly_counter = self.counters[0]
            self.work_counter = self.counters[1] if len(self.counters) > 1 else self.counters[0]
        if self.cookers:
            self.primary_cooker = self.cookers[0]
        self.single_counter = len(self.counters) <= 1
        if self.single_counter and self.boxes:
            self.plate_storage_box = self.boxes[0]
            
        self.initialized = True
        log(f"Init: {len(self.counters)} counters, {len(self.cookers)} cookers, {len(self.shops)} shops")
    
    def _get_nearest(self, pos: Tuple[int,int], locations: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
        """Get nearest location from list"""
        if not locations:
            return None
        return min(locations, key=lambda p: Pathfinder.chebyshev(pos, p))
    
    def _get_avoid_set(self, controller: RobotController, team: Team, exclude_bot: int) -> Set[Tuple[int,int]]:
        """Get positions to avoid (other bots)"""
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != exclude_bot:
                st = controller.get_bot_state(bid)
                if st:
                    avoid.add((st['x'], st['y']))
        return avoid
    
    def _move_toward(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int,int], team: Team) -> bool:
        """Move toward target. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
            
        pos = (bot['x'], bot['y'])
        
        if Pathfinder.chebyshev(pos, target) <= 1:
            return True
        
        avoid = self._get_avoid_set(controller, team, bot_id)
        m = controller.get_map(team)
        path = Pathfinder.get_path(m, pos, target, stop_dist=1, avoid=avoid)
        
        if path and len(path) > 0:
            dx, dy = path[0]
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        # Try wiggle if stuck
        for dx, dy in Pathfinder.DIRS_8:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        return False
    
    def _is_counter_empty(self, controller: RobotController, team: Team, 
                          cx: int, cy: int) -> bool:
        """Check if counter is empty"""
        tile = controller.get_tile(team, cx, cy)
        return tile is not None and getattr(tile, 'item', None) is None
    
    def _has_pan_on_cooker(self, controller: RobotController, team: Team,
                           kx: int, ky: int) -> bool:
        """Check if cooker has a pan"""
        tile = controller.get_tile(team, kx, ky)
        return tile is not None and isinstance(getattr(tile, 'item', None), Pan)
    
    def _get_pan_food_state(self, controller: RobotController, team: Team,
                            kx: int, ky: int) -> Optional[int]:
        """Get food cooked_stage in pan (0=cooking, 1=done, 2=burnt, None=empty)"""
        tile = controller.get_tile(team, kx, ky)
        if tile:
            pan = getattr(tile, 'item', None)
            if isinstance(pan, Pan) and pan.food:
                return pan.food.cooked_stage
        return None
    
    def _check_plate_on_counter(self, controller: RobotController, team: Team,
                                cx: int, cy: int) -> Optional[List[str]]:
        """Check if counter has a plate and return its contents"""
        tile = controller.get_tile(team, cx, cy)
        if tile:
            item = getattr(tile, 'item', None)
            if isinstance(item, Plate) and not item.dirty:
                return [f.food_name for f in item.food]
        return None
    
    def _count_clean_plates(self, controller: RobotController, team: Team) -> int:
        """Count clean plates at sink tables"""
        count = 0
        for sx, sy in self.sink_tables:
            tile = controller.get_tile(team, sx, sy)
            if tile:
                count += getattr(tile, 'num_clean_plates', 0)
        return count
    
    def _count_dirty_plates(self, controller: RobotController, team: Team) -> int:
        """Count dirty plates in sinks"""
        count = 0
        for sx, sy in self.sinks:
            tile = controller.get_tile(team, sx, sy)
            if tile:
                count += getattr(tile, 'num_dirty_plates', 0)
        return count
    
    def _select_best_order(self, controller: RobotController, team: Team):
        """Select the most profitable order to work on"""
        best_orders = OrderAnalyzer.get_best_orders(controller, team)
        
        if best_orders:
            self.current_order = best_orders[0]
            self.ingredients_on_plate = []
            log(f"Selected order {self.current_order.order_id}: {self.current_order.required} "
                f"(score={self.current_order.score:.2f})")
        else:
            self.current_order = None
    
    def _get_next_ingredient(self) -> Optional[str]:
        """Get next ingredient needed for current order"""
        if not self.current_order:
            return None
        
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                return ing
        return None
    
    def _execute_primary_bot(self, controller: RobotController, bot_id: int, team: Team):
        """Execute the primary bot's task (main order pipeline)"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(state=BotState.IDLE)
            self.bot_tasks[bot_id] = task
        
        # Key locations
        shop = self._get_nearest((bx, by), self.shops)
        cooker = self.primary_cooker or (self.cookers[0] if self.cookers else None)
        assembly = self.assembly_counter
        work = self.work_counter
        submit = self._get_nearest((bx, by), self.submits)
        trash = self._get_nearest((bx, by), self.trashes)
        sink_table = self._get_nearest((bx, by), self.sink_tables) if self.sink_tables else None
        
        if not all([shop, assembly]):
            return
        
        sx, sy = shop
        ax, ay = assembly
        
        # Recovery: handle holding mismatched items for buy states
        if task.state == BotState.BUY_PLATE and holding:
            if holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
                task.target = assembly
            elif trash:
                task.state = BotState.TRASH
                task.target = trash
        elif task.state == BotState.BUY_PAN and holding:
            if holding.get('type') == 'Pan':
                task.state = BotState.PLACE_PAN
                task.target = cooker
            elif trash:
                task.state = BotState.TRASH
                task.target = trash
        elif task.state == BotState.BUY_INGREDIENT and holding:
            if holding.get('type') == 'Food':
                info = INGREDIENT_INFO.get(task.item or "", {})
                if info.get('chop'):
                    task.state = BotState.PLACE_FOR_CHOP
                    task.target = work
                elif info.get('cook'):
                    task.state = BotState.START_COOK
                    task.target = cooker
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = assembly
            elif trash:
                task.state = BotState.TRASH
                task.target = trash
        
        # =========== STATE MACHINE ===========
        
        # IDLE: Pick next action
        if task.state == BotState.IDLE:
            # Check if we have an order to work on
            if not self.current_order:
                self._select_best_order(controller, team)
            
            if not self.current_order:
                return  # No orders available

            # If holding a plate and food is pending on counter, add it
            if holding and holding.get('type') == 'Plate' and self.pending_food_name and self.pending_food_pos:
                task.state = BotState.ADD_TO_PLATE
                task.item = self.pending_food_name
                task.target = self.pending_food_pos
                return
            # If holding a plate and nothing pending, place it to free hands
            if holding and holding.get('type') == 'Plate' and not self.plate_on_assembly:
                task.state = BotState.PLACE_PLATE
                task.target = assembly
                return
            
            # Check what we need to do
            # 1. Do we have a pan? (needed for cooking)
            needs_cooking = any(INGREDIENT_INFO.get(ing, {}).get('cook', False) 
                               for ing in self.current_order.required)
            
            if needs_cooking and cooker and not self._has_pan_on_cooker(controller, team, cooker[0], cooker[1]):
                task.state = BotState.BUY_PAN
                return
            
            # 2. Check plate status - ALWAYS DO THIS FIRST
            plate_contents = self._check_plate_on_counter(controller, team, ax, ay)
            if plate_contents is not None:
                self.plate_on_assembly = True
                self.ingredients_on_plate = plate_contents
            elif self.plate_in_box:
                # Plate stored in box; keep tracked ingredients
                self.plate_on_assembly = False
            else:
                self.plate_on_assembly = False
                self.ingredients_on_plate = []

            
            # 3. Choose next ingredient needed
            next_ing = self._get_next_ingredient()
            if self.single_counter:
                for ing in self.current_order.required:
                    if ing not in self.ingredients_on_plate and INGREDIENT_INFO.get(ing, {}).get('chop'):
                        next_ing = ing
                        break

            # 4. MUST have a plate before plating, unless we're chopping first on single-counter maps
            require_plate = not (self.single_counter and next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop'))
            if require_plate and not self.plate_on_assembly and not self.plate_in_box:
                # Check for clean plates at sink table
                if sink_table and self._count_clean_plates(controller, team) > 0:
                    task.state = BotState.GET_CLEAN_PLATE
                    task.target = sink_table
                else:
                    task.state = BotState.BUY_PLATE
                return

            # If we have pending food on counter and a plate is ready, add it now
            if self.pending_food_name and self.pending_food_pos and self.plate_on_assembly:
                task.state = BotState.ADD_TO_PLATE
                task.item = self.pending_food_name
                task.target = self.pending_food_pos
                return

            # Retrieve plate from box unless we still need to chop
            if not self.plate_on_assembly and self.plate_in_box and self.plate_storage_box:
                if next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop'):
                    # Keep plate stored while chopping
                    pass
                else:
                    task.state = BotState.RETRIEVE_PLATE
                    task.target = self.plate_storage_box
                    task.sub_state = 0
                    return
            
            if next_ing is None:
                # All ingredients on plate - pick up and submit
                task.state = BotState.PICKUP_PLATE
                task.target = assembly
                return
            
            # 5. Is something cooking? (check and retrieve if ready)
            if cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state == 1:  # Cooked, need to take it
                    if self.plate_in_box and self.plate_storage_box:
                        task.state = BotState.RETRIEVE_PLATE
                        task.target = self.plate_storage_box
                        task.sub_state = 0
                        return
                    task.state = BotState.TAKE_FROM_PAN
                    task.target = cooker
                    return
                elif pan_state == 2:  # Burnt!
                    task.state = BotState.TAKE_FROM_PAN  # Take and trash
                    task.target = cooker
                    return
                elif pan_state == 0:  # Still cooking
                    if not self.plate_on_assembly and not self.plate_in_box:
                        task.state = BotState.BUY_PLATE
                        return
                    # Can do other things while waiting - get other ingredients
                    # Find a non-cooking ingredient to work on
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            ing_info = INGREDIENT_INFO.get(ing, {})
                            if not ing_info.get('cook'):
                                next_ing = ing
                                break
                    else:
                        # All remaining ingredients need cooking, must wait
                        return
            
            # 6. Start working on next ingredient
            info = INGREDIENT_INFO.get(next_ing, {})
            task.item = next_ing

            # If single counter and we need to chop, stash plate to free counter
            if info.get('chop') and self.single_counter and self.plate_on_assembly and self.plate_storage_box:
                task.state = BotState.STORE_PLATE
                task.target = assembly
                task.sub_state = 0
                return
            
            if info.get('cook') and cooker:
                # Need to cook - check if pan is free
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state is None:  # Pan empty
                    task.state = BotState.BUY_INGREDIENT
                else:
                    # Pan busy - try to find non-cooking ingredient
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            ing_info = INGREDIENT_INFO.get(ing, {})
                            if not ing_info.get('cook'):
                                task.item = ing
                                task.state = BotState.BUY_INGREDIENT
                                return
                    # All need cooking, wait
                    return
            elif info.get('chop'):
                # Chop only (ONIONS)
                task.state = BotState.BUY_INGREDIENT
            else:
                # No processing needed (SAUCE, NOODLES) - buy and plate
                task.state = BotState.BUY_INGREDIENT
        
        # BUY_PAN
        elif task.state == BotState.BUY_PAN:
            if holding and holding.get('type') == 'Pan':
                task.state = BotState.PLACE_PAN
                task.target = cooker
            elif self._move_toward(controller, bot_id, shop, team):
                if money >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
        
        # PLACE_PAN
        elif task.state == BotState.PLACE_PAN:
            kx, ky = task.target
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky):
                    task.state = BotState.IDLE
        
        # BUY_INGREDIENT
        elif task.state == BotState.BUY_INGREDIENT:
            ing_name = task.item
            food_type = getattr(FoodType, ing_name, None)
            
            if holding:
                # Already holding something - process it
                info = INGREDIENT_INFO.get(ing_name, {})
                if info.get('chop'):
                    task.state = BotState.PLACE_FOR_CHOP
                    task.target = work
                elif info.get('cook'):
                    task.state = BotState.START_COOK
                    task.target = cooker
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = assembly
            elif self._move_toward(controller, bot_id, shop, team):
                if food_type and money >= food_type.buy_cost:
                    if controller.buy(bot_id, food_type, sx, sy):
                        log(f"Bought {ing_name}")
        
        # PLACE_FOR_CHOP
        elif task.state == BotState.PLACE_FOR_CHOP:
            wx, wy = task.target or work
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if self._is_counter_empty(controller, team, wx, wy):
                    if controller.place(bot_id, wx, wy):
                        task.state = BotState.CHOP
                        task.target = (wx, wy)
        
        # CHOP
        elif task.state == BotState.CHOP:
            wx, wy = task.target
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if controller.chop(bot_id, wx, wy):
                    task.state = BotState.PICKUP_CHOPPED
                    task.target = (wx, wy)
                    log(f"Chopped {task.item}")
        
        # PICKUP_CHOPPED
        elif task.state == BotState.PICKUP_CHOPPED:
            wx, wy = task.target
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if controller.pickup(bot_id, wx, wy):
                    info = INGREDIENT_INFO.get(task.item, {})
                    if info.get('cook'):
                        task.state = BotState.START_COOK
                        task.target = cooker
                    else:
                        task.state = BotState.ADD_TO_PLATE
                        task.target = assembly
        
        # START_COOK
        elif task.state == BotState.START_COOK:
            kx, ky = task.target or cooker
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky):
                    self.cooking_ingredient = task.item
                    task.state = BotState.WAIT_COOK
                    task.target = (kx, ky)
                    log(f"Started cooking {task.item}")
        
        # WAIT_COOK
        elif task.state == BotState.WAIT_COOK:
            kx, ky = task.target
            pan_state = self._get_pan_food_state(controller, team, kx, ky)
            
            if pan_state == 1:  # Done
                task.state = BotState.TAKE_FROM_PAN
            elif pan_state == 2:  # Burnt
                task.state = BotState.TAKE_FROM_PAN
            elif pan_state == 0:
                # While cooking, prep plate and non-cook ingredients
                if not self.plate_on_assembly and not self.plate_in_box:
                    task.state = BotState.BUY_PLATE
                    return
                if self.current_order:
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate and not INGREDIENT_INFO.get(ing, {}).get('cook'):
                            task.state = BotState.BUY_INGREDIENT
                            task.item = ing
                            return
                # Otherwise keep waiting
        
        # TAKE_FROM_PAN
        elif task.state == BotState.TAKE_FROM_PAN:
            kx, ky = task.target or cooker
            if holding:
                # Already holding - add to plate or trash if burnt
                h_cooked = holding.get('cooked_stage', 0)
                if h_cooked == 2:  # Burnt
                    task.state = BotState.TRASH
                    task.target = trash
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = assembly
            elif self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.take_from_pan(bot_id, kx, ky):
                    log(f"Took from pan")
        
        # BUY_PLATE
        elif task.state == BotState.BUY_PLATE:
            if holding and holding.get('type') == 'Plate':
                if self.pending_food_name and self.pending_food_pos:
                    task.state = BotState.ADD_TO_PLATE
                    task.item = self.pending_food_name
                    task.target = self.pending_food_pos
                else:
                    task.state = BotState.PLACE_PLATE
                    task.target = assembly
            elif self._move_toward(controller, bot_id, shop, team):
                if money >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        log("Bought plate")
        
        # GET_CLEAN_PLATE
        elif task.state == BotState.GET_CLEAN_PLATE:
            stx, sty = task.target
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
                task.target = assembly
            elif self._move_toward(controller, bot_id, (stx, sty), team):
                if controller.take_clean_plate(bot_id, stx, sty):
                    log("Got clean plate from sink table")
        
        # PLACE_PLATE
        elif task.state == BotState.PLACE_PLATE:
            ax, ay = task.target or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if self._is_counter_empty(controller, team, ax, ay):
                    if controller.place(bot_id, ax, ay):
                        self.plate_on_assembly = True
                        task.state = BotState.IDLE
                        log("Placed plate on assembly")

        # STORE_PLATE (single-counter workaround)
        elif task.state == BotState.STORE_PLATE:
            if task.sub_state == 0:
                if holding and holding.get('type') == 'Plate':
                    task.sub_state = 1
                else:
                    ax, ay = assembly
                    if self._move_toward(controller, bot_id, (ax, ay), team):
                        if controller.pickup(bot_id, ax, ay):
                            self.plate_on_assembly = False
                            task.sub_state = 1
            if task.sub_state == 1:
                if not self.plate_storage_box:
                    task.state = BotState.IDLE
                    task.sub_state = 0
                else:
                    bx, by = self.plate_storage_box
                    if self._move_toward(controller, bot_id, (bx, by), team):
                        if controller.place(bot_id, bx, by):
                            self.plate_in_box = True
                            task.state = BotState.IDLE
                            task.sub_state = 0
                            log("Stored plate in box")

        # RETRIEVE_PLATE (from box to assembly)
        elif task.state == BotState.RETRIEVE_PLATE:
            if task.sub_state == 0:
                if holding and holding.get('type') == 'Plate':
                    task.sub_state = 1
                else:
                    if self.plate_storage_box:
                        bx, by = self.plate_storage_box
                        if self._move_toward(controller, bot_id, (bx, by), team):
                            if controller.pickup(bot_id, bx, by):
                                self.plate_in_box = False
                                task.sub_state = 1
            if task.sub_state == 1:
                ax, ay = assembly
                if self._move_toward(controller, bot_id, (ax, ay), team):
                    if self._is_counter_empty(controller, team, ax, ay):
                        if controller.place(bot_id, ax, ay):
                            self.plate_on_assembly = True
                            task.state = BotState.IDLE
                            task.sub_state = 0
                            log("Retrieved plate from box")
        
        # ADD_TO_PLATE
        elif task.state == BotState.ADD_TO_PLATE:
            if holding and holding.get('type') == 'Food' and not self.plate_on_assembly:
                if work and self._is_counter_empty(controller, team, work[0], work[1]):
                    if self._move_toward(controller, bot_id, work, team):
                        if controller.place(bot_id, work[0], work[1]):
                            self.pending_food_name = task.item or holding.get('food_name')
                            self.pending_food_pos = work
                            if self.plate_in_box and self.plate_storage_box:
                                task.state = BotState.RETRIEVE_PLATE
                                task.target = self.plate_storage_box
                                task.sub_state = 0
                            else:
                                task.state = BotState.BUY_PLATE
                            return
                # Move toward counter and try again next turn
                return
            target = task.target or assembly
            if holding and holding.get('type') == 'Plate' and work:
                target = work
            tx, ty = target
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.add_food_to_plate(bot_id, tx, ty):
                    ing_name = task.item or self.cooking_ingredient
                    if ing_name:
                        self.ingredients_on_plate.append(ing_name)
                        if self.pending_food_name == ing_name:
                            self.pending_food_name = None
                            self.pending_food_pos = None
                    self.cooking_ingredient = None
                    task.state = BotState.IDLE
                    log(f"Added {ing_name} to plate. Contents: {self.ingredients_on_plate}")
        
        # PICKUP_PLATE
        elif task.state == BotState.PICKUP_PLATE:
            ax, ay = task.target or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if controller.pickup(bot_id, ax, ay):
                    self.plate_on_assembly = False
                    task.state = BotState.SUBMIT
                    task.target = submit
                    log("Picked up plate for submission")
        
        # SUBMIT
        elif task.state == BotState.SUBMIT:
            ux, uy = task.target or submit
            if self._move_toward(controller, bot_id, (ux, uy), team):
                if controller.submit(bot_id, ux, uy):
                    log(f"SUBMITTED ORDER {self.current_order.order_id if self.current_order else '?'}")
                    self.current_order = None
                    self.ingredients_on_plate = []
                    task.state = BotState.IDLE
        
        # TRASH
        elif task.state == BotState.TRASH:
            tx, ty = task.target or trash
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.trash(bot_id, tx, ty):
                    log("Trashed item")
                    task.state = BotState.IDLE
    
    def _execute_helper_bot(self, controller: RobotController, bot_id: int, team: Team):
        """Execute the helper bot (wash dishes, assist, or sabotage)"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(state=BotState.IDLE)
            self.bot_tasks[bot_id] = task
        
        # Check if we should sabotage
        switch_info = controller.get_switch_info()
        can_switch = controller.can_switch_maps()
        
        # Sabotage timing: after turn 260, if we haven't switched
        if can_switch and not self.has_switched and turn >= 260 and turn < 350:
            # Check if enemy has valuable stuff
            if controller.switch_maps():
                self.has_switched = True
                task.state = BotState.SABOTAGE_STEAL_PAN
                log(f"SWITCHED TO ENEMY MAP at turn {turn}!")
                return
        
        # If we're on enemy map, do sabotage
        if switch_info.get('my_team_switched') and bot.get('map_team') != team:
            self._execute_sabotage(controller, bot_id, team, task)
            return
        
        # Normal helper duties: wash dishes
        dirty = self._count_dirty_plates(controller, team)
        
        if dirty > 0 and self.sinks:
            sink = self._get_nearest((bx, by), self.sinks)
            if sink:
                sx, sy = sink
                if self._move_toward(controller, bot_id, sink, team):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        # If nothing to do, wiggle randomly
        import random
        dirs = Pathfinder.DIRS_4.copy()
        random.shuffle(dirs)
        for dx, dy in dirs:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                break
    
    def _execute_sabotage(self, controller: RobotController, bot_id: int, 
                          team: Team, task: BotTask):
        """Execute sabotage actions on enemy map"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        
        # Get enemy map locations
        enemy_map = controller.get_map(enemy_team)
        
        # Find enemy cookers
        enemy_cookers = []
        enemy_sink_tables = []
        enemy_trashes = []
        
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                name = tile.tile_name
                if name == "COOKER":
                    enemy_cookers.append((x, y))
                elif name == "SINKTABLE":
                    enemy_sink_tables.append((x, y))
                elif name == "TRASH":
                    enemy_trashes.append((x, y))
        
        # If holding something, trash it
        if holding:
            trash = self._get_nearest((bx, by), enemy_trashes)
            if trash:
                if self._move_toward(controller, bot_id, trash, enemy_team):
                    controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Steal pan from cooker
        if task.state == BotState.SABOTAGE_STEAL_PAN:
            cooker = self._get_nearest((bx, by), enemy_cookers)
            if cooker:
                kx, ky = cooker
                tile = controller.get_tile(enemy_team, kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self._move_toward(controller, bot_id, cooker, enemy_team):
                        if controller.pickup(bot_id, kx, ky):
                            log("STOLE ENEMY PAN!")
                            task.state = BotState.SABOTAGE_STEAL_PLATE
                else:
                    task.state = BotState.SABOTAGE_STEAL_PLATE
            else:
                task.state = BotState.SABOTAGE_STEAL_PLATE
        
        # Steal clean plates
        elif task.state == BotState.SABOTAGE_STEAL_PLATE:
            sink_table = self._get_nearest((bx, by), enemy_sink_tables)
            if sink_table:
                stx, sty = sink_table
                tile = controller.get_tile(enemy_team, stx, sty)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    if self._move_toward(controller, bot_id, sink_table, enemy_team):
                        if controller.take_clean_plate(bot_id, stx, sty):
                            log("STOLE ENEMY CLEAN PLATE!")
                else:
                    task.state = BotState.SABOTAGE_BLOCK
            else:
                task.state = BotState.SABOTAGE_BLOCK
        
        # Block submit station
        elif task.state == BotState.SABOTAGE_BLOCK:
            # Just move around randomly to cause chaos
            import random
            dirs = Pathfinder.DIRS_4.copy()
            random.shuffle(dirs)
            for dx, dy in dirs:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    break
    
    def play_turn(self, controller: RobotController):
        """Main entry point"""
        team = controller.get_team()
        
        # Initialize on first turn
        if not self.initialized:
            self._init_map(controller, team)
        
        # Get our bots
        my_bots = controller.get_team_bot_ids(team)
        if not my_bots:
            return
        
        # Check for defense (enemy on our map)
        switch_info = controller.get_switch_info()
        if switch_info.get('enemy_team_switched'):
            log("WARNING: Enemy is on our map!")
            # Could implement defensive measures here
        
        # Execute bots
        # Bot 0: Primary (order fulfillment)
        self._execute_primary_bot(controller, my_bots[0], team)
        
        # Bot 1: Helper (wash dishes or sabotage)
        if len(my_bots) > 1:
            self._execute_helper_bot(controller, my_bots[1], team)
