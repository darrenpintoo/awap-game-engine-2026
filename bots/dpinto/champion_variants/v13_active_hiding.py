"""
ULTIMATE Champion Bot - AWAP 2026 Tournament Winner
=====================================================

Combines the BEST strategies from all team bots:
1. Pre-computed numpy distance matrices (PipelineChefBot)
2. Dynamic order selection by profit/turn
3. Parallel cooking - work on other ingredients while cooking
4. Dual-bot coordination with clear role separation
5. Strategic sabotage with enemy state awareness
6. Plate recycling from sink tables
7. Box buffering for single-counter maps
8. Robust error recovery

Uses updated API format: get_team_money(team), get_team_bot_ids(team), etc.
"""

import numpy as np
import heapq
from collections import deque
from typing import Tuple, Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
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

DEBUG = False  # Set True for testing

def log(msg):
    msg = "[v13_active_hiding] " + str(msg)
    if DEBUG:
        print(f"[CHAMPION] {msg}")


# Ingredient processing info
INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'processing_turns': 0},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'processing_turns': 20},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'processing_turns': 3},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'processing_turns': 0},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'processing_turns': 25},
}


# =============================================================================
# PRE-COMPUTED PATHFINDING (from PipelineChefBot - FAST!)
# =============================================================================

class FastPathfinder:
    """
    Pre-computed BFS distance matrices for instant pathfinding.
    Uses numpy for speed. Supports 8-directional (Chebyshev) movement.
    """
    
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        
        # Pre-compute walkability matrix
        self.walkable = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                self.walkable[x, y] = getattr(map_obj.tiles[x][y], 'is_walkable', False)
        
        # Cache tile locations by type
        self.tile_cache: Dict[str, List[Tuple[int, int]]] = {}
        for x in range(self.width):
            for y in range(self.height):
                tile_name = map_obj.tiles[x][y].tile_name
                if tile_name not in self.tile_cache:
                    self.tile_cache[tile_name] = []
                self.tile_cache[tile_name].append((x, y))
        
        # Pre-compute distance matrices for key tiles
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        key_tiles = ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE', 'BOX']
        for tile_name in key_tiles:
            if tile_name in self.tile_cache:
                for pos in self.tile_cache[tile_name]:
                    self.dist_matrices[pos] = self._compute_distance_matrix(pos)
    
    def _compute_distance_matrix(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS to compute distance from every tile to adjacent-to-target"""
        dist = np.full((self.width, self.height), 9999.0)
        tx, ty = target
        
        queue = deque()
        # Start from tiles adjacent to target
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.walkable[nx, ny]:
                        dist[nx, ny] = 0
                        queue.append((nx, ny))
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in self.DIRS_8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.walkable[nx, ny] and dist[nx, ny] > dist[x, y] + 1:
                        dist[nx, ny] = dist[x, y] + 1
                        queue.append((nx, ny))
        return dist
    
    @staticmethod
    def chebyshev(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    def get_nearest_tile(self, pos: Tuple[int, int], tile_name: str) -> Optional[Tuple[int, int]]:
        """Get nearest tile of given type"""
        if tile_name not in self.tile_cache or not self.tile_cache[tile_name]:
            return None
        positions = self.tile_cache[tile_name]
        return min(positions, key=lambda p: self.chebyshev(pos, p))
    
    def get_best_step(self, controller: RobotController, bot_id: int, 
                      target: Tuple[int, int], avoid: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Get best next step toward target using pre-computed distances"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        
        bx, by = bot['x'], bot['y']
        
        # Already adjacent?
        if self.chebyshev((bx, by), target) <= 1:
            return None
        
        # Get pre-computed distances if available
        dist_matrix = self.dist_matrices.get(target)
        
        best_step = None
        best_dist = 9999.0
        
        for dx, dy in self.DIRS_8:
            if not controller.can_move(bot_id, dx, dy):
                continue
            
            nx, ny = bx + dx, by + dy
            
            if avoid and (nx, ny) in avoid:
                continue
            
            if dist_matrix is not None:
                step_dist = dist_matrix[nx, ny]
            else:
                step_dist = self.chebyshev((nx, ny), target)
            
            if step_dist < best_dist:
                best_dist = step_dist
                best_step = (dx, dy)
        
        return best_step


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
    score: float = 0
    
    def calculate(self, current_turn: int):
        """Calculate order profitability using conservative time estimates"""
        self.cost = ShopCosts.PLATE.buy_cost  # Plate cost
        self.turns_needed = 15  # Base overhead (movement, buying, placing)
        
        needs_cooking = False
        cook_items = 0
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50, 'chop': False, 'cook': False, 'processing_turns': 5})
            self.cost += info['cost']
            
            if info['cook']:
                needs_cooking = True
                cook_items += 1
            if info['chop']:
                self.turns_needed += 5  # Chop: place + chop + pickup
        
        # Cooking takes 20 turns but only 1 item can cook at a time
        if needs_cooking:
            # First cook item: 25 turns (buy + place + wait + take)
            # Additional cook items: +25 each (sequential)
            self.turns_needed = max(self.turns_needed, 30 + (cook_items - 1) * 25)
        
        # Add overhead for each additional ingredient (movement between stations)
        self.turns_needed += len(self.required) * 3
        
        time_left = self.expires_turn - current_turn
        
        # Be aggressive on early turns, conservative later
        if time_left < 25:  # Order about to expire
            self.score = -1000
            return
        
        # On tight maps, take risks - better to try and fail than do nothing
        # Only reject if REALLY impossible
        if self.turns_needed > time_left:
            self.score = -1000
            return
        
        self.profit = self.reward - self.cost
        
        # Score = profit per turn, with bonuses
        self.score = self.profit / max(self.turns_needed, 1)
        
        # BIG bonus for simpler orders (fewer ingredients = more reliable)
        self.score += (5 - len(self.required)) * 2.0
        
        # Bonus for orders with no cooking (faster completion)
        if not needs_cooking:
            self.score += 3.0
        
        # Bonus for single-ingredient orders
        if len(self.required) == 1:
            self.score += 5.0
        
        # Penalty for orders with multiple cook items (risky)
        if cook_items > 1:
            self.score -= cook_items * 2.0
        
        # Bonus for orders with only simple ingredients
        simple_count = sum(1 for ing in self.required if ing in ['SAUCE', 'NOODLES'])
        self.score += simple_count * 1.0


class OrderAnalyzer:
    """Analyzes and ranks orders by profitability"""
    
    @staticmethod
    def get_best_orders(controller: RobotController, team: Team, limit: int = 5) -> List[OrderScore]:
        """Get the most profitable orders - prioritize SIMPLE and FAST orders"""
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
        
        # If no viable orders, look for ANY simple order we might complete
        if not scored:
            for order in orders:
                if not order.get('is_active', False):
                    continue
                # Try super simple orders (1 ingredient, no cook)
                if len(order['required']) == 1:
                    ing = order['required'][0]
                    info = INGREDIENT_INFO.get(ing, {})
                    if not info.get('cook') and not info.get('chop'):
                        time_left = order['expires_turn'] - current_turn
                        if time_left > 20:  # Very lenient for simple orders
                            os = OrderScore(
                                order_id=order['order_id'],
                                required=order['required'],
                                reward=order['reward'],
                                penalty=order.get('penalty', 0),
                                expires_turn=order['expires_turn']
                            )
                            os.score = 10  # Force pick it
                            os.turns_needed = 15
                            os.profit = order['reward'] - 50
                            scored.append(os)
        
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
    GET_CLEAN_PLATE = auto()
    PLACE_PLATE = auto()
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
    SABOTAGE_STEAL_PAN = auto()
    SABOTAGE_STEAL_PLATE = auto()
    SABOTAGE_BLOCK = auto()


@dataclass
class BotTask:
    state: BotState
    target: Optional[Tuple[int, int]] = None
    item: Optional[str] = None
    order_id: Optional[int] = None
    sub_state: int = 0


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    """Ultimate Champion Bot - Combines all winning strategies"""
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.pathfinder: Optional[FastPathfinder] = None
        
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
        self.cook_start_turn: int = 0
        self.has_switched: bool = False
        self.single_counter: bool = False
        self.plate_storage_box: Optional[Tuple[int, int]] = None
        self.plate_in_box: bool = False
        self.pending_food_name: Optional[str] = None
        self.pending_food_pos: Optional[Tuple[int, int]] = None
        
        # Enemy tracking for strategic sabotage
        self.enemy_progress: int = 0
        self.our_progress: int = 0
    
    def _init_map(self, controller: RobotController, team: Team):
        """Initialize map data with pre-computed pathfinding"""
        m = controller.get_map(team)
        
        # Initialize fast pathfinder
        self.pathfinder = FastPathfinder(m)
        
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
        return min(locations, key=lambda p: FastPathfinder.chebyshev(pos, p))
    
    def _get_avoid_set(self, controller: RobotController, team: Team, exclude_bot: int) -> Set[Tuple[int,int]]:
        """Get positions to avoid (other bots)"""
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != exclude_bot:
                st = controller.get_bot_state(bid)
                if st:
                    avoid.add((st['x'], st['y']))
        return avoid

    def _is_enemy_near(self, controller: RobotController, team: Team, target: Tuple[int, int], dist: int = 3) -> bool:
        """Check if an enemy bot is near a target location"""
        enemy_team = controller.get_enemy_team()
        for bid in controller.get_team_bot_ids(enemy_team):
            st = controller.get_bot_state(bid)
            if st and st.get('map_team') == team:
                if FastPathfinder.chebyshev(target, (st['x'], st['y'])) <= dist:
                    return True
        return False

    def _hide_resources(self, controller: RobotController, bot_id: int, team: Team):
        """Emergency: Move pans from cookers to safe counters if enemy is near and actually threatening"""
        bot = controller.get_bot_state(bot_id)
        if not bot or bot.get('holding'): return

        # Only hide if we aren't currently cooking something important
        for kx, ky in self.cookers:
            pan_state = self._get_pan_food_state(controller, team, kx, ky)
            # If pan is empty or has cooked food, it's a target for theft
            if self._has_pan_on_cooker(controller, team, kx, ky) : # Hide everything regardless of state
                if self._is_enemy_near(controller, team, (kx, ky), dist=2): # Tighter radius
                    # Find a counter that is FAR from any enemy
                    safe_ctr = next((c for c in self.counters if self._is_counter_empty(controller, team, c[0], c[1]) 
                                    and not self._is_enemy_near(controller, team, c, dist=4)), None)
                    if safe_ctr:
                        if self._move_toward(controller, bot_id, (kx, ky), team):
                            if controller.pickup(bot_id, kx, ky):
                                log(f"DEFENSE: Saved pan from intruder near ({kx}, {ky})")
                        return

    
    def _move_toward(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int,int], team: Team) -> bool:
        """Move toward target using pre-computed distances. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        
        pos = (bot['x'], bot['y'])
        
        if FastPathfinder.chebyshev(pos, target) <= 1:
            return True
        
        avoid = self._get_avoid_set(controller, team, bot_id)
        step = self.pathfinder.get_best_step(controller, bot_id, target, avoid)
        
        if step:
            dx, dy = step
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        # Try wiggle if stuck
        for dx, dy in FastPathfinder.DIRS_8:
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
        """Get next ingredient needed for current order, prioritizing cooking items"""
        if not self.current_order:
            return None
        
        # First pass: prioritize ingredients that need cooking (start them first!)
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('cook'):
                    return ing
        
        # Second pass: ingredients that need chopping
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('chop') and not info.get('cook'):
                    return ing
        
        # Third pass: remaining ingredients (no processing needed)
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
        current_turn = controller.get_turn()
        
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
            
            # If holding a plate and nothing pending, place it
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
            
            # 2. Check plate status
            plate_contents = self._check_plate_on_counter(controller, team, ax, ay)
            if plate_contents is not None:
                self.plate_on_assembly = True
                self.ingredients_on_plate = plate_contents
            elif self.plate_in_box:
                self.plate_on_assembly = False
            else:
                self.plate_on_assembly = False
                self.ingredients_on_plate = []
            
            # 3. Choose next ingredient needed
            next_ing = self._get_next_ingredient()
            if self.single_counter:
                # On single counter maps, prioritize choppable items first
                for ing in self.current_order.required:
                    if ing not in self.ingredients_on_plate and INGREDIENT_INFO.get(ing, {}).get('chop'):
                        next_ing = ing
                        break

            # 4. MUST have a plate before plating
            require_plate = not (self.single_counter and next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop'))
            if require_plate and not self.plate_on_assembly and not self.plate_in_box:
                # Check for clean plates at sink table first (recycling!)
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

            # Retrieve plate from box if needed
            if not self.plate_on_assembly and self.plate_in_box and self.plate_storage_box:
                if next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop'):
                    pass  # Keep plate stored while chopping
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
                    task.state = BotState.TAKE_FROM_PAN
                    task.target = cooker
                    return
                elif pan_state == 0:  # Still cooking
                    if not self.plate_on_assembly and not self.plate_in_box:
                        task.state = BotState.BUY_PLATE
                        return
                    # Work on non-cooking ingredients while waiting
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            ing_info = INGREDIENT_INFO.get(ing, {})
                            if not ing_info.get('cook'):
                                next_ing = ing
                                break
                    else:
                        return  # All remaining need cooking, wait
            
            # 6. Start working on next ingredient
            info = INGREDIENT_INFO.get(next_ing, {})
            task.item = next_ing

            # If single counter and we need to chop, stash plate
            if info.get('chop') and self.single_counter and self.plate_on_assembly and self.plate_storage_box:
                task.state = BotState.STORE_PLATE
                task.target = assembly
                task.sub_state = 0
                return
            
            if info.get('cook') and cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state is None:  # Pan empty
                    task.state = BotState.BUY_INGREDIENT
                else:
                    # Pan busy - try non-cooking ingredient
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            ing_info = INGREDIENT_INFO.get(ing, {})
                            if not ing_info.get('cook'):
                                task.item = ing
                                task.state = BotState.BUY_INGREDIENT
                                return
                    return  # Wait
            elif info.get('chop'):
                task.state = BotState.BUY_INGREDIENT
            else:
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
                    self.cook_start_turn = current_turn
                    task.state = BotState.WAIT_COOK
                    task.target = (kx, ky)
                    log(f"Started cooking {task.item}")
        
        # WAIT_COOK
        elif task.state == BotState.WAIT_COOK:
            kx, ky = task.target
            pan_state = self._get_pan_food_state(controller, team, kx, ky)
            
            if pan_state == 1:  # Done
                task.state = BotState.TAKE_FROM_PAN
                return
            elif pan_state == 2:  # Burnt
                task.state = BotState.TAKE_FROM_PAN
                return
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
        
        # TAKE_FROM_PAN
        elif task.state == BotState.TAKE_FROM_PAN:
            kx, ky = task.target or cooker
            if holding:
                h_cooked = holding.get('cooked_stage', 0)
                if h_cooked == 2:  # Burnt
                    task.state = BotState.TRASH
                    task.target = trash
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = assembly
                    # Set the item to the cooked ingredient
                    task.item = self.cooking_ingredient or holding.get('food_name')
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

        # RETRIEVE_PLATE
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
                    self.our_progress += 1
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
        
        # Strategic sabotage check
        switch_info = controller.get_switch_info()
        can_switch = controller.can_switch_maps()
        
        # Sabotage timing: after turn 240, if we're ahead or tied and haven't switched
        # OR if we're behind and desperate after turn 300
        enemy_team = controller.get_enemy_team()
        our_money = controller.get_team_money(team)
        enemy_money = controller.get_team_money(enemy_team)
        
        should_sabotage = False
        # Adaptive Sabotage: Switch only if behind or if late game and tied
        should_sabotage = False
        if can_switch and not self.has_switched:
            # Performance gap trigger: Only switch if enemy is winning or very close to winning
            gap = enemy_money - our_money
            if turn >= 200:
                if gap > 50: # We are losing significantly
                    should_sabotage = True
                elif turn >= 320 and gap >= -20: # Tight race near the end
                    should_sabotage = True
        
        if should_sabotage:
            if controller.switch_maps():
                self.has_switched = True
                task.state = BotState.SABOTAGE_STEAL_PAN
                log(f"ADAPTIVE SWITCH at turn {turn}! (Gap: ${gap})")
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

        # Collaborative Prep: Help with ingredients if primary is busy and we have SPACE
        # Only attempt if there are at least 3 counters (assembly + work + helper)
        if self.current_order and len(self.counters) >= 3:
            primary_bot_id = controller.get_team_bot_ids(team)[0]
            primary_task = self.bot_tasks.get(primary_bot_id)
            primary_item = primary_task.item if primary_task else None
            
            # Use a dedicated helper counter (index 2 or further)
            helper_ctr = self.counters[2]
            
            # Find an ingredient that needs prep but isn't being handled by primary
            helper_ing = None
            for ing in self.current_order.required:
                if ing not in self.ingredients_on_plate and ing != primary_item:
                    if not INGREDIENT_INFO.get(ing, {}).get('cook'):
                        helper_ing = ing
                        break
            
            if helper_ing:
                task.item = helper_ing
                info = INGREDIENT_INFO.get(helper_ing, {})
                
                if holding:
                    if info.get('chop') and task.state != BotState.CHOP:
                        task.state = BotState.PLACE_FOR_CHOP
                        task.target = helper_ctr
                    else:
                        task.state = BotState.ADD_TO_PLATE
                        task.target = self.assembly_counter
                else:
                    task.state = BotState.BUY_INGREDIENT
                
                if task.state == BotState.BUY_INGREDIENT:
                    shop = self._get_nearest((bx, by), self.shops)
                    food_type = getattr(FoodType, helper_ing, None)
                    if self._move_toward(controller, bot_id, shop, team):
                        if food_type and controller.get_team_money(team) >= food_type.buy_cost:
                            controller.buy(bot_id, food_type, shop[0], shop[1])
                    return
                elif task.state == BotState.PLACE_FOR_CHOP:
                    if self._move_toward(controller, bot_id, helper_ctr, team):
                        if self._is_counter_empty(controller, team, helper_ctr[0], helper_ctr[1]):
                            controller.place(bot_id, helper_ctr[0], helper_ctr[1])
                            task.state = BotState.CHOP
                    return
                elif task.state == BotState.CHOP:
                    if self._move_toward(controller, bot_id, helper_ctr, team):
                        if controller.chop(bot_id, helper_ctr[0], helper_ctr[1]):
                            task.state = BotState.ADD_TO_PLATE
                    return
                elif task.state == BotState.ADD_TO_PLATE:
                    assembly = self.assembly_counter
                    if self.plate_on_assembly:
                        if self._move_toward(controller, bot_id, assembly, team):
                            if controller.add_food_to_plate(bot_id, assembly[0], assembly[1]):
                                self.ingredients_on_plate.append(helper_ing)
                                task.state = BotState.IDLE
                    else:
                        # Drop it on a spare counter or stay idle
                        task.state = BotState.IDLE
                    return
        
        # If nothing to do, wiggle randomly
        import random
        dirs = FastPathfinder.DIRS_4.copy()
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
        
        enemy_cookers = []
        enemy_sink_tables = []
        enemy_trashes = []
        enemy_counters = []
        
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
                elif name == "COUNTER":
                    enemy_counters.append((x, y))
        
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
        
        # Block/disrupt enemy
        elif task.state == BotState.SABOTAGE_BLOCK:
            # Try to steal items from counters
            for counter in enemy_counters:
                cx, cy = counter
                tile = controller.get_tile(enemy_team, cx, cy)
                if tile and getattr(tile, 'item', None):
                    if self._move_toward(controller, bot_id, counter, enemy_team):
                        controller.pickup(bot_id, cx, cy)
                    return
            
            # Otherwise move randomly to cause chaos
            import random
            dirs = FastPathfinder.DIRS_4.copy()
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
            # Only Bot 1 (Helper) should prioritize defense to avoid crippling Bot 0's production
            if len(my_bots) > 1:
                self._hide_resources(controller, my_bots[1], team)
        
        # Execute bots
        # Bot 0: Primary (order fulfillment)
        self._execute_primary_bot(controller, my_bots[0], team)
        
        # Bot 1: Helper (wash dishes or sabotage)
        if len(my_bots) > 1:
            self._execute_helper_bot(controller, my_bots[1], team)
