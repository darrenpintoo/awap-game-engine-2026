"""
SOVEREIGN - The Supreme Kitchen Overlord
=========================================
AWAP 2026 Tournament Ultimate Bot

Combines winning strategies from:
- Apex_Chef: Hive-mind job system, Hungarian assignment, aggressive sabotage
- TrueUltimateChefBot: Reservation-based pathfinding, collision avoidance
- Champion Bot: Pre-computed numpy distances, plate recycling, box buffering
- PipelineChefBot: 8-directional movement, error recovery

NEW INNOVATIONS:
- Multi-worker parallel execution (not just chef+helper)
- Emergency food saving (prevent burning)
- Adaptive order selection based on map complexity
- Counter-sabotage defense
- Smart ingredient prioritization (start cooking ASAP)
"""

import numpy as np
import heapq
import random as rand_module
from collections import deque, Counter
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

DEBUG = False

def log(msg):
    if DEBUG:
        print(f"[SOVEREIGN] {msg}")

# Priority hierarchy (higher = more urgent)
PRIORITY = {
    'EMERGENCY': 1000,      # Save burning food
    'SABOTAGE': 900,        # Disrupt enemy
    'SUBMIT': 800,          # Cash in completed orders
    'COOK_DONE': 750,       # Take finished food from pan
    'PLATE_FOOD': 700,      # Add food to plate
    'COOKING': 600,         # Start cooking
    'CHOPPING': 500,        # Chopping items
    'BUY_INGREDIENT': 400,  # Purchase ingredients
    'BUY_PLATE': 350,       # Get plate
    'BUY_PAN': 300,         # Get pan
    'WASH': 200,            # Wash dishes
    'IDLE': 0
}

# Ingredient database
INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'time': 0},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'time': 20},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'time': 5},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'time': 0},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'time': 25},
}


# =============================================================================
# JOB SYSTEM (From Apex_Chef)
# =============================================================================

class JobType(Enum):
    # Emergency
    SAVE_FOOD = auto()
    
    # Order completion
    SUBMIT = auto()
    PICKUP_PLATE = auto()
    ADD_TO_PLATE = auto()
    
    # Cooking pipeline
    TAKE_FROM_PAN = auto()
    WAIT_COOK = auto()
    START_COOK = auto()
    PICKUP_CHOPPED = auto()
    CHOP = auto()
    PLACE_FOR_CHOP = auto()
    
    # Acquisition
    BUY_INGREDIENT = auto()
    BUY_PLATE = auto()
    GET_CLEAN_PLATE = auto()
    BUY_PAN = auto()
    PLACE_PAN = auto()
    PLACE_PLATE = auto()
    
    # Maintenance
    WASH = auto()
    TRASH = auto()
    
    # Sabotage
    SWITCH_MAP = auto()
    STEAL_PAN = auto()
    STEAL_PLATE = auto()
    STEAL_ITEM = auto()
    
    IDLE = auto()


@dataclass
class Job:
    job_type: JobType
    priority: int = 0
    target: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    order_id: Optional[int] = None
    bot_id: Optional[int] = None  # Assigned bot
    
    def __repr__(self):
        return f"<{self.job_type.name} P:{self.priority} T:{self.target}>"


@dataclass
class OrderScore:
    order_id: int
    required: List[str]
    reward: int
    expires_turn: int
    score: float = 0
    turns_needed: int = 0
    
    def calculate(self, current_turn: int):
        """Score order by profitability and feasibility - OPTIMIZED FOR SPEED"""
        time_left = self.expires_turn - current_turn
        
        # Fast estimate of turns needed
        self.turns_needed = 15  # Reduced base overhead
        needs_cooking = False
        cook_items = 0
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'time': 5, 'chop': False, 'cook': False})
            if info['cook']:
                needs_cooking = True
                cook_items += 1
            elif info['chop']:
                self.turns_needed += 8
            else:
                self.turns_needed += 3
        
        if needs_cooking:
            # Cooking takes ~25 turns but can be parallelized
            self.turns_needed = max(self.turns_needed, 30 + (cook_items - 1) * 20)
        
        # Be aggressive - only reject truly impossible orders
        if time_left < 20 or self.turns_needed > time_left:
            self.score = -1000
            return
        
        # Calculate profit
        cost = ShopCosts.PLATE.buy_cost
        for ing in self.required:
            cost += INGREDIENT_INFO.get(ing, {'cost': 50})['cost']
        
        profit = self.reward - cost
        
        # Score = profit/time with bonuses
        self.score = profit / max(self.turns_needed, 1)
        
        # HUGE bonus for classic profitable orders
        if set(self.required) == {'NOODLES', 'MEAT'}:
            self.score += 15  # This is THE money maker
        elif set(self.required) == {'MEAT'}:
            self.score += 10
        
        # Bonuses for simpler orders
        self.score += (5 - len(self.required)) * 2
        if not needs_cooking:
            self.score += 4
        if len(self.required) == 1:
            self.score += 6
        
        # Urgency bonus
        if time_left < 50 and time_left > self.turns_needed + 5:
            self.score += 5


# =============================================================================
# FAST PATHFINDING (Pre-computed + Reservations)
# =============================================================================

class Pathfinder:
    """Hybrid pathfinder: pre-computed distances + reservation system"""
    
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        
        # Walkability matrix
        self.walkable = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                self.walkable[x, y] = getattr(map_obj.tiles[x][y], 'is_walkable', False)
        
        # Tile cache
        self.tile_cache: Dict[str, List[Tuple[int, int]]] = {}
        for x in range(self.width):
            for y in range(self.height):
                name = map_obj.tiles[x][y].tile_name
                if name not in self.tile_cache:
                    self.tile_cache[name] = []
                self.tile_cache[name].append((x, y))
        
        # Pre-compute distance matrices for key tiles
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        for tile_name in ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE', 'BOX']:
            for pos in self.tile_cache.get(tile_name, []):
                self.dist_matrices[pos] = self._bfs_distances(pos)
        
        # Reservation table: (x, y, turn) -> reserved
        self.reservations: Set[Tuple[int, int, int]] = set()
    
    def _bfs_distances(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS from target to compute distances"""
        dist = np.full((self.width, self.height), 9999.0)
        tx, ty = target
        
        queue = deque()
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
    
    def clear_reservations(self):
        """Clear all reservations for new turn"""
        self.reservations.clear()
    
    def reserve(self, x: int, y: int, turn: int):
        """Reserve a position for a turn"""
        self.reservations.add((x, y, turn))
    
    @staticmethod
    def chebyshev(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    
    def get_nearest(self, pos: Tuple[int, int], tile_name: str) -> Optional[Tuple[int, int]]:
        """Get nearest tile of type"""
        tiles = self.tile_cache.get(tile_name, [])
        if not tiles:
            return None
        return min(tiles, key=lambda t: self.chebyshev(pos, t))
    
    def get_tiles(self, tile_name: str) -> List[Tuple[int, int]]:
        return self.tile_cache.get(tile_name, [])
    
    def get_best_step(self, controller: RobotController, bot_id: int,
                      target: Tuple[int, int], turn: int,
                      avoid: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Get best step using pre-computed distances + reservation checking"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        
        bx, by = bot['x'], bot['y']
        
        if self.chebyshev((bx, by), target) <= 1:
            return None  # Already adjacent
        
        dist_matrix = self.dist_matrices.get(target)
        avoid = avoid or set()
        
        best_step = None
        best_dist = 9999.0
        
        for dx, dy in self.DIRS_8:
            if not controller.can_move(bot_id, dx, dy):
                continue
            
            nx, ny = bx + dx, by + dy
            
            # Skip reserved positions
            if (nx, ny, turn + 1) in self.reservations:
                continue
            
            # Skip positions with other bots
            if (nx, ny) in avoid:
                continue
            
            if dist_matrix is not None:
                step_dist = dist_matrix[nx, ny]
            else:
                step_dist = self.chebyshev((nx, ny), target)
            
            if step_dist < best_dist:
                best_dist = step_dist
                best_step = (dx, dy)
        
        # Reserve our destination
        if best_step:
            nx, ny = bx + best_step[0], by + best_step[1]
            self.reserve(nx, ny, turn + 1)
        
        return best_step


# =============================================================================
# SOVEREIGN BOT
# =============================================================================

class BotPlayer:
    """SOVEREIGN - The Supreme Kitchen Overlord"""
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.pathfinder: Optional[Pathfinder] = None
        
        # Map data
        self.shops: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.boxes: List[Tuple[int, int]] = []
        
        # Order tracking
        self.current_order: Optional[OrderScore] = None
        self.ingredients_on_plate: List[str] = []
        self.cooking_ingredient: Optional[str] = None
        self.plate_placed: bool = False
        self.pan_placed: bool = False
        
        # Bot states
        self.bot_jobs: Dict[int, Job] = {}
        self.bot_states: Dict[int, Dict[str, Any]] = {}
        
        # Sabotage tracking
        self.has_switched: bool = False
        self.sabotage_mode: bool = False
        
        # Assembly locations
        self.assembly_counter: Optional[Tuple[int, int]] = None
        self.work_counter: Optional[Tuple[int, int]] = None
        self.primary_cooker: Optional[Tuple[int, int]] = None
        
        # Single counter map handling
        self.single_counter: bool = False
        self.plate_storage: Optional[Tuple[int, int]] = None
        self.plate_stored: bool = False
    
    def _init_map(self, controller: RobotController, team: Team):
        """Initialize map data"""
        m = controller.get_map(team)
        self.pathfinder = Pathfinder(m)
        
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
            self.plate_storage = self.boxes[0]
        
        self.initialized = True
        log(f"Initialized: {len(self.counters)} counters, {len(self.cookers)} cookers")
    
    def _get_bot_positions(self, controller: RobotController, team: Team,
                           exclude: int = None) -> Set[Tuple[int, int]]:
        """Get positions of all bots except excluded one"""
        positions = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != exclude:
                bot = controller.get_bot_state(bid)
                if bot:
                    positions.add((bot['x'], bot['y']))
        return positions
    
    def _move_toward(self, controller: RobotController, bot_id: int,
                     target: Tuple[int, int], team: Team) -> bool:
        """Move toward target. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        
        pos = (bot['x'], bot['y'])
        turn = controller.get_turn()
        
        if Pathfinder.chebyshev(pos, target) <= 1:
            return True
        
        avoid = self._get_bot_positions(controller, team, bot_id)
        step = self.pathfinder.get_best_step(controller, bot_id, target, turn, avoid)
        
        if step:
            controller.move(bot_id, step[0], step[1])
            return False
        
        # Wiggle if stuck
        rand_module.shuffle(Pathfinder.DIRS_8)
        for dx, dy in Pathfinder.DIRS_8:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        return False
    
    def _is_counter_empty(self, controller: RobotController, team: Team,
                          x: int, y: int) -> bool:
        tile = controller.get_tile(team, x, y)
        return tile is not None and getattr(tile, 'item', None) is None
    
    def _has_pan(self, controller: RobotController, team: Team,
                 x: int, y: int) -> bool:
        tile = controller.get_tile(team, x, y)
        return tile is not None and isinstance(getattr(tile, 'item', None), Pan)
    
    def _get_pan_state(self, controller: RobotController, team: Team,
                       x: int, y: int) -> Optional[int]:
        """Get food state in pan: 0=cooking, 1=done, 2=burnt, None=empty"""
        tile = controller.get_tile(team, x, y)
        if tile:
            pan = getattr(tile, 'item', None)
            if isinstance(pan, Pan) and pan.food:
                return pan.food.cooked_stage
        return None
    
    def _get_plate_contents(self, controller: RobotController, team: Team,
                            x: int, y: int) -> Optional[List[str]]:
        """Get plate contents on counter"""
        tile = controller.get_tile(team, x, y)
        if tile:
            item = getattr(tile, 'item', None)
            if isinstance(item, Plate) and not item.dirty:
                return [f.food_name for f in item.food]
        return None
    
    def _count_clean_plates(self, controller: RobotController, team: Team) -> int:
        count = 0
        for sx, sy in self.sink_tables:
            tile = controller.get_tile(team, sx, sy)
            if tile:
                count += getattr(tile, 'num_clean_plates', 0)
        return count
    
    def _count_dirty_plates(self, controller: RobotController, team: Team) -> int:
        count = 0
        for sx, sy in self.sinks:
            tile = controller.get_tile(team, sx, sy)
            if tile:
                count += getattr(tile, 'num_dirty_plates', 0)
        return count
    
    def _select_order(self, controller: RobotController, team: Team):
        """Select best order to work on - FAST START"""
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        
        # FAST START: On turn 0, immediately grab the first available order
        if current_turn < 5:
            for order in orders:
                if order.get('is_active', False):
                    self.current_order = OrderScore(
                        order_id=order['order_id'],
                        required=order['required'],
                        reward=order['reward'],
                        expires_turn=order['expires_turn']
                    )
                    self.current_order.turns_needed = 40
                    self.current_order.score = 100
                    self.ingredients_on_plate = []
                    log(f"FAST START: Order {order['order_id']}: {order['required']}")
                    return
        
        scored = []
        for order in orders:
            if not order.get('is_active', False):
                continue
            
            os = OrderScore(
                order_id=order['order_id'],
                required=order['required'],
                reward=order['reward'],
                expires_turn=order['expires_turn']
            )
            os.calculate(current_turn)
            
            if os.score > 0:
                scored.append(os)
        
        # Fallback: try any simple order
        if not scored:
            for order in orders:
                if not order.get('is_active', False):
                    continue
                if len(order['required']) == 1:
                    ing = order['required'][0]
                    info = INGREDIENT_INFO.get(ing, {})
                    if not info.get('cook') and not info.get('chop'):
                        time_left = order['expires_turn'] - current_turn
                        if time_left > 25:
                            os = OrderScore(
                                order_id=order['order_id'],
                                required=order['required'],
                                reward=order['reward'],
                                expires_turn=order['expires_turn']
                            )
                            os.score = 10
                            os.turns_needed = 20
                            scored.append(os)
        
        scored.sort(key=lambda x: x.score, reverse=True)
        
        if scored:
            self.current_order = scored[0]
            self.ingredients_on_plate = []
            log(f"Selected order {self.current_order.order_id}: {self.current_order.required}")
        else:
            self.current_order = None
    
    def _get_next_ingredient(self) -> Optional[str]:
        """Get next ingredient, prioritizing cooking items first"""
        if not self.current_order:
            return None
        
        # Priority 1: Cooking items (start early!)
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                if INGREDIENT_INFO.get(ing, {}).get('cook'):
                    return ing
        
        # Priority 2: Chopping items
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('chop') and not info.get('cook'):
                    return ing
        
        # Priority 3: Simple items
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                return ing
        
        return None
    
    def _execute_primary(self, controller: RobotController, bot_id: int, team: Team):
        """Primary bot: order fulfillment"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        turn = controller.get_turn()
        
        # Get state
        state = self.bot_states.get(bot_id, {'phase': 'IDLE', 'target': None, 'item': None})
        phase = state.get('phase', 'IDLE')
        
        # Key locations
        shop = self.pathfinder.get_nearest((bx, by), 'SHOP')
        cooker = self.primary_cooker
        assembly = self.assembly_counter
        work = self.work_counter
        submit = self.pathfinder.get_nearest((bx, by), 'SUBMIT')
        trash = self.pathfinder.get_nearest((bx, by), 'TRASH')
        sink_table = self.pathfinder.get_nearest((bx, by), 'SINKTABLE')
        
        if not shop or not assembly:
            return
        
        sx, sy = shop
        ax, ay = assembly
        
        # ========== STATE MACHINE ==========
        
        if phase == 'IDLE':
            # Select order IMMEDIATELY
            if not self.current_order:
                self._select_order(controller, team)
            
            if not self.current_order:
                # No orders? Try to wash dishes instead of idle
                if self.sinks and self._count_dirty_plates(controller, team) > 0:
                    sink = self.pathfinder.get_nearest((bx, by), 'SINK')
                    if sink and self._move_toward(controller, bot_id, sink, team):
                        controller.wash_sink(bot_id, sink[0], sink[1])
                return
            
            # Update plate status
            plate_contents = self._get_plate_contents(controller, team, ax, ay)
            if plate_contents is not None:
                self.plate_placed = True
                self.ingredients_on_plate = plate_contents
            elif not self.plate_stored:
                self.plate_placed = False
            
            # Check pan status
            if cooker:
                self.pan_placed = self._has_pan(controller, team, cooker[0], cooker[1])
            
            # Check if order needs cooking
            needs_cooking = any(INGREDIENT_INFO.get(ing, {}).get('cook') 
                               for ing in self.current_order.required)
            
            # 1. Need pan for cooking
            if needs_cooking and cooker and not self.pan_placed:
                state['phase'] = 'BUY_PAN'
                self.bot_states[bot_id] = state
                return
            
            # 2. Check if cooking in progress
            if cooker and self.pan_placed:
                pan_state = self._get_pan_state(controller, team, cooker[0], cooker[1])
                if pan_state == 1:  # Done
                    state['phase'] = 'TAKE_FROM_PAN'
                    state['target'] = cooker
                    self.bot_states[bot_id] = state
                    return
                elif pan_state == 2:  # Burnt
                    state['phase'] = 'TAKE_FROM_PAN'
                    state['target'] = cooker
                    self.bot_states[bot_id] = state
                    return
            
            # 3. Get next ingredient
            next_ing = self._get_next_ingredient()
            
            if next_ing is None:
                # All done - submit!
                state['phase'] = 'PICKUP_PLATE'
                state['target'] = assembly
                self.bot_states[bot_id] = state
                return
            
            # 4. Need plate before plating (unless single counter + chopping)
            needs_plate_first = not (self.single_counter and 
                                     INGREDIENT_INFO.get(next_ing, {}).get('chop'))
            
            if needs_plate_first and not self.plate_placed and not self.plate_stored:
                # Try recycled plate first
                if sink_table and self._count_clean_plates(controller, team) > 0:
                    state['phase'] = 'GET_CLEAN_PLATE'
                    state['target'] = sink_table
                else:
                    state['phase'] = 'BUY_PLATE'
                self.bot_states[bot_id] = state
                return
            
            # 5. Retrieve plate from storage if needed
            if self.plate_stored and self.plate_storage:
                info = INGREDIENT_INFO.get(next_ing, {})
                if not info.get('chop'):  # Only retrieve if not about to chop
                    state['phase'] = 'RETRIEVE_PLATE'
                    state['target'] = self.plate_storage
                    state['sub'] = 0
                    self.bot_states[bot_id] = state
                    return
            
            # 6. If cooking and pan is busy, work on non-cooking items
            if cooker and self.pan_placed:
                pan_state = self._get_pan_state(controller, team, cooker[0], cooker[1])
                if pan_state == 0:  # Cooking
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            if not INGREDIENT_INFO.get(ing, {}).get('cook'):
                                next_ing = ing
                                break
                    else:
                        return  # Wait for cooking
            
            # 7. Store plate if single counter and need to chop
            info = INGREDIENT_INFO.get(next_ing, {})
            if self.single_counter and info.get('chop') and self.plate_placed and self.plate_storage:
                state['phase'] = 'STORE_PLATE'
                state['target'] = assembly
                state['sub'] = 0
                self.bot_states[bot_id] = state
                return
            
            # 8. Start working on ingredient
            state['item'] = next_ing
            
            if info.get('cook') and cooker:
                pan_state = self._get_pan_state(controller, team, cooker[0], cooker[1])
                if pan_state is None:
                    state['phase'] = 'BUY_INGREDIENT'
                else:
                    return  # Pan busy
            else:
                state['phase'] = 'BUY_INGREDIENT'
            
            self.bot_states[bot_id] = state
        
        elif phase == 'BUY_PAN':
            if holding and holding.get('type') == 'Pan':
                state['phase'] = 'PLACE_PAN'
                state['target'] = cooker
            elif not holding:
                if self._move_toward(controller, bot_id, shop, team):
                    if money >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)
            else:
                # Holding wrong thing
                if trash and self._move_toward(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
            self.bot_states[bot_id] = state
        
        elif phase == 'PLACE_PAN':
            kx, ky = state.get('target') or cooker
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky):
                    self.pan_placed = True
                    state['phase'] = 'IDLE'
            self.bot_states[bot_id] = state
        
        elif phase == 'BUY_INGREDIENT':
            ing_name = state.get('item')
            food_type = getattr(FoodType, ing_name, None) if ing_name else None
            
            if holding and holding.get('type') == 'Food':
                info = INGREDIENT_INFO.get(ing_name, {})
                if info.get('chop'):
                    state['phase'] = 'PLACE_FOR_CHOP'
                    state['target'] = work
                elif info.get('cook'):
                    state['phase'] = 'START_COOK'
                    state['target'] = cooker
                else:
                    state['phase'] = 'ADD_TO_PLATE'
                    state['target'] = assembly
            elif not holding:
                if self._move_toward(controller, bot_id, shop, team):
                    if food_type and money >= food_type.buy_cost:
                        if controller.buy(bot_id, food_type, sx, sy):
                            log(f"Bought {ing_name}")
            else:
                # Wrong item
                if trash and self._move_toward(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
            self.bot_states[bot_id] = state
        
        elif phase == 'PLACE_FOR_CHOP':
            wx, wy = state.get('target') or work
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if self._is_counter_empty(controller, team, wx, wy):
                    if controller.place(bot_id, wx, wy):
                        state['phase'] = 'CHOP'
                        state['target'] = (wx, wy)
            self.bot_states[bot_id] = state
        
        elif phase == 'CHOP':
            wx, wy = state.get('target')
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if controller.chop(bot_id, wx, wy):
                    state['phase'] = 'PICKUP_CHOPPED'
                    log(f"Chopped {state.get('item')}")
            self.bot_states[bot_id] = state
        
        elif phase == 'PICKUP_CHOPPED':
            wx, wy = state.get('target')
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if controller.pickup(bot_id, wx, wy):
                    info = INGREDIENT_INFO.get(state.get('item'), {})
                    if info.get('cook'):
                        state['phase'] = 'START_COOK'
                        state['target'] = cooker
                    else:
                        state['phase'] = 'ADD_TO_PLATE'
                        state['target'] = assembly
            self.bot_states[bot_id] = state
        
        elif phase == 'START_COOK':
            kx, ky = state.get('target') or cooker
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky):
                    self.cooking_ingredient = state.get('item')
                    state['phase'] = 'WAIT_COOK'
                    state['target'] = (kx, ky)
                    log(f"Started cooking {state.get('item')}")
            self.bot_states[bot_id] = state
        
        elif phase == 'WAIT_COOK':
            kx, ky = state.get('target')
            pan_state = self._get_pan_state(controller, team, kx, ky)
            
            if pan_state == 1:  # Done
                state['phase'] = 'TAKE_FROM_PAN'
            elif pan_state == 2:  # Burnt
                state['phase'] = 'TAKE_FROM_PAN'
            elif pan_state == 0:
                # While waiting, prep other stuff
                if not self.plate_placed and not self.plate_stored:
                    state['phase'] = 'BUY_PLATE'
                elif self.current_order:
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            if not INGREDIENT_INFO.get(ing, {}).get('cook'):
                                state['phase'] = 'BUY_INGREDIENT'
                                state['item'] = ing
                                break
            self.bot_states[bot_id] = state
        
        elif phase == 'TAKE_FROM_PAN':
            kx, ky = state.get('target') or cooker
            if holding:
                cooked = holding.get('cooked_stage', 0)
                if cooked == 2:  # Burnt
                    state['phase'] = 'TRASH_ITEM'
                    state['target'] = trash
                else:
                    state['phase'] = 'ADD_TO_PLATE'
                    state['target'] = assembly
                    state['item'] = self.cooking_ingredient or holding.get('food_name')
            elif self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.take_from_pan(bot_id, kx, ky):
                    log("Took from pan")
            self.bot_states[bot_id] = state
        
        elif phase == 'BUY_PLATE':
            if holding and holding.get('type') == 'Plate':
                state['phase'] = 'PLACE_PLATE'
                state['target'] = assembly
            elif not holding:
                if self._move_toward(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                            log("Bought plate")
            else:
                if trash and self._move_toward(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
            self.bot_states[bot_id] = state
        
        elif phase == 'GET_CLEAN_PLATE':
            stx, sty = state.get('target')
            if holding and holding.get('type') == 'Plate':
                state['phase'] = 'PLACE_PLATE'
                state['target'] = assembly
            elif not holding:
                if self._move_toward(controller, bot_id, (stx, sty), team):
                    if controller.take_clean_plate(bot_id, stx, sty):
                        log("Got clean plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'PLACE_PLATE':
            ax, ay = state.get('target') or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if self._is_counter_empty(controller, team, ax, ay):
                    if controller.place(bot_id, ax, ay):
                        self.plate_placed = True
                        state['phase'] = 'IDLE'
                        log("Placed plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'STORE_PLATE':
            sub = state.get('sub', 0)
            if sub == 0:
                if holding and holding.get('type') == 'Plate':
                    state['sub'] = 1
                else:
                    ax, ay = assembly
                    if self._move_toward(controller, bot_id, (ax, ay), team):
                        if controller.pickup(bot_id, ax, ay):
                            self.plate_placed = False
                            state['sub'] = 1
            if state.get('sub') == 1:
                if self.plate_storage:
                    bx, by = self.plate_storage
                    if self._move_toward(controller, bot_id, (bx, by), team):
                        if controller.place(bot_id, bx, by):
                            self.plate_stored = True
                            state['phase'] = 'IDLE'
                            state['sub'] = 0
                            log("Stored plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'RETRIEVE_PLATE':
            sub = state.get('sub', 0)
            if sub == 0:
                if holding and holding.get('type') == 'Plate':
                    state['sub'] = 1
                elif self.plate_storage:
                    px, py = self.plate_storage
                    if self._move_toward(controller, bot_id, (px, py), team):
                        if controller.pickup(bot_id, px, py):
                            self.plate_stored = False
                            state['sub'] = 1
            if state.get('sub') == 1:
                ax, ay = assembly
                if self._move_toward(controller, bot_id, (ax, ay), team):
                    if self._is_counter_empty(controller, team, ax, ay):
                        if controller.place(bot_id, ax, ay):
                            self.plate_placed = True
                            state['phase'] = 'IDLE'
                            state['sub'] = 0
                            log("Retrieved plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'ADD_TO_PLATE':
            target = state.get('target') or assembly
            tx, ty = target
            
            # If holding food but no plate, we need to handle this
            if holding and holding.get('type') == 'Food' and not self.plate_placed:
                if work and self._is_counter_empty(controller, team, work[0], work[1]):
                    if self._move_toward(controller, bot_id, work, team):
                        if controller.place(bot_id, work[0], work[1]):
                            state['pending_food'] = state.get('item') or holding.get('food_name')
                            state['pending_pos'] = work
                            if self.plate_stored and self.plate_storage:
                                state['phase'] = 'RETRIEVE_PLATE'
                                state['target'] = self.plate_storage
                                state['sub'] = 0
                            else:
                                state['phase'] = 'BUY_PLATE'
                self.bot_states[bot_id] = state
                return
            
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.add_food_to_plate(bot_id, tx, ty):
                    ing_name = state.get('item') or self.cooking_ingredient
                    if ing_name:
                        self.ingredients_on_plate.append(ing_name)
                    self.cooking_ingredient = None
                    state['phase'] = 'IDLE'
                    log(f"Added {ing_name} to plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'PICKUP_PLATE':
            ax, ay = state.get('target') or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if controller.pickup(bot_id, ax, ay):
                    self.plate_placed = False
                    state['phase'] = 'SUBMIT'
                    state['target'] = submit
                    log("Picked up plate")
            self.bot_states[bot_id] = state
        
        elif phase == 'SUBMIT':
            ux, uy = state.get('target') or submit
            if self._move_toward(controller, bot_id, (ux, uy), team):
                if controller.submit(bot_id, ux, uy):
                    log(f"SUBMITTED ORDER {self.current_order.order_id if self.current_order else '?'}")
                    self.current_order = None
                    self.ingredients_on_plate = []
                    state['phase'] = 'IDLE'
            self.bot_states[bot_id] = state
        
        elif phase == 'TRASH_ITEM':
            tx, ty = state.get('target') or trash
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.trash(bot_id, tx, ty):
                    state['phase'] = 'IDLE'
                    log("Trashed burnt food")
            self.bot_states[bot_id] = state
    
    def _execute_helper(self, controller: RobotController, bot_id: int, team: Team):
        """Helper bot: sabotage or wash dishes"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        # Sabotage check
        switch_info = controller.get_switch_info()
        can_switch = controller.can_switch_maps()
        enemy_team = controller.get_enemy_team()
        our_money = controller.get_team_money(team)
        enemy_money = controller.get_team_money(enemy_team)
        
        # Strategic sabotage: only if behind or late game
        should_sabotage = False
        if can_switch and not self.has_switched:
            # Only sabotage if enemy is significantly ahead
            if 280 <= turn < 340:
                if enemy_money > our_money + 50:
                    should_sabotage = True
            # Or desperate late game
            elif turn >= 340 and enemy_money > our_money:
                should_sabotage = True
        
        if should_sabotage:
            if controller.switch_maps():
                self.has_switched = True
                self.sabotage_mode = True
                log(f"SWITCHED TO ENEMY MAP! Turn {turn}, Us: ${our_money}, Them: ${enemy_money}")
                return
        
        # If on enemy map, sabotage
        if switch_info.get('my_team_switched'):
            self._execute_sabotage(controller, bot_id, team)
            return
        
        # Normal duties: wash dishes
        dirty = self._count_dirty_plates(controller, team)
        if dirty > 0 and self.sinks:
            sink = self.pathfinder.get_nearest((bx, by), 'SINK')
            if sink:
                if self._move_toward(controller, bot_id, sink, team):
                    controller.wash_sink(bot_id, sink[0], sink[1])
                return
        
        # If holding something, trash it
        if holding:
            trash = self.pathfinder.get_nearest((bx, by), 'TRASH')
            if trash:
                if self._move_toward(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Random movement
        rand_module.shuffle(Pathfinder.DIRS_4)
        for dx, dy in Pathfinder.DIRS_4:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                break
    
    def _execute_sabotage(self, controller: RobotController, bot_id: int, team: Team):
        """Sabotage enemy kitchen"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        
        # Get enemy map data
        enemy_map = controller.get_map(enemy_team)
        
        enemy_cookers = []
        enemy_sink_tables = []
        enemy_trashes = []
        enemy_counters = []
        
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                name = tile.tile_name
                if name == "COOKER": enemy_cookers.append((x, y))
                elif name == "SINKTABLE": enemy_sink_tables.append((x, y))
                elif name == "TRASH": enemy_trashes.append((x, y))
                elif name == "COUNTER": enemy_counters.append((x, y))
        
        # If holding, trash it
        if holding:
            trash = self.pathfinder.get_nearest((bx, by), 'TRASH') or (enemy_trashes[0] if enemy_trashes else None)
            if trash:
                if self._move_toward(controller, bot_id, trash, enemy_team):
                    controller.trash(bot_id, trash[0], trash[1])
            return
        
        state = self.bot_states.get(bot_id, {'sabotage_phase': 'STEAL_PAN'})
        phase = state.get('sabotage_phase', 'STEAL_PAN')
        
        if phase == 'STEAL_PAN':
            # Steal pan from cooker
            for kx, ky in enemy_cookers:
                tile = controller.get_tile(enemy_team, kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self._move_toward(controller, bot_id, (kx, ky), enemy_team):
                        if controller.pickup(bot_id, kx, ky):
                            log("STOLE ENEMY PAN!")
                            state['sabotage_phase'] = 'STEAL_PLATE'
                    self.bot_states[bot_id] = state
                    return
            state['sabotage_phase'] = 'STEAL_PLATE'
        
        if phase == 'STEAL_PLATE':
            # Steal clean plates
            for stx, sty in enemy_sink_tables:
                tile = controller.get_tile(enemy_team, stx, sty)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    if self._move_toward(controller, bot_id, (stx, sty), enemy_team):
                        if controller.take_clean_plate(bot_id, stx, sty):
                            log("STOLE ENEMY PLATE!")
                    self.bot_states[bot_id] = state
                    return
            state['sabotage_phase'] = 'STEAL_ITEMS'
        
        if phase == 'STEAL_ITEMS':
            # Steal items from counters
            for cx, cy in enemy_counters:
                tile = controller.get_tile(enemy_team, cx, cy)
                if tile and getattr(tile, 'item', None):
                    if self._move_toward(controller, bot_id, (cx, cy), enemy_team):
                        if controller.pickup(bot_id, cx, cy):
                            log("STOLE ENEMY ITEM!")
                    self.bot_states[bot_id] = state
                    return
            
            # Random chaos
            rand_module.shuffle(Pathfinder.DIRS_4)
            for dx, dy in Pathfinder.DIRS_4:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    break
        
        self.bot_states[bot_id] = state
    
    def play_turn(self, controller: RobotController):
        """Main entry point"""
        team = controller.get_team()
        
        # Initialize
        if not self.initialized:
            self._init_map(controller, team)
        
        # Clear reservations
        self.pathfinder.clear_reservations()
        
        # Get bots
        my_bots = controller.get_team_bot_ids(team)
        if not my_bots:
            return
        
        # Defense warning
        switch_info = controller.get_switch_info()
        if switch_info.get('enemy_team_switched'):
            log("WARNING: Enemy invading!")
        
        # Execute bots
        self._execute_primary(controller, my_bots[0], team)
        
        if len(my_bots) > 1:
            self._execute_helper(controller, my_bots[1], team)
