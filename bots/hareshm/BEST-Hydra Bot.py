"""
ULTIMATE TOURNAMENT BOT - The Perfect Fusion
==============================================

Combines ALL winning strategies from existing bots with ZERO weaknesses.

ARCHITECTURE:
1. MAP CLASSIFIER - Analyze map at startup, detect optimal strategy
2. MODE SELECTOR - Choose best mode: RUSH, EFFICIENCY, TURTLE, AGGRESSIVE, BALANCED
3. RUNTIME ADAPTATION - Monitor scores, adjust strategy dynamically
4. CORE SYSTEMS - Always active: FastPathfinder, emergency saves, plate recycling
5. ORDER SELECTION ENGINE - Mode-aware composite scoring
6. BOT COORDINATION - Dynamic role assignment based on mode
7. SABOTAGE DECISION TREE - Strategic switching with score awareness
8. EDGE CASE HANDLERS - Graceful recovery from any situation

SOURCES:
- champion_bot: FastPathfinder, order scoring, plate recycling, box buffering
- zone_coordinator: Map topology analysis, chokepoint detection
- aggressive_saboteur: Early switch timing, pan/plate stealing
- turtle_defender: Parallel completion, role specialization
- efficiency_maximizer: Reward-to-effort ratio optimization
- rush_bot: Simple order focus, volume over optimization
- adaptive_switcher: Score monitoring, conditional sabotage

TARGET: Beat ALL bots on ALL maps with consistent, adaptive play.
"""

import numpy as np
from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import random

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
    if DEBUG:
        print(f"[ULTIMATE] {msg}")


# Ingredient processing info
INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'turns': 3},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'turns': 25},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'turns': 8},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'turns': 3},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'turns': 35},
}


# =============================================================================
# SECTION 1: STRATEGIC MODES
# =============================================================================

class GameMode(Enum):
    """Strategic modes the bot can operate in"""
    RUSH = auto()        # Fast simple orders, volume over optimization
    EFFICIENCY = auto()  # Best reward-to-effort ratio
    TURTLE = auto()      # Pure throughput, no sabotage, parallel bots
    AGGRESSIVE = auto()  # Early sabotage focused
    BALANCED = auto()    # Adaptive default


class MapSize(Enum):
    """Map size categories"""
    TINY = auto()      # < 100 walkable tiles
    SMALL = auto()     # 100-200 walkable tiles
    MEDIUM = auto()    # 200-400 walkable tiles
    LARGE = auto()     # > 400 walkable tiles


class OrderPattern(Enum):
    """Order complexity patterns"""
    SIMPLE_HEAVY = auto()   # Mostly 1-2 ingredient orders
    COMPLEX_HEAVY = auto()  # Mostly 4-5 ingredient orders
    MIXED = auto()          # Variety of complexities
    TIME_PRESSURE = auto()  # Short duration orders


class BotRole(Enum):
    """Dynamic bot role assignments"""
    PRIMARY = auto()       # Main order processor
    HELPER = auto()        # Dish washing, assistance
    PARALLEL = auto()      # Independent order processing
    SABOTEUR = auto()      # Enemy disruption
    COOKER = auto()        # Handles cooking items
    RUNNER = auto()        # Handles simple non-cooking items


class BotState(Enum):
    """Bot state machine states"""
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
    SABOTAGE_STEAL_COUNTER = auto()
    SABOTAGE_BLOCK = auto()


# =============================================================================
# SECTION 2: FAST PATHFINDER (from champion_bot)
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
        self.walkable_set: Set[Tuple[int, int]] = set()
        
        for x in range(self.width):
            for y in range(self.height):
                is_walk = getattr(map_obj.tiles[x][y], 'is_walkable', False)
                self.walkable[x, y] = is_walk
                if is_walk:
                    self.walkable_set.add((x, y))
        
        # Cache tile locations by type
        self.tile_cache: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for x in range(self.width):
            for y in range(self.height):
                tile_name = getattr(map_obj.tiles[x][y], 'tile_name', '')
                if tile_name:
                    self.tile_cache[tile_name].append((x, y))
        
        # Pre-compute distance matrices for key tiles
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        key_tiles = ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE', 'BOX']
        for tile_name in key_tiles:
            for pos in self.tile_cache.get(tile_name, []):
                self.dist_matrices[pos] = self._compute_distance_matrix(pos)
    
    def _compute_distance_matrix(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS to compute distance from every tile to adjacent-to-target"""
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
    
    @staticmethod
    def chebyshev(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    def get_dist(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Get distance between two points using pre-computed matrices"""
        matrix = self.dist_matrices.get(to_pos)
        if matrix is not None:
            return matrix[from_pos[0], from_pos[1]]
        return self.chebyshev(from_pos, to_pos)
    
    def get_nearest_tile(self, pos: Tuple[int, int], tile_name: str) -> Optional[Tuple[int, int]]:
        """Get nearest tile of given type"""
        locations = self.tile_cache.get(tile_name, [])
        if not locations:
            return None
        return min(locations, key=lambda p: self.get_dist(pos, p))
    
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
# SECTION 3: MAP CLASSIFIER (from zone_coordinator + new logic)
# =============================================================================

@dataclass
class MapAnalysis:
    """Results of map analysis"""
    size: MapSize = MapSize.MEDIUM
    order_pattern: OrderPattern = OrderPattern.MIXED
    num_cookers: int = 0
    num_counters: int = 0
    num_sinks: int = 0
    num_boxes: int = 0
    has_chokepoints: bool = False
    chokepoints: List[Tuple[int, int]] = field(default_factory=list)
    walkable_tiles: int = 0
    recommended_mode: GameMode = GameMode.BALANCED


class MapClassifier:
    """Analyzes map and recommends optimal strategy"""
    
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def __init__(self, pathfinder: FastPathfinder):
        self.pf = pathfinder
    
    def classify(self, controller: RobotController, team: Team) -> MapAnalysis:
        """Run full map classification"""
        analysis = MapAnalysis()
        
        # Count resources
        analysis.num_cookers = len(self.pf.tile_cache.get('COOKER', []))
        analysis.num_counters = len(self.pf.tile_cache.get('COUNTER', []))
        analysis.num_sinks = len(self.pf.tile_cache.get('SINK', []))
        analysis.num_boxes = len(self.pf.tile_cache.get('BOX', []))
        analysis.walkable_tiles = len(self.pf.walkable_set)
        
        # Classify size
        if analysis.walkable_tiles < 100:
            analysis.size = MapSize.TINY
        elif analysis.walkable_tiles < 200:
            analysis.size = MapSize.SMALL
        elif analysis.walkable_tiles < 400:
            analysis.size = MapSize.MEDIUM
        else:
            analysis.size = MapSize.LARGE
        
        # Detect chokepoints
        analysis.chokepoints = self._detect_chokepoints()
        analysis.has_chokepoints = len(analysis.chokepoints) > 2
        
        # Analyze order patterns
        analysis.order_pattern = self._classify_orders(controller, team)
        
        # Recommend mode based on analysis
        analysis.recommended_mode = self._recommend_mode(analysis)
        
        log(f"Map Analysis: size={analysis.size.name}, orders={analysis.order_pattern.name}, "
            f"cookers={analysis.num_cookers}, chokepoints={len(analysis.chokepoints)}, "
            f"mode={analysis.recommended_mode.name}")
        
        return analysis
    
    def _detect_chokepoints(self) -> List[Tuple[int, int]]:
        """Detect tiles that are bottlenecks (articulation points)"""
        if len(self.pf.walkable_set) < 10:
            return []
        
        # Build adjacency for walkable tiles
        adj = defaultdict(list)
        for (x, y) in self.pf.walkable_set:
            for dx, dy in self.DIRS_4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.pf.walkable_set:
                    adj[(x, y)].append((nx, ny))
        
        # Simplified chokepoint detection: tiles with <= 2 neighbors
        # that connect different regions
        chokepoints = []
        for tile in self.pf.walkable_set:
            neighbors = len(adj[tile])
            if neighbors == 2:
                # Potential chokepoint - check if it's a narrow passage
                chokepoints.append(tile)
        
        return chokepoints[:10]  # Limit to prevent excessive analysis
    
    def _classify_orders(self, controller: RobotController, team: Team) -> OrderPattern:
        """Classify order pattern based on available orders"""
        orders = controller.get_orders(team)
        if not orders:
            return OrderPattern.MIXED
        
        simple_count = 0
        complex_count = 0
        short_duration = 0
        
        current_turn = controller.get_turn()
        
        for order in orders:
            required = order.get('required', [])
            duration = order.get('expires_turn', 500) - current_turn
            
            if len(required) <= 2:
                simple_count += 1
            elif len(required) >= 4:
                complex_count += 1
            
            if duration < 60:
                short_duration += 1
        
        total = len(orders)
        
        if short_duration > total * 0.5:
            return OrderPattern.TIME_PRESSURE
        elif simple_count > total * 0.6:
            return OrderPattern.SIMPLE_HEAVY
        elif complex_count > total * 0.5:
            return OrderPattern.COMPLEX_HEAVY
        
        return OrderPattern.MIXED
    
    def _recommend_mode(self, analysis: MapAnalysis) -> GameMode:
        """Recommend best mode based on map characteristics"""
        # RUSH: Small maps with simple orders
        if analysis.size in [MapSize.TINY, MapSize.SMALL]:
            if analysis.order_pattern == OrderPattern.SIMPLE_HEAVY:
                return GameMode.RUSH
            if analysis.num_cookers <= 1:
                return GameMode.RUSH
        
        # EFFICIENCY: Mixed orders with good infrastructure
        if analysis.order_pattern == OrderPattern.MIXED:
            if analysis.num_counters >= 3 and analysis.num_cookers >= 2:
                return GameMode.EFFICIENCY
        
        # TURTLE: Large maps with abundant resources
        if analysis.size == MapSize.LARGE:
            if analysis.num_cookers >= 3 and analysis.num_counters >= 5:
                return GameMode.TURTLE
        
        # AGGRESSIVE: Small maps where sabotage is impactful
        if analysis.size in [MapSize.TINY, MapSize.SMALL]:
            if analysis.num_cookers <= 2:
                return GameMode.AGGRESSIVE
        
        # Default to BALANCED
        return GameMode.BALANCED


# =============================================================================
# SECTION 4: ORDER SELECTION ENGINE
# =============================================================================

@dataclass
class ScoredOrder:
    """Order with calculated scores"""
    order_id: int
    required: List[str]
    reward: int
    penalty: int
    expires_turn: int
    
    # Calculated
    cost: int = 0
    turns_needed: int = 0
    profit: float = 0
    efficiency: float = 0  # profit per turn
    base_score: float = 0
    mode_score: float = 0  # Mode-adjusted score
    
    def calculate(self, current_turn: int, mode: GameMode):
        """Calculate order scores based on current mode"""
        # Base cost (plate + ingredients)
        self.cost = ShopCosts.PLATE.buy_cost
        cook_count = 0
        chop_count = 0
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50, 'turns': 5})
            self.cost += info['cost']
            if info.get('cook'):
                cook_count += 1
            if info.get('chop'):
                chop_count += 1
        
        # Estimate turns needed
        self.turns_needed = 15  # Base overhead
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'turns': 5})
            self.turns_needed += info['turns']
        
        # Sequential cooking penalty
        if cook_count > 1:
            self.turns_needed += (cook_count - 1) * 20
        
        # Check if completable
        time_left = self.expires_turn - current_turn
        if self.turns_needed > time_left - 5:  # Buffer for safety
            self.mode_score = -9999
            return
        
        # Base calculations
        self.profit = self.reward - self.cost
        self.efficiency = self.profit / max(self.turns_needed, 1)
        self.base_score = self.efficiency
        
        # Mode-specific adjustments
        if mode == GameMode.RUSH:
            # Rush: Strongly prefer simple non-cooking orders
            if len(self.required) <= 2 and cook_count == 0:
                self.mode_score = self.base_score * 2.0 + 10
            elif len(self.required) <= 2:
                self.mode_score = self.base_score * 1.2
            elif cook_count > 0:
                self.mode_score = self.base_score * 0.3
            else:
                self.mode_score = self.base_score * 0.5
        
        elif mode == GameMode.EFFICIENCY:
            # Efficiency: Prefer best profit/turn, penalize complexity
            self.mode_score = self.base_score
            if len(self.required) > 3:
                self.mode_score *= 0.5  # Soft penalty for complex
            if cook_count == 0:
                self.mode_score *= 1.3  # Bonus for no cooking
            if len(self.required) == 1:
                self.mode_score *= 1.2  # Bonus for single ingredient
        
        elif mode == GameMode.TURTLE:
            # Turtle: Balance throughput, accept moderate complexity
            self.mode_score = self.base_score
            if len(self.required) <= 3:
                self.mode_score *= 1.2
            # Slight preference for cooking (use both bots)
            if cook_count == 1:
                self.mode_score *= 1.1
        
        elif mode == GameMode.AGGRESSIVE:
            # Aggressive: Focus on quick wins while sabotaging
            if len(self.required) <= 2 and cook_count == 0:
                self.mode_score = self.base_score * 1.5 + 5
            else:
                self.mode_score = self.base_score * 0.3
        
        else:  # BALANCED
            # Balanced: Moderate preferences
            self.mode_score = self.base_score
            if len(self.required) <= 2:
                self.mode_score += 2.0
            if cook_count == 0:
                self.mode_score += 1.5
            if len(self.required) >= 5:
                self.mode_score *= 0.7


class OrderSelector:
    """Selects best orders based on current mode"""
    
    @staticmethod
    def get_best_orders(controller: RobotController, team: Team, 
                        mode: GameMode, limit: int = 5, 
                        exclude_ids: Set[int] = None) -> List[ScoredOrder]:
        """Get best orders for current mode"""
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        
        scored = []
        for order in orders:
            if not order.get('is_active', False):
                continue
            
            order_id = order['order_id']
            if exclude_ids and order_id in exclude_ids:
                continue
            
            so = ScoredOrder(
                order_id=order_id,
                required=order['required'],
                reward=order['reward'],
                penalty=order.get('penalty', 0),
                expires_turn=order['expires_turn']
            )
            so.calculate(current_turn, mode)
            
            if so.mode_score > -1000:
                scored.append(so)
        
        scored.sort(key=lambda x: x.mode_score, reverse=True)
        return scored[:limit]
    
    @staticmethod
    def get_rush_order(controller: RobotController, team: Team,
                       exclude_ids: Set[int] = None) -> Optional[ScoredOrder]:
        """Get simplest non-cooking order for rush mode"""
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        
        best = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            order_id = order['order_id']
            if exclude_ids and order_id in exclude_ids:
                continue
            
            required = order['required']
            time_left = order['expires_turn'] - current_turn
            
            if time_left < 15:
                continue
            
            # Rush rule: max 2 ingredients, no cooking
            if len(required) > 2:
                continue
            
            has_cooking = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in required)
            if has_cooking:
                continue
            
            score = 100 - len(required) * 30 + order['reward'] / 10
            if score > best_score:
                best_score = score
                so = ScoredOrder(
                    order_id=order_id,
                    required=required,
                    reward=order['reward'],
                    penalty=order.get('penalty', 0),
                    expires_turn=order['expires_turn']
                )
                so.mode_score = score
                best = so
        
        return best


# =============================================================================
# SECTION 5: BOT TASK MANAGEMENT
# =============================================================================

@dataclass
class BotTask:
    """Current task for a bot"""
    state: BotState = BotState.IDLE
    role: BotRole = BotRole.PRIMARY
    target: Optional[Tuple[int, int]] = None
    item: Optional[str] = None
    order: Optional[ScoredOrder] = None
    
    # Locations
    plate_counter: Optional[Tuple[int, int]] = None
    work_counter: Optional[Tuple[int, int]] = None
    cooker: Optional[Tuple[int, int]] = None
    
    # Progress tracking
    ingredients_done: List[str] = field(default_factory=list)
    cooking_ingredient: Optional[str] = None
    cook_start_turn: int = 0
    
    # Box buffering for single-counter maps
    plate_in_box: bool = False
    
    # Stuck detection
    last_state_change: int = 0
    stuck_counter: int = 0


# =============================================================================
# SECTION 6: MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    """
    ULTIMATE TOURNAMENT BOT
    
    Adaptive strategy selection based on map analysis.
    Combines best techniques from all reference bots.
    """
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Core systems
        self.pathfinder: Optional[FastPathfinder] = None
        self.map_analysis: Optional[MapAnalysis] = None
        self.current_mode: GameMode = GameMode.BALANCED
        
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
        
        # Strategy state
        self.single_counter_mode: bool = False
        self.plate_storage_box: Optional[Tuple[int, int]] = None
        
        # Sabotage tracking
        self.has_switched: bool = False
        self.sabotage_start_turn: int = 0
        self.max_sabotage_duration: int = 80
        
        # Score monitoring
        self.last_score_check: int = 0
        self.our_last_score: int = 0
        self.enemy_last_score: int = 0
        
        # Stuck detection
        self.no_progress_turns: int = 0
        self.last_order_completed: int = 0
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _init_map(self, controller: RobotController, team: Team):
        """Initialize map data with pre-computed pathfinding and analysis"""
        m = controller.get_map(team)
        
        # Initialize fast pathfinder
        self.pathfinder = FastPathfinder(m)
        
        # Cache tile locations
        self.shops = self.pathfinder.tile_cache.get('SHOP', [])
        self.cookers = self.pathfinder.tile_cache.get('COOKER', [])
        self.counters = self.pathfinder.tile_cache.get('COUNTER', [])
        self.submits = self.pathfinder.tile_cache.get('SUBMIT', [])
        self.trashes = self.pathfinder.tile_cache.get('TRASH', [])
        self.sinks = self.pathfinder.tile_cache.get('SINK', [])
        self.sink_tables = self.pathfinder.tile_cache.get('SINKTABLE', [])
        self.boxes = self.pathfinder.tile_cache.get('BOX', [])
        
        # Assign key locations
        if self.counters:
            self.assembly_counter = self.counters[0]
            self.work_counter = self.counters[1] if len(self.counters) > 1 else self.counters[0]
        if self.cookers:
            self.primary_cooker = self.cookers[0]
        
        # Single counter mode (box buffering needed)
        self.single_counter_mode = len(self.counters) <= 1
        if self.single_counter_mode and self.boxes:
            self.plate_storage_box = self.boxes[0]
        
        # Run map analysis
        classifier = MapClassifier(self.pathfinder)
        self.map_analysis = classifier.classify(controller, team)
        self.current_mode = self.map_analysis.recommended_mode
        
        # Assign initial bot roles based on mode
        self._assign_initial_roles(controller, team)
        
        self.initialized = True
        
        log(f"Initialized: {len(self.counters)} counters, {len(self.cookers)} cookers, "
            f"mode={self.current_mode.name}")
    
    def _assign_initial_roles(self, controller: RobotController, team: Team):
        """Assign initial roles to bots based on selected mode"""
        bots = controller.get_team_bot_ids(team)
        
        for i, bot_id in enumerate(bots):
            task = BotTask()
            
            if self.current_mode == GameMode.RUSH:
                # Both bots run independent parallel pipelines
                task.role = BotRole.PARALLEL
            
            elif self.current_mode == GameMode.TURTLE:
                # Split by cooking vs non-cooking
                task.role = BotRole.COOKER if i == 0 else BotRole.RUNNER
            
            elif self.current_mode == GameMode.AGGRESSIVE:
                # One maintains production, one prepares for sabotage
                task.role = BotRole.PRIMARY if i == 0 else BotRole.HELPER
            
            elif self.current_mode == GameMode.EFFICIENCY:
                # Primary processes orders, helper washes dishes
                task.role = BotRole.PRIMARY if i == 0 else BotRole.HELPER
            
            else:  # BALANCED
                task.role = BotRole.PRIMARY if i == 0 else BotRole.HELPER
            
            self.bot_tasks[bot_id] = task
    
    # =========================================================================
    # RUNTIME ADAPTATION
    # =========================================================================
    
    def _check_runtime_adaptation(self, controller: RobotController, team: Team):
        """Monitor game state and adapt strategy if needed"""
        turn = controller.get_turn()
        
        # Only check every 20 turns
        if turn - self.last_score_check < 20:
            return
        
        self.last_score_check = turn
        
        our_money = controller.get_team_money(team)
        enemy_team = controller.get_enemy_team()
        enemy_money = controller.get_team_money(enemy_team)
        
        score_diff = our_money - enemy_money
        
        # Stuck detection
        if our_money == self.our_last_score:
            self.no_progress_turns += 20
        else:
            self.no_progress_turns = 0
        
        self.our_last_score = our_money
        self.enemy_last_score = enemy_money
        
        # Adaptation logic
        bots = controller.get_team_bot_ids(team)
        
        # If significantly behind, consider sabotage
        if score_diff < -150 and not self.has_switched:
            if turn >= 200 and turn <= 400:  # Mid-game window
                if self.current_mode != GameMode.AGGRESSIVE:
                    log(f"Adapting: Behind by {-score_diff}, considering sabotage")
                    # Don't change mode, but allow sabotage check
        
        # If significantly ahead, switch to pure turtle
        if score_diff > 200:
            if self.current_mode == GameMode.AGGRESSIVE:
                log(f"Adapting: Ahead by {score_diff}, switching to TURTLE")
                self.current_mode = GameMode.TURTLE
                self._reassign_roles(bots)
        
        # Emergency: If stuck for too long, try to reset
        if self.no_progress_turns > 60:
            log(f"WARNING: No progress for {self.no_progress_turns} turns, resetting states")
            for bot_id in bots:
                if bot_id in self.bot_tasks:
                    self.bot_tasks[bot_id] = BotTask(role=self.bot_tasks[bot_id].role)
            self.no_progress_turns = 0
    
    def _reassign_roles(self, bots: List[int]):
        """Reassign roles based on current mode"""
        for i, bot_id in enumerate(bots):
            task = self.bot_tasks.get(bot_id, BotTask())
            
            if self.current_mode == GameMode.TURTLE:
                task.role = BotRole.COOKER if i == 0 else BotRole.RUNNER
            elif self.current_mode == GameMode.RUSH:
                task.role = BotRole.PARALLEL
            else:
                task.role = BotRole.PRIMARY if i == 0 else BotRole.HELPER
            
            self.bot_tasks[bot_id] = task
    
    # =========================================================================
    # SABOTAGE DECISION TREE
    # =========================================================================
    
    def _should_sabotage(self, controller: RobotController, team: Team) -> bool:
        """Determine if we should switch to enemy map for sabotage"""
        if self.has_switched:
            return False
        
        if not controller.can_switch_maps():
            return False
        
        turn = controller.get_turn()
        our_money = controller.get_team_money(team)
        enemy_money = controller.get_team_money(controller.get_enemy_team())
        score_diff = our_money - enemy_money
        
        # Never sabotage if winning by a lot
        if score_diff > 200:
            return False
        
        # Aggressive mode: Early sabotage
        if self.current_mode == GameMode.AGGRESSIVE:
            if 80 <= turn <= 150:
                return True
        
        # Adaptive: Sabotage if behind
        if score_diff < -150:
            if 200 <= turn <= 380:
                return True
        
        # Late game desperation
        if turn >= 350 and score_diff < -100:
            return True
        
        # Strategic timing: If enemy is ahead and gaining
        if turn >= 280 and turn <= 380:
            if enemy_money > our_money:
                return True
        
        return False
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_nearest(self, pos: Tuple[int,int], locations: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
        """Get nearest location from list using pathfinder"""
        if not locations:
            return None
        return min(locations, key=lambda p: self.pathfinder.get_dist(pos, p))
    
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
        dirs = list(FastPathfinder.DIRS_8)
        random.shuffle(dirs)
        for dx, dy in dirs:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        return False
    
    def _is_counter_empty(self, controller: RobotController, team: Team, 
                          cx: int, cy: int) -> bool:
        """Check if counter is empty"""
        tile = controller.get_tile(team, cx, cy)
        return tile is not None and getattr(tile, 'item', None) is None
    
    def _get_free_counter(self, controller: RobotController, team: Team,
                          pos: Tuple[int, int], exclude: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Get nearest free counter"""
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            if self._is_counter_empty(controller, team, c[0], c[1]):
                free.append(c)
        return self._get_nearest(pos, free)
    
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
    
    def _get_plate_contents(self, controller: RobotController, team: Team,
                            cx: int, cy: int) -> Optional[List[str]]:
        """Get contents of plate on counter"""
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
    
    def _get_next_ingredient(self, order: ScoredOrder, done: List[str]) -> Optional[str]:
        """Get next ingredient needed, prioritizing cooking items first"""
        if not order:
            return None
        
        # First: Ingredients that need cooking (start early!)
        for ing in order.required:
            if ing not in done:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('cook'):
                    return ing
        
        # Second: Ingredients that need chopping
        for ing in order.required:
            if ing not in done:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('chop') and not info.get('cook'):
                    return ing
        
        # Third: Simple ingredients
        for ing in order.required:
            if ing not in done:
                return ing
        
        return None
    
    # =========================================================================
    # PRIMARY BOT EXECUTION (Full pipeline)
    # =========================================================================
    
    def _execute_primary_bot(self, controller: RobotController, bot_id: int, team: Team):
        """Execute primary bot with full order pipeline"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        current_turn = controller.get_turn()
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(role=BotRole.PRIMARY)
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
        
        # =========== STATE MACHINE ===========
        
        # IDLE: Select order and plan next action
        if task.state == BotState.IDLE:
            # Select order if needed
            if not task.order:
                orders = OrderSelector.get_best_orders(controller, team, self.current_mode)
                if orders:
                    task.order = orders[0]
                    task.ingredients_done = []
                    task.plate_counter = None
                    log(f"Selected order {task.order.order_id}: {task.order.required}")
            
            if not task.order:
                return
            
            # Recovery: holding a plate, place it
            if holding and holding.get('type') == 'Plate' and not task.plate_counter:
                task.state = BotState.PLACE_PLATE
                task.target = assembly
                return
            
            # Check if cooking needed and we have a pan
            needs_cooking = any(INGREDIENT_INFO.get(ing, {}).get('cook') 
                               for ing in task.order.required)
            
            if needs_cooking and cooker and not self._has_pan_on_cooker(controller, team, cooker[0], cooker[1]):
                task.state = BotState.BUY_PAN
                return
            
            # Get plate status
            if task.plate_counter:
                plate_contents = self._get_plate_contents(controller, team, task.plate_counter[0], task.plate_counter[1])
                if plate_contents is not None:
                    task.ingredients_done = plate_contents
            
            # Get plate if needed
            if not task.plate_counter and not task.plate_in_box:
                if sink_table and self._count_clean_plates(controller, team) > 0:
                    task.state = BotState.GET_CLEAN_PLATE
                    task.target = sink_table
                else:
                    task.state = BotState.BUY_PLATE
                return
            
            # Retrieve plate from box if stored
            if task.plate_in_box and self.plate_storage_box:
                next_ing = self._get_next_ingredient(task.order, task.ingredients_done)
                # Keep plate in box while chopping
                if next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop') and not INGREDIENT_INFO.get(next_ing, {}).get('cook'):
                    pass
                else:
                    task.state = BotState.RETRIEVE_PLATE
                    task.target = self.plate_storage_box
                    return
            
            # Check next ingredient
            next_ing = self._get_next_ingredient(task.order, task.ingredients_done)
            
            if next_ing is None:
                # All done - pickup and submit
                task.state = BotState.PICKUP_PLATE
                task.target = task.plate_counter
                return
            
            # Check cooking status
            if cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state == 1:  # Done
                    task.state = BotState.TAKE_FROM_PAN
                    task.target = cooker
                    return
                elif pan_state == 2:  # Burnt
                    task.state = BotState.TAKE_FROM_PAN
                    task.target = cooker
                    return
                elif pan_state == 0:  # Cooking
                    # Work on non-cooking ingredients while waiting
                    for ing in task.order.required:
                        if ing not in task.ingredients_done:
                            if not INGREDIENT_INFO.get(ing, {}).get('cook'):
                                next_ing = ing
                                break
                    else:
                        return  # Wait for cooking
            
            # Process next ingredient
            info = INGREDIENT_INFO.get(next_ing, {})
            task.item = next_ing
            
            # Single counter mode: stash plate before chopping
            if info.get('chop') and self.single_counter_mode and task.plate_counter and self.plate_storage_box:
                task.state = BotState.STORE_PLATE
                task.target = task.plate_counter
                return
            
            if info.get('cook') and cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state is None:  # Pan empty
                    task.state = BotState.BUY_INGREDIENT
                else:
                    # Pan busy - try non-cooking ingredient
                    for ing in task.order.required:
                        if ing not in task.ingredients_done and not INGREDIENT_INFO.get(ing, {}).get('cook'):
                            task.item = ing
                            task.state = BotState.BUY_INGREDIENT
                            return
                    return  # Wait
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
            if task.target:
                kx, ky = task.target
                if self._move_toward(controller, bot_id, (kx, ky), team):
                    if controller.place(bot_id, kx, ky):
                        task.state = BotState.IDLE
        
        # BUY_INGREDIENT
        elif task.state == BotState.BUY_INGREDIENT:
            ing_name = task.item
            food_type = getattr(FoodType, ing_name, None) if ing_name else None
            
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
                    task.target = task.plate_counter or assembly
            elif self._move_toward(controller, bot_id, shop, team):
                if food_type and money >= food_type.buy_cost:
                    if controller.buy(bot_id, food_type, sx, sy):
                        log(f"Bought {ing_name}")
        
        # PLACE_FOR_CHOP
        elif task.state == BotState.PLACE_FOR_CHOP:
            exclude = {task.plate_counter} if task.plate_counter else set()
            free_counter = self._get_free_counter(controller, team, (bx, by), exclude)
            target = free_counter or work
            
            if target and self._move_toward(controller, bot_id, target, team):
                if self._is_counter_empty(controller, team, target[0], target[1]):
                    if controller.place(bot_id, target[0], target[1]):
                        task.state = BotState.CHOP
                        task.target = target
        
        # CHOP
        elif task.state == BotState.CHOP:
            if task.target:
                wx, wy = task.target
                if self._move_toward(controller, bot_id, (wx, wy), team):
                    tile = controller.get_tile(team, wx, wy)
                    if tile and isinstance(getattr(tile, 'item', None), Food):
                        if tile.item.chopped:
                            task.state = BotState.PICKUP_CHOPPED
                        else:
                            controller.chop(bot_id, wx, wy)
        
        # PICKUP_CHOPPED
        elif task.state == BotState.PICKUP_CHOPPED:
            if task.target:
                wx, wy = task.target
                if self._move_toward(controller, bot_id, (wx, wy), team):
                    if controller.pickup(bot_id, wx, wy):
                        info = INGREDIENT_INFO.get(task.item, {})
                        if info.get('cook'):
                            task.state = BotState.START_COOK
                            task.target = cooker
                        else:
                            task.state = BotState.ADD_TO_PLATE
                            task.target = task.plate_counter or assembly
        
        # START_COOK
        elif task.state == BotState.START_COOK:
            target = task.target or cooker
            if target and self._move_toward(controller, bot_id, target, team):
                if controller.place(bot_id, target[0], target[1]):
                    task.cooking_ingredient = task.item
                    task.cook_start_turn = current_turn
                    task.state = BotState.WAIT_COOK
                    task.target = target
        
        # WAIT_COOK
        elif task.state == BotState.WAIT_COOK:
            if task.target:
                kx, ky = task.target
                pan_state = self._get_pan_food_state(controller, team, kx, ky)
                
                if pan_state == 1:  # Done
                    task.state = BotState.TAKE_FROM_PAN
                    return
                elif pan_state == 2:  # Burnt
                    task.state = BotState.TAKE_FROM_PAN
                    return
                elif pan_state == 0:  # Cooking
                    # While waiting, prep non-cook ingredients or get plate
                    if not task.plate_counter and not task.plate_in_box:
                        task.state = BotState.BUY_PLATE
                        return
                    if task.order:
                        for ing in task.order.required:
                            if ing not in task.ingredients_done and not INGREDIENT_INFO.get(ing, {}).get('cook'):
                                task.state = BotState.BUY_INGREDIENT
                                task.item = ing
                                return
        
        # TAKE_FROM_PAN
        elif task.state == BotState.TAKE_FROM_PAN:
            target = task.target or cooker
            if holding:
                h_cooked = holding.get('cooked_stage', 0)
                if h_cooked == 2:  # Burnt
                    task.state = BotState.TRASH
                    task.target = trash
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = task.plate_counter or assembly
                    task.item = task.cooking_ingredient or holding.get('food_name')
            elif target and self._move_toward(controller, bot_id, target, team):
                if controller.take_from_pan(bot_id, target[0], target[1]):
                    log("Took from pan")
        
        # BUY_PLATE
        elif task.state == BotState.BUY_PLATE:
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
                task.target = assembly
            elif self._move_toward(controller, bot_id, shop, team):
                if money >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        log("Bought plate")
        
        # GET_CLEAN_PLATE
        elif task.state == BotState.GET_CLEAN_PLATE:
            if task.target:
                stx, sty = task.target
                if holding and holding.get('type') == 'Plate':
                    task.state = BotState.PLACE_PLATE
                    task.target = assembly
                elif self._move_toward(controller, bot_id, (stx, sty), team):
                    if controller.take_clean_plate(bot_id, stx, sty):
                        log("Got clean plate from sink table")
        
        # PLACE_PLATE
        elif task.state == BotState.PLACE_PLATE:
            target = task.target or assembly
            ax, ay = target
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if self._is_counter_empty(controller, team, ax, ay):
                    if controller.place(bot_id, ax, ay):
                        task.plate_counter = (ax, ay)
                        task.plate_in_box = False
                        task.state = BotState.IDLE
                        log("Placed plate on assembly")
                else:
                    # Counter not empty - find another
                    free = self._get_free_counter(controller, team, (bx, by))
                    if free:
                        task.target = free
        
        # STORE_PLATE (single-counter workaround)
        elif task.state == BotState.STORE_PLATE:
            if holding and holding.get('type') == 'Plate':
                # Store in box
                if self.plate_storage_box:
                    if self._move_toward(controller, bot_id, self.plate_storage_box, team):
                        if controller.place(bot_id, self.plate_storage_box[0], self.plate_storage_box[1]):
                            task.plate_in_box = True
                            task.plate_counter = None
                            task.state = BotState.IDLE
                            log("Stored plate in box")
            else:
                # Pick up plate first
                if task.target:
                    if self._move_toward(controller, bot_id, task.target, team):
                        if controller.pickup(bot_id, task.target[0], task.target[1]):
                            task.plate_counter = None
        
        # RETRIEVE_PLATE
        elif task.state == BotState.RETRIEVE_PLATE:
            if holding and holding.get('type') == 'Plate':
                # Place on counter
                ax, ay = assembly
                if self._move_toward(controller, bot_id, (ax, ay), team):
                    if self._is_counter_empty(controller, team, ax, ay):
                        if controller.place(bot_id, ax, ay):
                            task.plate_counter = (ax, ay)
                            task.plate_in_box = False
                            task.state = BotState.IDLE
                            log("Retrieved plate from box")
            else:
                # Pick up from box
                if task.target:
                    if self._move_toward(controller, bot_id, task.target, team):
                        if controller.pickup(bot_id, task.target[0], task.target[1]):
                            task.plate_in_box = False
        
        # ADD_TO_PLATE
        elif task.state == BotState.ADD_TO_PLATE:
            target = task.target or task.plate_counter or assembly
            
            # If holding food but no plate on counter, need to handle
            if holding and holding.get('type') == 'Food' and not task.plate_counter:
                # Place food on work counter temporarily
                if work and self._is_counter_empty(controller, team, work[0], work[1]):
                    if self._move_toward(controller, bot_id, work, team):
                        controller.place(bot_id, work[0], work[1])
                        task.target = work  # Remember where food is
                        task.state = BotState.BUY_PLATE
                return
            
            tx, ty = target
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.add_food_to_plate(bot_id, tx, ty):
                    ing_name = task.item or task.cooking_ingredient
                    if ing_name:
                        task.ingredients_done.append(ing_name)
                    task.cooking_ingredient = None
                    task.state = BotState.IDLE
                    log(f"Added {ing_name} to plate. Contents: {task.ingredients_done}")
        
        # PICKUP_PLATE
        elif task.state == BotState.PICKUP_PLATE:
            target = task.target or task.plate_counter
            if target:
                ax, ay = target
                if self._move_toward(controller, bot_id, (ax, ay), team):
                    if controller.pickup(bot_id, ax, ay):
                        task.plate_counter = None
                        task.state = BotState.SUBMIT
                        task.target = submit
                        log("Picked up plate for submission")
        
        # SUBMIT
        elif task.state == BotState.SUBMIT:
            target = task.target or submit
            if target:
                ux, uy = target
                if self._move_toward(controller, bot_id, (ux, uy), team):
                    if controller.submit(bot_id, ux, uy):
                        log(f"SUBMITTED ORDER {task.order.order_id if task.order else '?'}")
                        self.last_order_completed = current_turn
                        task.order = None
                        task.ingredients_done = []
                        task.plate_counter = None
                        task.state = BotState.IDLE
        
        # TRASH
        elif task.state == BotState.TRASH:
            target = task.target or trash
            if target:
                tx, ty = target
                if self._move_toward(controller, bot_id, (tx, ty), team):
                    if controller.trash(bot_id, tx, ty):
                        log("Trashed item")
                        task.state = BotState.IDLE
    
    # =========================================================================
    # HELPER BOT EXECUTION
    # =========================================================================
    
    def _execute_helper_bot(self, controller: RobotController, bot_id: int, team: Team):
        """Execute helper bot (wash dishes, assist, or sabotage)"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(role=BotRole.HELPER)
            self.bot_tasks[bot_id] = task
        
        # Strategic sabotage check
        switch_info = controller.get_switch_info()
        
        if self._should_sabotage(controller, team):
            if controller.switch_maps():
                self.has_switched = True
                self.sabotage_start_turn = turn
                task.state = BotState.SABOTAGE_STEAL_PAN
                log(f"SWITCHED TO ENEMY MAP at turn {turn}!")
                return
        
        # If on enemy map, execute sabotage
        if switch_info.get('my_team_switched') and bot.get('map_team') != team:
            # Check sabotage duration limit
            if turn - self.sabotage_start_turn > self.max_sabotage_duration:
                # Time's up, return to production mode
                task.state = BotState.IDLE
                return
            
            self._execute_sabotage(controller, bot_id, team, task)
            return
        
        # Normal helper duties
        
        # If holding something, trash it (cleanup)
        if holding:
            trash = self._get_nearest((bx, by), self.trashes)
            if trash and self._move_toward(controller, bot_id, trash, team):
                controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Primary duty: wash dishes
        dirty = self._count_dirty_plates(controller, team)
        if dirty > 0 and self.sinks:
            sink = self._get_nearest((bx, by), self.sinks)
            if sink:
                sx, sy = sink
                if self._move_toward(controller, bot_id, sink, team):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        # Idle movement (avoid getting in the way)
        dirs = list(FastPathfinder.DIRS_4)
        random.shuffle(dirs)
        for dx, dy in dirs:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                break
    
    # =========================================================================
    # PARALLEL BOT EXECUTION (for RUSH mode)
    # =========================================================================
    
    def _execute_parallel_bot(self, controller: RobotController, bot_id: int, 
                              team: Team, exclude_order_ids: Set[int] = None):
        """Execute bot with independent rush pipeline"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(role=BotRole.PARALLEL)
            self.bot_tasks[bot_id] = task
        
        shop = self._get_nearest((bx, by), self.shops)
        submit = self._get_nearest((bx, by), self.submits)
        trash = self._get_nearest((bx, by), self.trashes)
        sink_table = self._get_nearest((bx, by), self.sink_tables)
        
        # Select rush order if needed
        if task.state == BotState.IDLE:
            if not task.order:
                task.order = OrderSelector.get_rush_order(controller, team, exclude_order_ids)
                task.ingredients_done = []
                task.plate_counter = None
            
            if not task.order:
                return None
            
            # Get plate
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
            else:
                task.state = BotState.BUY_PLATE
            
            return task.order.order_id
        
        # State machine for rush pipeline
        if task.state == BotState.BUY_PLATE:
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
            else:
                # Try sink table first
                if sink_table:
                    tile = controller.get_tile(team, sink_table[0], sink_table[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self._move_toward(controller, bot_id, sink_table, team):
                            controller.take_clean_plate(bot_id, sink_table[0], sink_table[1])
                        return task.order.order_id if task.order else None
                
                if shop and self._move_toward(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
        
        elif task.state == BotState.PLACE_PLATE:
            # Get free counter not used by other bot
            other_counters = set()
            for bid, t in self.bot_tasks.items():
                if bid != bot_id and t.plate_counter:
                    other_counters.add(t.plate_counter)
            
            counter = self._get_free_counter(controller, team, (bx, by), other_counters)
            if counter and self._move_toward(controller, bot_id, counter, team):
                if controller.place(bot_id, counter[0], counter[1]):
                    task.plate_counter = counter
                    task.state = BotState.BUY_INGREDIENT
        
        elif task.state == BotState.BUY_INGREDIENT:
            if not task.order:
                task.state = BotState.IDLE
                return None
            
            # Check plate contents
            plate_contents = []
            if task.plate_counter:
                contents = self._get_plate_contents(controller, team, task.plate_counter[0], task.plate_counter[1])
                if contents:
                    plate_contents = contents
            
            missing = [i for i in task.order.required if i not in plate_contents]
            if not missing:
                task.state = BotState.PICKUP_PLATE
                task.target = task.plate_counter
                return task.order.order_id
            
            next_ing = missing[0]
            task.item = next_ing
            
            if holding and holding.get('type') == 'Food':
                info = INGREDIENT_INFO.get(next_ing, {})
                if info.get('chop') and not holding.get('chopped'):
                    task.state = BotState.PLACE_FOR_CHOP
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = task.plate_counter
            elif shop and self._move_toward(controller, bot_id, shop, team):
                food_type = getattr(FoodType, next_ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        elif task.state == BotState.PLACE_FOR_CHOP:
            exclude = {task.plate_counter} if task.plate_counter else set()
            counter = self._get_free_counter(controller, team, (bx, by), exclude)
            if counter and self._move_toward(controller, bot_id, counter, team):
                if controller.place(bot_id, counter[0], counter[1]):
                    task.target = counter
                    task.state = BotState.CHOP
        
        elif task.state == BotState.CHOP:
            if task.target and self._move_toward(controller, bot_id, task.target, team):
                tile = controller.get_tile(team, task.target[0], task.target[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    if tile.item.chopped:
                        if controller.pickup(bot_id, task.target[0], task.target[1]):
                            task.state = BotState.ADD_TO_PLATE
                            task.target = task.plate_counter
                    else:
                        controller.chop(bot_id, task.target[0], task.target[1])
        
        elif task.state == BotState.ADD_TO_PLATE:
            if task.plate_counter and self._move_toward(controller, bot_id, task.plate_counter, team):
                if controller.add_food_to_plate(bot_id, task.plate_counter[0], task.plate_counter[1]):
                    task.state = BotState.BUY_INGREDIENT
        
        elif task.state == BotState.PICKUP_PLATE:
            if task.plate_counter and self._move_toward(controller, bot_id, task.plate_counter, team):
                if controller.pickup(bot_id, task.plate_counter[0], task.plate_counter[1]):
                    task.state = BotState.SUBMIT
                    task.target = submit
        
        elif task.state == BotState.SUBMIT:
            if submit and self._move_toward(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    log(f"RUSH: Submitted order {task.order.order_id if task.order else '?'}")
                    task.order = None
                    task.plate_counter = None
                    task.state = BotState.IDLE
                    return None
        
        return task.order.order_id if task.order else None
    
    # =========================================================================
    # SABOTAGE EXECUTION
    # =========================================================================
    
    def _execute_sabotage(self, controller: RobotController, bot_id: int, 
                          team: Team, task: BotTask):
        """Execute sabotage actions on enemy map"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        enemy_map = controller.get_map(enemy_team)
        
        # Scan enemy map
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
        
        # If holding something, trash it
        if holding:
            trash = self._get_nearest((bx, by), enemy_trashes)
            if trash and self._move_toward(controller, bot_id, trash, enemy_team):
                controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Sabotage priorities
        
        # Priority 1: Steal pan from cooker
        if task.state == BotState.SABOTAGE_STEAL_PAN:
            for cooker in enemy_cookers:
                tile = controller.get_tile(enemy_team, cooker[0], cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self._move_toward(controller, bot_id, cooker, enemy_team):
                        if controller.pickup(bot_id, cooker[0], cooker[1]):
                            log("STOLE ENEMY PAN!")
                            return
            task.state = BotState.SABOTAGE_STEAL_PLATE
        
        # Priority 2: Steal clean plates
        if task.state == BotState.SABOTAGE_STEAL_PLATE:
            for st in enemy_sink_tables:
                tile = controller.get_tile(enemy_team, st[0], st[1])
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    if self._move_toward(controller, bot_id, st, enemy_team):
                        if controller.take_clean_plate(bot_id, st[0], st[1]):
                            log("STOLE ENEMY CLEAN PLATE!")
                            return
            task.state = BotState.SABOTAGE_STEAL_COUNTER
        
        # Priority 3: Steal from counters
        if task.state == BotState.SABOTAGE_STEAL_COUNTER:
            for counter in enemy_counters:
                tile = controller.get_tile(enemy_team, counter[0], counter[1])
                if tile and getattr(tile, 'item', None):
                    if self._move_toward(controller, bot_id, counter, enemy_team):
                        if controller.pickup(bot_id, counter[0], counter[1]):
                            log("STOLE ITEM FROM COUNTER!")
                            return
            task.state = BotState.SABOTAGE_BLOCK
        
        # Priority 4: Block/chaos
        if task.state == BotState.SABOTAGE_BLOCK:
            # Reset and cycle
            task.state = BotState.SABOTAGE_STEAL_PAN
            # Random movement
            dirs = list(FastPathfinder.DIRS_4)
            random.shuffle(dirs)
            for dx, dy in dirs:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    break
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    def play_turn(self, controller: RobotController):
        """Main entry point - called every turn"""
        team = controller.get_team()
        
        # Initialize on first turn
        if not self.initialized:
            self._init_map(controller, team)
        
        # Runtime adaptation
        self._check_runtime_adaptation(controller, team)
        
        # Get our bots
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Check for enemy on our map (defensive awareness)
        switch_info = controller.get_switch_info()
        if switch_info.get('enemy_team_switched'):
            log("WARNING: Enemy is on our map!")
            # Could implement defensive measures here
        
        # Execute based on current mode
        if self.current_mode == GameMode.RUSH:
            # Both bots run independent parallel pipelines
            order_id_0 = self._execute_parallel_bot(controller, bots[0], team)
            if len(bots) > 1:
                exclude = {order_id_0} if order_id_0 else set()
                self._execute_parallel_bot(controller, bots[1], team, exclude)
        
        elif self.current_mode == GameMode.TURTLE:
            # Both bots process orders, split by role (handled in execute)
            self._execute_primary_bot(controller, bots[0], team)
            if len(bots) > 1:
                # Helper washes dishes or assists
                self._execute_helper_bot(controller, bots[1], team)
        
        elif self.current_mode == GameMode.AGGRESSIVE:
            # Primary does orders, helper prepared for/executes sabotage
            self._execute_primary_bot(controller, bots[0], team)
            if len(bots) > 1:
                self._execute_helper_bot(controller, bots[1], team)
        
        else:  # BALANCED, EFFICIENCY
            # Primary order fulfillment, helper supports
            self._execute_primary_bot(controller, bots[0], team)
            if len(bots) > 1:
                self._execute_helper_bot(controller, bots[1], team)
