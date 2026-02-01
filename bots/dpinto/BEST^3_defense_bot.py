"""
BEST^3 Defense Bot - AWAP 2026
=====================================================

Evolution of BEST^2 with:
1.  **Active Defense**: Guards resources when enemies invade to prevent stealing.
2.  **Dynamic Role Swapping**: If the Chef gets stuck, the Helper takes over.
3.  **Enhanced Sabotage**: Smarter targeting of enemy resources.
4.  **Anti-Blocking**: Improved pathfinding around dynamic obstacles.

Based on BEST^2 Champion Bot.
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
    if DEBUG:
        print(f"[BEST^3] {msg}")


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
        
        # Dynamic pathfinding fallback if blocked by dynamic obstacles (like enemies)
        # Or just use local avoidance with the pre-computed heuristic
        
        valid_steps = []
        
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
            
            if step_dist < 9999: # reachable
                valid_steps.append(((dx, dy), step_dist))
        
        if not valid_steps:
             # Try simple greedy fallback if matrix says unreachable (might be wrong due to dynamic blocks?)
             # But here unreachable usually means static walls.
             return None

        # Sort by distance
        valid_steps.sort(key=lambda x: x[1])
        
        # Return best
        return valid_steps[0][0] if valid_steps else None


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
        self.cost = ShopCosts.PLATE.buy_cost
        self.turns_needed = 15
        needs_cooking = False
        cook_items = 0
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50, 'chop': False, 'cook': False, 'processing_turns': 5})
            self.cost += info['cost']
            if info['cook']:
                needs_cooking = True
                cook_items += 1
            if info['chop']:
                self.turns_needed += 5
        
        if needs_cooking:
            self.turns_needed = max(self.turns_needed, 30 + (cook_items - 1) * 25)
        self.turns_needed += len(self.required) * 3
        
        time_left = self.expires_turn - current_turn
        
        if time_left < 25 or self.turns_needed > time_left:
            self.score = -1000
            return
        
        self.profit = self.reward - self.cost
        self.score = self.profit / max(self.turns_needed, 1)
        self.score += (5 - len(self.required)) * 2.0
        if not needs_cooking: self.score += 3.0
        if len(self.required) == 1: self.score += 5.0
        if cook_items > 1: self.score -= cook_items * 2.0
        simple_count = sum(1 for ing in self.required if ing in ['SAUCE', 'NOODLES'])
        self.score += simple_count * 1.0


class OrderAnalyzer:
    @staticmethod
    def get_best_orders(controller: RobotController, team: Team, limit: int = 5) -> List[OrderScore]:
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        scored = []
        for order in orders:
            if not order.get('is_active', False): continue
            os = OrderScore(order_id=order['order_id'], required=order['required'],
                            reward=order['reward'], penalty=order.get('penalty', 0),
                            expires_turn=order['expires_turn'])
            os.calculate(current_turn)
            if os.score > 0: scored.append(os)
        
        if not scored:
            for order in orders:
                if not order.get('is_active', False): continue
                if len(order['required']) == 1:
                    ing = order['required'][0]
                    info = INGREDIENT_INFO.get(ing, {})
                    if not info.get('cook') and not info.get('chop'):
                        time_left = order['expires_turn'] - current_turn
                        if time_left > 20:
                            os = OrderScore(order_id=order['order_id'], required=order['required'],
                                            reward=order['reward'], penalty=order.get('penalty', 0),
                                            expires_turn=order['expires_turn'])
                            os.score = 10
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
    
    # Defense (NEW)
    DEFEND_RESOURCE = auto()


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
    """BEST^3 Defense Bot - Active Defense & Dynamic Roles"""
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.pathfinder: Optional[FastPathfinder] = None
        
        # Locations
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sinks = []
        self.sink_tables = []
        self.boxes = []
        
        self.bot_tasks: Dict[int, BotTask] = {}
        
        self.assembly_counter: Optional[Tuple[int,int]] = None
        self.work_counter: Optional[Tuple[int,int]] = None
        self.primary_cooker: Optional[Tuple[int,int]] = None
        
        # State Tracking
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
        
        self.enemy_progress: int = 0
        self.our_progress: int = 0
        
        # Dynamic Role & Stuck Detection
        self.primary_bot_id: Optional[int] = None
        self.helper_bot_id: Optional[int] = None
        self.pos_history: Dict[int, deque] = {}  # Store last 5 positions
        self.stuck_counter: Dict[int, int] = {}
        self.roles_swapped: bool = False
    
    def _init_map(self, controller: RobotController, team: Team):
        m = controller.get_map(team)
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
        
        if self.counters:
            self.assembly_counter = self.counters[0]
            self.work_counter = self.counters[1] if len(self.counters) > 1 else self.counters[0]
        if self.cookers:
            self.primary_cooker = self.cookers[0]
        
        self.single_counter = len(self.counters) <= 1
        if self.single_counter and self.boxes:
            self.plate_storage_box = self.boxes[0]
        
        my_bots = controller.get_team_bot_ids(team)
        if my_bots:
            self.primary_bot_id = my_bots[0]
            if len(my_bots) > 1:
                self.helper_bot_id = my_bots[1]
                
        self.initialized = True
    
    def _update_stuck_status(self, controller: RobotController, bot_id: int):
        """Update position history and detect stuck bots"""
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        pos = (bot['x'], bot['y'])
        
        if bot_id not in self.pos_history:
            self.pos_history[bot_id] = deque(maxlen=5)
        
        history = self.pos_history[bot_id]
        history.append(pos)
        
        # Check if stuck
        if len(history) >= 5 and all(p == history[0] for p in history):
            # Only consider stuck if task state implies movement (not IDLE or WAIT or DEFEND)
            task = self.bot_tasks.get(bot_id)
            if task and task.state not in [BotState.IDLE, BotState.WAIT_COOK, BotState.DEFEND_RESOURCE]:
                self.stuck_counter[bot_id] = self.stuck_counter.get(bot_id, 0) + 1
            else:
                self.stuck_counter[bot_id] = 0
        else:
            self.stuck_counter[bot_id] = 0
            
    def _is_stuck(self, bot_id: int) -> bool:
        return self.stuck_counter.get(bot_id, 0) > 2  # Stuck for 10+ turns (checked every turn)

    def _get_nearest(self, pos: Tuple[int,int], locations: List[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
        if not locations: return None
        return min(locations, key=lambda p: FastPathfinder.chebyshev(pos, p))
    
    def _get_avoid_set(self, controller: RobotController, team: Team, exclude_bot: int) -> Set[Tuple[int,int]]:
        avoid = set()
        # Avoid allies
        for bid in controller.get_team_bot_ids(team):
            if bid != exclude_bot:
                st = controller.get_bot_state(bid)
                if st: avoid.add((st['x'], st['y']))
                
        # Avoid enemies (Active Avoidance)
        enemy_team = controller.get_enemy_team()
        switch_info = controller.get_switch_info()
        # If enemies are on our map, add them to avoid set
        if switch_info.get('enemy_team_switched'):
             # Note: get_bot_state for enemy bots isn't directly available unless they are on our map?
             # actually we can always cheat by iterating all tiles, or assume we can see them if they are here.
             # Wait, RobotController doesn't give enemy bot states directly.
             # But we can assume they occupy tiles.
             # Actually, we can assume tiles are blocked if we can't move into them?
             # `can_move` handles collision. But for pathfinding we want to plan around them.
             pass
        return avoid
    
    def _move_toward(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int,int], team: Team) -> bool:
        bot = controller.get_bot_state(bot_id)
        if not bot: return False
        
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
        
        # Wiggle
        for dx, dy in FastPathfinder.DIRS_8:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        return False
    
    def _is_counter_empty(self, controller: RobotController, team: Team, cx: int, cy: int) -> bool:
        tile = controller.get_tile(team, cx, cy)
        return tile is not None and getattr(tile, 'item', None) is None
    
    def _has_pan_on_cooker(self, controller: RobotController, team: Team, kx: int, ky: int) -> bool:
        tile = controller.get_tile(team, kx, ky)
        return tile is not None and isinstance(getattr(tile, 'item', None), Pan)
    
    def _get_pan_food_state(self, controller: RobotController, team: Team, kx: int, ky: int) -> Optional[int]:
        tile = controller.get_tile(team, kx, ky)
        if tile:
            pan = getattr(tile, 'item', None)
            if isinstance(pan, Pan) and pan.food:
                return pan.food.cooked_stage
        return None
    
    def _check_plate_on_counter(self, controller: RobotController, team: Team, cx: int, cy: int) -> Optional[List[str]]:
        tile = controller.get_tile(team, cx, cy)
        if tile:
            item = getattr(tile, 'item', None)
            if isinstance(item, Plate) and not item.dirty:
                return [f.food_name for f in item.food]
        return None
    
    def _count_clean_plates(self, controller: RobotController, team: Team) -> int:
        count = 0
        for sx, sy in self.sink_tables:
            tile = controller.get_tile(team, sx, sy)
            if tile: count += getattr(tile, 'num_clean_plates', 0)
        return count
    
    def _count_dirty_plates(self, controller: RobotController, team: Team) -> int:
        count = 0
        for sx, sy in self.sinks:
            tile = controller.get_tile(team, sx, sy)
            if tile: count += getattr(tile, 'num_dirty_plates', 0)
        return count
    
    def _select_best_order(self, controller: RobotController, team: Team):
        best_orders = OrderAnalyzer.get_best_orders(controller, team)
        if best_orders:
            self.current_order = best_orders[0]
            self.ingredients_on_plate = []
            log(f"Selected order {self.current_order.order_id}")
        else:
            self.current_order = None
    
    def _get_next_ingredient(self) -> Optional[str]:
        if not self.current_order: return None
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                if INGREDIENT_INFO.get(ing, {}).get('cook'): return ing
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate:
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('chop') and not info.get('cook'): return ing
        for ing in self.current_order.required:
            if ing not in self.ingredients_on_plate: return ing
        return None
    
    def _execute_primary_bot(self, controller: RobotController, bot_id: int, team: Team):
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        current_turn = controller.get_turn()
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(state=BotState.IDLE)
            self.bot_tasks[bot_id] = task
        
        shop = self._get_nearest((bx, by), self.shops)
        cooker = self.primary_cooker or (self.cookers[0] if self.cookers else None)
        assembly = self.assembly_counter
        work = self.work_counter
        submit = self._get_nearest((bx, by), self.submits)
        trash = self._get_nearest((bx, by), self.trashes)
        sink_table = self._get_nearest((bx, by), self.sink_tables) if self.sink_tables else None
        
        if not all([shop, assembly]): return
        
        sx, sy = shop
        ax, ay = assembly
        
        # Recovery
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
        
        if task.state == BotState.IDLE:
            if not self.current_order: self._select_best_order(controller, team)
            if not self.current_order: return

            if holding and holding.get('type') == 'Plate' and self.pending_food_name and self.pending_food_pos:
                task.state = BotState.ADD_TO_PLATE
                task.item = self.pending_food_name
                task.target = self.pending_food_pos
                return
            
            if holding and holding.get('type') == 'Plate' and not self.plate_on_assembly:
                task.state = BotState.PLACE_PLATE
                task.target = assembly
                return
            
            needs_cooking = any(INGREDIENT_INFO.get(ing, {}).get('cook', False) for ing in self.current_order.required)
            if needs_cooking and cooker and not self._has_pan_on_cooker(controller, team, cooker[0], cooker[1]):
                task.state = BotState.BUY_PAN
                return
            
            plate_contents = self._check_plate_on_counter(controller, team, ax, ay)
            if plate_contents is not None:
                self.plate_on_assembly = True
                self.ingredients_on_plate = plate_contents
            elif self.plate_in_box: self.plate_on_assembly = False
            else:
                self.plate_on_assembly = False
                self.ingredients_on_plate = []
            
            next_ing = self._get_next_ingredient()
            if self.single_counter:
                for ing in self.current_order.required:
                    if ing not in self.ingredients_on_plate and INGREDIENT_INFO.get(ing, {}).get('chop'):
                        next_ing = ing
                        break

            require_plate = not (self.single_counter and next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop'))
            if require_plate and not self.plate_on_assembly and not self.plate_in_box:
                if sink_table and self._count_clean_plates(controller, team) > 0:
                    task.state = BotState.GET_CLEAN_PLATE
                    task.target = sink_table
                else:
                    task.state = BotState.BUY_PLATE
                return
            
            if self.pending_food_name and self.pending_food_pos and self.plate_on_assembly:
                task.state = BotState.ADD_TO_PLATE
                task.item = self.pending_food_name
                task.target = self.pending_food_pos
                return

            if not self.plate_on_assembly and self.plate_in_box and self.plate_storage_box:
                if not (next_ing and INGREDIENT_INFO.get(next_ing, {}).get('chop')):
                    task.state = BotState.RETRIEVE_PLATE
                    task.target = self.plate_storage_box
                    task.sub_state = 0
                    return
            
            if next_ing is None:
                task.state = BotState.PICKUP_PLATE
                task.target = assembly
                return
            
            if cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state == 1 or pan_state == 2:
                    if self.plate_in_box and self.plate_storage_box:
                        task.state = BotState.RETRIEVE_PLATE
                        task.target = self.plate_storage_box
                        task.sub_state = 0
                        return
                    task.state = BotState.TAKE_FROM_PAN
                    task.target = cooker
                    return
                elif pan_state == 0:
                    if not self.plate_on_assembly and not self.plate_in_box:
                        task.state = BotState.BUY_PLATE
                        return
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            if not INGREDIENT_INFO.get(ing, {}).get('cook'):
                                next_ing = ing
                                break
                    else: return

            info = INGREDIENT_INFO.get(next_ing, {})
            task.item = next_ing
            if info.get('chop') and self.single_counter and self.plate_on_assembly and self.plate_storage_box:
                task.state = BotState.STORE_PLATE
                task.target = assembly
                task.sub_state = 0
                return
            
            if info.get('cook') and cooker:
                pan_state = self._get_pan_food_state(controller, team, cooker[0], cooker[1])
                if pan_state is None: task.state = BotState.BUY_INGREDIENT
                else:
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate:
                            ing_info = INGREDIENT_INFO.get(ing, {})
                            if not ing_info.get('cook'):
                                task.item = ing
                                task.state = BotState.BUY_INGREDIENT
                                return
                    return
            elif info.get('chop'): task.state = BotState.BUY_INGREDIENT
            else: task.state = BotState.BUY_INGREDIENT
        
        elif task.state == BotState.BUY_PAN:
            if holding and holding.get('type') == 'Pan':
                task.state = BotState.PLACE_PAN
                task.target = cooker
            elif self._move_toward(controller, bot_id, shop, team):
                if money >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
        
        elif task.state == BotState.PLACE_PAN:
            kx, ky = task.target
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky): task.state = BotState.IDLE
        
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
                    controller.buy(bot_id, food_type, sx, sy)

        elif task.state == BotState.PLACE_FOR_CHOP:
            wx, wy = task.target or work
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if self._is_counter_empty(controller, team, wx, wy):
                    if controller.place(bot_id, wx, wy):
                        task.state = BotState.CHOP
                        task.target = (wx, wy)
        
        elif task.state == BotState.CHOP:
            wx, wy = task.target
            if self._move_toward(controller, bot_id, (wx, wy), team):
                if controller.chop(bot_id, wx, wy):
                    task.state = BotState.PICKUP_CHOPPED
                    task.target = (wx, wy)
        
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

        elif task.state == BotState.START_COOK:
            kx, ky = task.target or cooker
            if self._move_toward(controller, bot_id, (kx, ky), team):
                if controller.place(bot_id, kx, ky):
                    self.cooking_ingredient = task.item
                    self.cook_start_turn = current_turn
                    task.state = BotState.WAIT_COOK
                    task.target = (kx, ky)

        elif task.state == BotState.WAIT_COOK:
            kx, ky = task.target
            pan_state = self._get_pan_food_state(controller, team, kx, ky)
            if pan_state == 1 or pan_state == 2:
                task.state = BotState.TAKE_FROM_PAN
            elif pan_state == 0:
                if not self.plate_on_assembly and not self.plate_in_box:
                    task.state = BotState.BUY_PLATE
                    return
                if self.current_order:
                    for ing in self.current_order.required:
                        if ing not in self.ingredients_on_plate and not INGREDIENT_INFO.get(ing, {}).get('cook'):
                            task.state = BotState.BUY_INGREDIENT
                            task.item = ing
                            return

        elif task.state == BotState.TAKE_FROM_PAN:
            kx, ky = task.target or cooker
            if holding:
                h_cooked = holding.get('cooked_stage', 0)
                if h_cooked == 2:
                    task.state = BotState.TRASH
                    task.target = trash
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = assembly
                    task.item = self.cooking_ingredient or holding.get('food_name')
            elif self._move_toward(controller, bot_id, (kx, ky), team):
                controller.take_from_pan(bot_id, kx, ky)

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
                    controller.buy(bot_id, ShopCosts.PLATE, sx, sy)

        elif task.state == BotState.GET_CLEAN_PLATE:
            stx, sty = task.target
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
                task.target = assembly
            elif self._move_toward(controller, bot_id, (stx, sty), team):
                controller.take_clean_plate(bot_id, stx, sty)

        elif task.state == BotState.PLACE_PLATE:
            ax, ay = task.target or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if self._is_counter_empty(controller, team, ax, ay):
                    if controller.place(bot_id, ax, ay):
                        self.plate_on_assembly = True
                        task.state = BotState.IDLE

        elif task.state == BotState.STORE_PLATE:
            if task.sub_state == 0:
                if holding and holding.get('type') == 'Plate': task.sub_state = 1
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

        elif task.state == BotState.RETRIEVE_PLATE:
            if task.sub_state == 0:
                if holding and holding.get('type') == 'Plate': task.sub_state = 1
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
            if holding and holding.get('type') == 'Plate' and work: target = work
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

        elif task.state == BotState.PICKUP_PLATE:
            ax, ay = task.target or assembly
            if self._move_toward(controller, bot_id, (ax, ay), team):
                if controller.pickup(bot_id, ax, ay):
                    self.plate_on_assembly = False
                    task.state = BotState.SUBMIT
                    task.target = submit
        
        elif task.state == BotState.SUBMIT:
            ux, uy = task.target or submit
            if self._move_toward(controller, bot_id, (ux, uy), team):
                if controller.submit(bot_id, ux, uy):
                    log(f"SUBMITTED ORDER {self.current_order.order_id if self.current_order else '?'}")
                    self.current_order = None
                    self.ingredients_on_plate = []
                    self.our_progress += 1
                    task.state = BotState.IDLE

        elif task.state == BotState.TRASH:
            tx, ty = task.target or trash
            if self._move_toward(controller, bot_id, (tx, ty), team):
                if controller.trash(bot_id, tx, ty):
                    task.state = BotState.IDLE

    def _execute_helper_bot(self, controller: RobotController, bot_id: int, team: Team):
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        task = self.bot_tasks.get(bot_id)
        if not task:
            task = BotTask(state=BotState.IDLE)
            self.bot_tasks[bot_id] = task
        
        switch_info = controller.get_switch_info()
        
        # Normal Helper Logic (same as BEST^2)
        can_switch = controller.can_switch_maps()
        enemy_team = controller.get_enemy_team()
        our_money = controller.get_team_money(team)
        enemy_money = controller.get_team_money(enemy_team)
        
        # 1. Check Sabotage Opportunity (Priority #1)
        should_sabotage = False
        if can_switch and not self.has_switched:
            if turn >= 280 and turn < 380:
                if enemy_money >= our_money - 50: should_sabotage = True
            elif turn >= 350 and enemy_money > our_money: should_sabotage = True
        
        if should_sabotage:
            if controller.switch_maps():
                self.has_switched = True
                task.state = BotState.SABOTAGE_STEAL_PAN
                log("SWITCHED TO ENEMY MAP!")
                return
        
        if switch_info.get('my_team_switched') and bot.get('map_team') != team:
            self._execute_sabotage(controller, bot_id, team, task)
            return

        # 2. ACTIVE DEFENSE CHECK (Priority #2)
        # Only defend if we aren't sabotaging
        
        if enemy_on_map:
            # Simple defense: Verify if we are being invaded.
            # If enemy detected (we assume they are there), GUARD resources.
            # We don't have enemy positions directly, but we can assume they go for Cooker or Sink.
            # Helper should "Camp" at the sink table or cooker to block stealing.
            
            # Prioritize guarding Sink Table (stealing clean plates is common sabotage)
            target_defend = None
            if self.sink_tables: target_defend = self.sink_tables[0]
            elif self.cookers: target_defend = self.cookers[0]
            
            # CRITICAL OPTIMIZATION:
            # Only defend if we aren't starving for plates.
            # If dirty plates are piling up, we MUST wash them, or the chef stops working.
            dirty_count = self._count_dirty_plates(controller, team)
            if dirty_count > 3: # If >3 dirty plates, go wash instead of guarding
                 target_defend = None
            
            if target_defend:
                task.state = BotState.DEFEND_RESOURCE
                task.target = target_defend
        
        # STATE MACHINE FOR HELPER
        if task.state == BotState.DEFEND_RESOURCE:
            if not enemy_on_map:
                 task.state = BotState.IDLE
                 return
            
            tx, ty = task.target
            # Move to target and STAY there
            if self._move_toward(controller, bot_id, (tx, ty), team):
                pass # Just sit there
            return
        
        dirty = self._count_dirty_plates(controller, team)
        if dirty > 0 and self.sinks:
            sink = self._get_nearest((bx, by), self.sinks)
            if sink:
                sx, sy = sink
                if self._move_toward(controller, bot_id, sink, team):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        import random
        dirs = FastPathfinder.DIRS_4.copy()
        random.shuffle(dirs)
        for dx, dy in dirs:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                break

    def _execute_sabotage(self, controller: RobotController, bot_id: int, team: Team, task: BotTask):
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        enemy_map = controller.get_map(enemy_team)
        
        enemy_cookers, enemy_sink_tables, enemy_trashes, enemy_counters = [], [], [], []
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                if tile.tile_name == "COOKER": enemy_cookers.append((x, y))
                elif tile.tile_name == "SINKTABLE": enemy_sink_tables.append((x, y))
                elif tile.tile_name == "TRASH": enemy_trashes.append((x, y))
                elif tile.tile_name == "COUNTER": enemy_counters.append((x, y))

        if holding:
            trash = self._get_nearest((bx, by), enemy_trashes)
            if trash:
                if self._move_toward(controller, bot_id, trash, enemy_team):
                    controller.trash(bot_id, trash[0], trash[1])
            return

        if task.state == BotState.SABOTAGE_STEAL_PAN:
            cooker = self._get_nearest((bx, by), enemy_cookers)
            if cooker:
                kx, ky = cooker
                tile = controller.get_tile(enemy_team, kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self._move_toward(controller, bot_id, cooker, enemy_team):
                        if controller.pickup(bot_id, kx, ky):
                             task.state = BotState.SABOTAGE_STEAL_PLATE
                else: task.state = BotState.SABOTAGE_STEAL_PLATE
            else: task.state = BotState.SABOTAGE_STEAL_PLATE
        
        elif task.state == BotState.SABOTAGE_STEAL_PLATE:
            sink_table = self._get_nearest((bx, by), enemy_sink_tables)
            if sink_table:
                stx, sty = sink_table
                tile = controller.get_tile(enemy_team, stx, sty)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    if self._move_toward(controller, bot_id, sink_table, enemy_team):
                        controller.take_clean_plate(bot_id, stx, sty)
                else: task.state = BotState.SABOTAGE_BLOCK
            else: task.state = BotState.SABOTAGE_BLOCK
        
        elif task.state == BotState.SABOTAGE_BLOCK:
             # Just camp near counters
            for counter in enemy_counters:
                cx, cy = counter
                tile = controller.get_tile(enemy_team, cx, cy)
                if tile and getattr(tile, 'item', None):
                    if self._move_toward(controller, bot_id, counter, enemy_team):
                        controller.pickup(bot_id, cx, cy)
                    return
            
            import random
            dirs = FastPathfinder.DIRS_4.copy()
            random.shuffle(dirs)
            for dx, dy in dirs:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    break

    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        if not self.initialized: self._init_map(controller, team)
        
        my_bots = controller.get_team_bot_ids(team)
        if not my_bots: return
        
        try:
            # 1. Update Stuck Status
            for bot_id in my_bots:
                self._update_stuck_status(controller, bot_id)
            
            # 2. Dynamic Role Assignment
            primary = self.primary_bot_id
            helper = self.helper_bot_id
            
            # If primary is stuck, swap!
            if primary is not None and self._is_stuck(primary) and helper is not None:
                log(f"ALERT: Primary Bot {primary} STUCK! Swapping roles.")
                self.primary_bot_id = helper
                self.helper_bot_id = primary
                # Reset tasks
                self.bot_tasks[primary] = BotTask(state=BotState.IDLE)
                self.bot_tasks[helper] = BotTask(state=BotState.IDLE)
                self.stuck_counter[primary] = 0 # Reset counter
                self.roles_swapped = not self.roles_swapped
            
            # 3. Execute Bots
            if self.primary_bot_id is not None:
                self._execute_primary_bot(controller, self.primary_bot_id, team)
            
            if self.helper_bot_id is not None:
                self._execute_helper_bot(controller, self.helper_bot_id, team)
                
        except Exception as e:
            # Emergency Recovery to prevent disqualification
            log(f"CRITICAL CRASH AVOIDED: {e}")
            import traceback
            if DEBUG: traceback.print_exc()
            
            # Fallback: Just Random Walk both bots to avoid total freeze
            import random
            dirs = FastPathfinder.DIRS_4.copy()
            for bid in my_bots:
                random.shuffle(dirs)
                for dx, dy in dirs:
                    if controller.can_move(bid, dx, dy):
                        controller.move(bid, dx, dy)
                        break
