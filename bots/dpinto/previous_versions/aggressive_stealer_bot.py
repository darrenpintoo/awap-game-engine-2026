"""
Aggressive Stealer Bot - AWAP 2026 Tournament Entry
===================================================

Strategy:
- Primary bot focuses on efficiency (same as Champion).
- Helper bot aggressively switches to enemy map EARLY (turn 60).
- Thief priority:
  1. Steal Pans (crucial resource).
  2. Steal Clean Plates (disrupts plating).
  3. Body block the Submit station or Cookers.
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

DEBUG = False

def log(msg):
    if DEBUG:
        print(f"[STEALER] {msg}")

# Ingredient processing info
INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'processing_turns': 0},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'processing_turns': 20},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'processing_turns': 3},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'processing_turns': 0},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'processing_turns': 25},
}

# =============================================================================
# PRE-COMPUTED PATHFINDING (FastPathfinder)
# =============================================================================

class FastPathfinder:
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        
        self.walkable = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                self.walkable[x, y] = getattr(map_obj.tiles[x][y], 'is_walkable', False)
        
        self.tile_cache = {}
        for x in range(self.width):
            for y in range(self.height):
                tile_name = map_obj.tiles[x][y].tile_name
                if tile_name not in self.tile_cache:
                    self.tile_cache[tile_name] = []
                self.tile_cache[tile_name].append((x, y))
        
        self.dist_matrices = {}
        key_tiles = ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE', 'BOX']
        for tile_name in key_tiles:
            if tile_name in self.tile_cache:
                for pos in self.tile_cache[tile_name]:
                    self.dist_matrices[pos] = self._compute_distance_matrix(pos)
    
    def _compute_distance_matrix(self, target):
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
    def chebyshev(p1, p2):
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    def get_best_step(self, controller, bot_id, target, avoid=None):
        bot = controller.get_bot_state(bot_id)
        if not bot: return None
        bx, by = bot['x'], bot['y']
        
        if self.chebyshev((bx, by), target) <= 1:
            return None
        
        dist_matrix = self.dist_matrices.get(target)
        best_step = None
        best_dist = 9999.0
        
        for dx, dy in self.DIRS_8:
            if not controller.can_move(bot_id, dx, dy): continue
            nx, ny = bx + dx, by + dy
            if avoid and (nx, ny) in avoid: continue
            
            step_dist = dist_matrix[nx, ny] if dist_matrix is not None else self.chebyshev((nx, ny), target)
            
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
    cost: int = 0
    turns_needed: int = 0
    profit: float = 0
    score: float = 0
    
    def calculate(self, current_turn):
        self.cost = ShopCosts.PLATE.buy_cost
        self.turns_needed = 15
        needs_cooking = False
        cook_items = 0
        
        for ing in self.required:
            info = INGREDIENT_INFO.get(ing, {'cost': 50, 'chop': False, 'cook': False})
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
        
        if time_left < 20 or self.turns_needed > time_left:
            self.score = -1000
            return
        
        self.profit = self.reward - self.cost
        self.score = self.profit / max(self.turns_needed, 1)
        
        # Simpler scoring than champion - just money
        self.score += (5 - len(self.required)) * 1.5

class OrderAnalyzer:
    @staticmethod
    def get_best_orders(controller, team, limit=5):
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        scored = []
        for order in orders:
            if not order.get('is_active', False): continue
            os = OrderScore(order['order_id'], order['required'], order['reward'], 
                          order.get('penalty', 0), order['expires_turn'])
            os.calculate(current_turn)
            if os.score > 0: scored.append(os)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

# =============================================================================
# BOT STATES
# =============================================================================

class BotState(Enum):
    IDLE = auto()
    BUY_PAN = auto()
    PLACE_PAN = auto()
    BUY_INGREDIENT = auto()
    PLACE_FOR_CHOP = auto()
    CHOP = auto()
    PICKUP_CHOPPED = auto()
    START_COOK = auto()
    WAIT_COOK = auto()
    TAKE_FROM_PAN = auto()
    BUY_PLATE = auto()
    PLACE_PLATE = auto()
    ADD_TO_PLATE = auto()
    PICKUP_PLATE = auto()
    SUBMIT = auto()
    TRASH = auto()
    
    # Sabotage
    SABOTAGE_STEAL_PAN = auto()
    SABOTAGE_STEAL_PLATE = auto()
    SABOTAGE_BLOCK_SUBMIT = auto()
    SABOTAGE_BLOCK_COOKER = auto()

@dataclass
class BotTask:
    state: BotState
    target: Optional[Tuple[int, int]] = None
    item: Optional[str] = None
    order_id: Optional[int] = None

# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.pathfinder = None
        
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.boxes = []
        self.sinks = []
        self.sink_tables = []
        
        self.bot_tasks = {}
        self.current_order = None
        self.plate_on_assembly = False
        self.ingredients_on_plate = []
        self.cooking_ingredient = None
        self.has_switched = False
        
        self.counters = []
        self.assembly_counter = None
        self.work_counter = None
    
    def _init_map(self, controller, team):
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
        self.initialized = True
    
    def _get_nearest(self, pos, locations):
        if not locations: return None
        return min(locations, key=lambda p: FastPathfinder.chebyshev(pos, p))
    
    def _move_toward(self, controller, bot_id, target, team):
        bot = controller.get_bot_state(bot_id)
        if not bot: return False
        if FastPathfinder.chebyshev((bot['x'], bot['y']), target) <= 1: return True
        
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != bot_id:
                st = controller.get_bot_state(bid)
                if st: avoid.add((st['x'], st['y']))
        
        step = self.pathfinder.get_best_step(controller, bot_id, target, avoid)
        if step:
            controller.move(bot_id, step[0], step[1])
            return False
        
        # Wiggle
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        return False
    
    def _execute_sabotage(self, controller, bot_id, team, task):
        bot = controller.get_bot_state(bot_id)
        enemy_team = controller.get_enemy_team()
        enemy_map = controller.get_map(enemy_team)
        
        # Identify enemy key locations
        enemy_cookers = []
        enemy_submits = []
        enemy_sinktables = []
        
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                if tile.tile_name == "COOKER": enemy_cookers.append((x,y))
                elif tile.tile_name == "SUBMIT": enemy_submits.append((x,y))
                elif tile.tile_name == "SINKTABLE": enemy_sinktables.append((x,y))
        
        bx, by = bot['x'], bot['y']
        
        # If holding something, run to trash (unless we want to keep it to deny)
        if bot.get('holding'):
            # Find nearest trash
            pass # Keep holding it to deny resource!
        
        # STATE 1: Steal Pan
        if task.state == BotState.SABOTAGE_STEAL_PAN:
            target_cooker = None
            for cx, cy in enemy_cookers:
                tile = controller.get_tile(enemy_team, cx, cy)
                # Check for pan
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                    target_cooker = (cx, cy)
                    break
            
            if target_cooker:
                if self._move_toward(controller, bot_id, target_cooker, enemy_team):
                    controller.pickup(bot_id, target_cooker[0], target_cooker[1])
                    log("STOLE PAN")
                    task.state = BotState.SABOTAGE_BLOCK_SUBMIT # Now go be annoying
            else:
                task.state = BotState.SABOTAGE_STEAL_PLATE # No pans, try plates

        # STATE 2: Steal Plate
        elif task.state == BotState.SABOTAGE_STEAL_PLATE:
            target_sink = None
            for sx, sy in enemy_sinktables:
                tile = controller.get_tile(enemy_team, sx, sy)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    target_sink = (sx, sy)
                    break
            
            if target_sink:
                if self._move_toward(controller, bot_id, target_sink, enemy_team):
                    controller.take_clean_plate(bot_id, target_sink[0], target_sink[1])
                    log("STOLE PLATE")
                    task.state = BotState.SABOTAGE_BLOCK_SUBMIT
            else:
                task.state = BotState.SABOTAGE_BLOCK_SUBMIT # Nothing to steal
        
        # STATE 3: Block Submit (Stand adjacent to submit to block pathfinding)
        elif task.state == BotState.SABOTAGE_BLOCK_SUBMIT:
            if not enemy_submits: return
            sub = enemy_submits[0]
            
            # Find a walkable tile adjacent to submit
            block_spot = None
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = sub[0]+dx, sub[1]+dy
                    if controller.get_tile(enemy_team, nx, ny) and getattr(controller.get_tile(enemy_team, nx, ny), 'is_walkable', False):
                        block_spot = (nx, ny)
                        break
            
            if block_spot:
                if self._move_toward(controller, bot_id, block_spot, enemy_team):
                    # We are there, stay there!
                    pass

    def _execute_primary(self, controller, bot_id, team):
        # SIMPLIFIED VERSION OF CHAMPION LOGIC
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        task = self.bot_tasks.get(bot_id, BotTask(BotState.IDLE))
        self.bot_tasks[bot_id] = task
        
        bx, by = bot['x'], bot['y']
        
        if task.state == BotState.IDLE:
            if not self.current_order:
                best = OrderAnalyzer.get_best_orders(controller, team)
                if best: self.current_order = best[0]
            
            if not self.current_order: return
            
            # Simple state machine logic
            # 1. Pan needed?
            needs_cook = any(INGREDIENT_INFO[i]['cook'] for i in self.current_order.required)
            cooker = self.cookers[0] if self.cookers else None
            
            has_pan = False
            if cooker:
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    has_pan = True
            
            if needs_cook and not has_pan:
                task.state = BotState.BUY_PAN
                return

            # 2. Plate needed?
            if not self.plate_on_assembly:
                task.state = BotState.BUY_PLATE
                return
            
            # 3. Next Ingredient
            needed = [i for i in self.current_order.required if i not in self.ingredients_on_plate]
            if not needed:
                task.state = BotState.PICKUP_PLATE
                return
            
            next_ing = needed[0]
            # Prioritize cook
            for i in needed:
                if INGREDIENT_INFO[i]['cook']:
                    next_ing = i
                    break
            
            task.item = next_ing
            info = INGREDIENT_INFO[next_ing]
            
            if info['cook']:
                # Check if cooking
                if cooker:
                    tile = controller.get_tile(team, cooker[0], cooker[1])
                    if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                        pan = tile.item
                        if pan.food:
                            if pan.food.cooked_stage == 1:
                                task.state = BotState.TAKE_FROM_PAN
                                return
                            elif pan.food.cooked_stage == 0:
                                return # Wait
                        else:
                            task.state = BotState.BUY_INGREDIENT
                    else:
                        task.state = BotState.BUY_INGREDIENT
            elif info['chop']:
                task.state = BotState.BUY_INGREDIENT
            else:
                task.state = BotState.BUY_INGREDIENT
        
        # Execute states (copy-paste simplified logic moves)
        # BUY / PLACE / ETC (Minimal implementation for brevity as this is a variant)
        # ... logic consistent with champion ...
        # (Implementing simplified "move & do" for actions)
        
        # Buy Pan
        if task.state == BotState.BUY_PAN:
            shop = self._get_nearest((bx,by), self.shops)
            if self._move_toward(controller, bot_id, shop, team):
                if controller.get_team_money(team) >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    task.state = BotState.PLACE_PAN
                    task.target = self.cookers[0]
        elif task.state == BotState.PLACE_PAN:
             if self._move_toward(controller, bot_id, task.target, team):
                 controller.place(bot_id, task.target[0], task.target[1])
                 task.state = BotState.IDLE

        elif task.state == BotState.BUY_PLATE:
            shop = self._get_nearest((bx,by), self.shops)
            if self._move_toward(controller, bot_id, shop, team):
                if controller.get_team_money(team) >= ShopCosts.PLATE.buy_cost:
                    controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
                    task.state = BotState.PLACE_PLATE
                    task.target = self.assembly_counter
        elif task.state == BotState.PLACE_PLATE:
            if self._move_toward(controller, bot_id, task.target, team):
                controller.place(bot_id, task.target[0], task.target[1])
                self.plate_on_assembly = True
                task.state = BotState.IDLE

        elif task.state == BotState.BUY_INGREDIENT:
            shop = self._get_nearest((bx,by), self.shops)
            if self._move_toward(controller, bot_id, shop, team):
                ft = getattr(FoodType, task.item)
                if controller.get_team_money(team) >= ft.buy_cost:
                    controller.buy(bot_id, ft, shop[0], shop[1])
                    info = INGREDIENT_INFO[task.item]
                    if info['chop']:
                        task.state = BotState.PLACE_FOR_CHOP
                        task.target = self.work_counter
                    elif info['cook']:
                        task.state = BotState.START_COOK
                        task.target = self.cookers[0]
                    else:
                        task.state = BotState.ADD_TO_PLATE
                        task.target = self.assembly_counter
        
        elif task.state == BotState.PLACE_FOR_CHOP:
            if self._move_toward(controller, bot_id, task.target, team):
                controller.place(bot_id, task.target[0], task.target[1])
                task.state = BotState.CHOP
        
        elif task.state == BotState.CHOP:
            if self._move_toward(controller, bot_id, task.target, team):
                controller.chop(bot_id, task.target[0], task.target[1])
                task.state = BotState.PICKUP_CHOPPED
        
        elif task.state == BotState.PICKUP_CHOPPED:
             if self._move_toward(controller, bot_id, task.target, team):
                controller.pickup(bot_id, task.target[0], task.target[1])
                info = INGREDIENT_INFO[task.item]
                if info['cook']:
                    task.state = BotState.START_COOK
                    task.target = self.cookers[0]
                else:
                    task.state = BotState.ADD_TO_PLATE
                    task.target = self.assembly_counter
        
        elif task.state == BotState.START_COOK:
            if self._move_toward(controller, bot_id, task.target, team):
                controller.place(bot_id, task.target[0], task.target[1])
                task.state = BotState.IDLE # Go do something else while waiting?
        
        elif task.state == BotState.TAKE_FROM_PAN:
            if self._move_toward(controller, bot_id, task.target or self.cookers[0], team):
                controller.take_from_pan(bot_id, task.target[0], task.target[1])
                task.state = BotState.ADD_TO_PLATE
                task.target = self.assembly_counter

        elif task.state == BotState.ADD_TO_PLATE:
            if self._move_toward(controller, bot_id, task.target, team):
                if controller.add_food_to_plate(bot_id, task.target[0], task.target[1]):
                    self.ingredients_on_plate.append(task.item)
                    task.state = BotState.IDLE
        
        elif task.state == BotState.PICKUP_PLATE:
             if self._move_toward(controller, bot_id, self.assembly_counter, team):
                 controller.pickup(bot_id, self.assembly_counter[0], self.assembly_counter[1])
                 self.plate_on_assembly = False
                 task.state = BotState.SUBMIT
                 task.target = self.submits[0]
        
        elif task.state == BotState.SUBMIT:
            if self._move_toward(controller, bot_id, task.target, team):
                controller.submit(bot_id, task.target[0], task.target[1])
                self.current_order = None
                self.ingredients_on_plate = []
                task.state = BotState.IDLE

    def play_turn(self, controller: RobotController):
        if not self.initialized: self._init_map(controller, controller.get_team())
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        if not bots: return
        
        # PRIMARY BOT (0) - Standard cooking
        self._execute_primary(controller, bots[0], team)
        
        # AGGRESSIVE STEALER BOT (1)
        if len(bots) > 1:
            bot2 = bots[1]
            task = self.bot_tasks.get(bot2, BotTask(BotState.SABOTAGE_STEAL_PAN))
            self.bot_tasks[bot2] = task
            
            # Switch Logic (Turn 60)
            if not self.has_switched and controller.can_switch_maps() and controller.get_turn() > 60:
                controller.switch_maps()
                self.has_switched = True
                task.state = BotState.SABOTAGE_STEAL_PAN
            
            # If on enemy map, wreak havoc
            bot_state = controller.get_bot_state(bot2)
            if bot_state and bot_state.get('map_team') != team:
                self._execute_sabotage(controller, bot2, team, task)
            else:
                # Still on home map? Just wait or wiggle
                pass
