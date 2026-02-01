"""
Single-Active-Bot Controller (v3 - Multi-Ingredient Optimized)
===============================================================

STRICT RULE: Only ONE bot issues any actions per turn.
When blocked, control swaps to the other bot with goal state preserved.

KEY FIXES:
- Process ALL chop-required ingredients before getting plate
- Store chopped items on counters, track their locations
- Only go to GET_PLATE once all pre-processing is done
- Properly track plate_counter throughout the process
"""

from collections import deque
from typing import Optional, List, Dict, Tuple, Set, Any
import os

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
    from tiles import Box, Cooker, Counter
except ImportError:
    pass


INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True},
}

# Enable/disable logging
DEBUG_LOG_ENABLED = True
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "bot_debug.log")


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.team = None
        
        # Map feature caches (positions only, NOT walkability)
        self.shops: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        
        # Single-bot control
        self.active_bot_id: Optional[int] = None
        self.inactive_bot_id: Optional[int] = None
        self.goal: Optional[Dict[str, Any]] = None
        self.goal_counter: int = 0

        # Dynamic Ingredient Info
        # Populate from FoodType enum
        self.ingredient_info: Dict[str, Dict] = {}
        for food_type_name in dir(FoodType):
            if food_type_name.startswith('_'): continue
            ft = getattr(FoodType, food_type_name)
            if isinstance(ft, FoodType):
                self.ingredient_info[ft.food_name] = {
                    'cost': ft.buy_cost,
                    'chop': ft.can_chop,
                    'cook': ft.can_cook
                }
        self._log(f"INIT: Discovered ingredients: {self.ingredient_info}")

        
        # Cleanup tracking
        self.abandoned_plate: Optional[Tuple[int, int]] = None
        
        # Track completed orders to avoid re-selecting them
        self.completed_order_ids: Set[int] = set()
        
        # Inactive bot as mobile storage
        # When counters are full, the inactive bot can temporarily hold an ingredient
        # Format: {'ingredient': str, 'for_order_id': int, 'pickup_tile': (x,y) or None}
        self.inactive_bot_storage: Optional[Dict[str, Any]] = None
        
        # Debug logging
        self.log_initialized = False
        self.last_action = None
    
    def _log(self, message: str) -> None:
        """Write debug message to log file."""
        if not DEBUG_LOG_ENABLED:
            return
        try:
            with open(DEBUG_LOG_PATH, 'a') as f:
                f.write(message + '\n')
        except Exception:
            pass
    
    def _init_log(self) -> None:
        """Initialize log file (clear previous content)."""
        if not DEBUG_LOG_ENABLED or self.log_initialized:
            return
        try:
            with open(DEBUG_LOG_PATH, 'w') as f:
                f.write("=== Bot Debug Log ===\n\n")
            self.log_initialized = True
        except Exception:
            pass
    
    def _log_state(self, controller: RobotController, team: Team, action: str = "") -> None:
        """Log comprehensive state information."""
        if not DEBUG_LOG_ENABLED:
            return
        
        turn = controller.get_turn()
        money = controller.get_team_money(team)
        
        lines = [
            f"\n{'='*60}",
            f"TURN {turn} | Money: ${money}",
            f"{'='*60}",
        ]
        
        # Bot positions
        bots = controller.get_team_bot_ids(team)
        for bot_id in bots:
            bot = controller.get_bot_state(bot_id)
            pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            holding_str = "nothing"
            if holding:
                if holding.get('type') == 'Food':
                    holding_str = f"{holding.get('food_name')}"
                    if holding.get('chopped'):
                        holding_str += " (chopped)"
                    if holding.get('cooked_stage', 0) >= 1:
                        holding_str += f" (cooked:{holding.get('cooked_stage')})"
                else:
                    holding_str = holding.get('type', 'unknown')
            
            active_marker = " [ACTIVE]" if bot_id == self.active_bot_id else " [inactive]"
            lines.append(f"  Bot {bot_id}{active_marker}: pos={pos}, holding={holding_str}")
        
        # Current goal
        if self.goal:
            order = self.goal.get('order', {})
            order_id = order.get('order_id', '?')
            required = order.get('required', [])
            expires = order.get('expires_turn', 0)
            state = self.goal.get('state', 'UNKNOWN')
            
            lines.append(f"\nGOAL: Order #{order_id}")
            lines.append(f"  Required: {required}")
            lines.append(f"  Expires: turn {expires} ({expires - turn} turns left)")
            lines.append(f"  State: {state}")
            
            # Goal details
            plate_counter = self.goal.get('plate_counter')
            plate_contents = self.goal.get('plate_contents', [])
            cooking_items = self.goal.get('cooking_items', {})
            stored_items = self.goal.get('stored_items', {})
            current_ing = self.goal.get('current_ingredient')
            
            lines.append(f"  Plate counter: {plate_counter}")
            lines.append(f"  Plate contents: {plate_contents}")
            lines.append(f"  Cooking: {cooking_items}")
            lines.append(f"  Stored: {stored_items}")
            if current_ing:
                lines.append(f"  Current ingredient: {current_ing}")
            
            # Queues
            chop_cook = self.goal.get('chop_cook_queue', [])
            chop_only = self.goal.get('chop_only_queue', [])
            cook_only = self.goal.get('cook_only_queue', [])
            simple = self.goal.get('simple_queue', [])
            if chop_cook or chop_only or cook_only or simple:
                lines.append(f"  Queues: chop_cook={chop_cook}, chop_only={chop_only}, cook_only={cook_only}, simple={simple}")
        else:
            lines.append("\nGOAL: None (selecting order or preparing)")
        
        # Cooker status
        if self.cookers:
            lines.append("\nCOOKER STATUS:")
            for cooker_pos in self.cookers:
                tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                if tile:
                    pan = getattr(tile, 'item', None)
                    cook_progress = getattr(tile, 'cook_progress', 0)
                    if isinstance(pan, Pan):
                        if pan.food:
                            lines.append(f"  {cooker_pos}: {pan.food.food_name} (stage={pan.food.cooked_stage}, progress={cook_progress})")
                        else:
                            lines.append(f"  {cooker_pos}: empty pan")
                    else:
                        lines.append(f"  {cooker_pos}: no pan")
        
        # Inactive bot storage status
        if self.inactive_bot_storage:
            storage = self.inactive_bot_storage
            lines.append(f"\nSTORAGE BOT: holding {storage.get('ingredient')} for order #{storage.get('for_order_id')}")
            lines.append(f"  State: {storage.get('state')}, pickup_tile: {storage.get('pickup_tile')}")
        
        # Action taken
        if action:
            lines.append(f"\nACTION: {action}")
        
        self._log('\n'.join(lines))
    
    def _init_map(self, controller: RobotController, team: Team) -> None:
        """Cache map feature positions (NOT walkability)."""
        m = controller.get_map(team)
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                pos = (x, y)
                name = tile.tile_name
                if name == "SHOP":
                    self.shops.append(pos)
                elif name == "COOKER":
                    self.cookers.append(pos)
                elif name == "COUNTER":
                    self.counters.append(pos)
                elif name == "SUBMIT":
                    self.submits.append(pos)
                elif name == "TRASH":
                    self.trashes.append(pos)
                elif name == "SINK":
                    self.sinks.append(pos)
                elif name == "SINKTABLE":
                    self.sink_tables.append(pos)
        self.initialized = True
        
        # Log detected map features
        # Scan map for Boxes (which we treat as counters)
        my_map = controller.get_map(team)
        width = my_map.width
        height = my_map.height
        for x in range(width):
            for y in range(height):
                tile = controller.get_tile(team, x, y)
                if tile and tile.tile_name == "BOX":
                    self.counters.append((x, y))
                    
        self._log(f"MAP INIT: counters={self.counters}")
        self._log(f"MAP INIT: cookers={self.cookers}")
        self._log(f"MAP INIT: shops={self.shops}")
        self._log(f"MAP INIT: submits={self.submits}")
        self._log(f"MAP INIT: trashes={self.trashes}")
        self._log(f"MAP INIT: sink_tables={self.sink_tables}")
    
    def _get_bot_pos(self, controller: RobotController, bot_id: int) -> Tuple[int, int]:
        """Get bot position as tuple."""
        bot = controller.get_bot_state(bot_id)
        return (bot['x'], bot['y'])
    
    def _chebyshev_dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Chess king distance."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    def _get_adjacent_walkable(self, target: Tuple[int, int], game_map) -> Set[Tuple[int, int]]:
        """Get walkable positions adjacent to target (Chebyshev dist <= 1)."""
        adj = set()
        tx, ty = target
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = tx + dx, ty + dy
                if game_map.is_tile_walkable(nx, ny):
                    adj.add((nx, ny))
        return adj
    
    def _get_other_bot_pos(self, controller: RobotController) -> Optional[Tuple[int, int]]:
        """Get the position of the inactive bot."""
        if self.inactive_bot_id is None:
            return None
        return self._get_bot_pos(controller, self.inactive_bot_id)
    
    def _bfs_path(self, start: Tuple[int, int], goal_adjacent: Set[Tuple[int, int]], 
                  occupied: Set[Tuple[int, int]], game_map) -> Optional[List[Tuple[int, int]]]:
        """BFS from start to ANY position in goal_adjacent set."""
        if start in goal_adjacent:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            curr, path = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if nxt in visited or nxt in occupied:
                        continue
                    if not game_map.is_tile_walkable(nxt[0], nxt[1]):
                        continue
                    new_path = path + [(dx, dy)]
                    if nxt in goal_adjacent:
                        return new_path
                    visited.add(nxt)
                    queue.append((nxt, new_path))
        return None
    
    def _bfs_reachable_positions(self, start: Tuple[int, int], occupied: Set[Tuple[int, int]], 
                                  game_map) -> Set[Tuple[int, int]]:
        """Get all positions reachable from start."""
        reachable = {start}
        queue = deque([start])
        
        while queue:
            curr = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if nxt in reachable or nxt in occupied:
                        continue
                    if not game_map.is_tile_walkable(nxt[0], nxt[1]):
                        continue
                    reachable.add(nxt)
                    queue.append(nxt)
        return reachable
    
    def _move_toward(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int, int], team: Team) -> bool:
        """Move bot toward target. Returns True if already adjacent to target."""
        bot_pos = self._get_bot_pos(controller, bot_id)
        
        if self._chebyshev_dist(bot_pos, target) <= 1:
            return True
        
        game_map = controller.get_map(team)
        occupied = set()
        other_pos = self._get_other_bot_pos(controller)
        if other_pos:
            occupied.add(other_pos)
        
        goal_adjacent = self._get_adjacent_walkable(target, game_map)
        if not goal_adjacent:
            return False
        
        path = self._bfs_path(bot_pos, goal_adjacent, occupied, game_map)
        if path is None or not path:
            return path == []
        
        dx, dy = path[0]
        if controller.can_move(bot_id, dx, dy):
            controller.move(bot_id, dx, dy)
        
        return False
    
    def _is_blocked(self, controller: RobotController, bot_id: int, 
                    target: Tuple[int, int], team: Team) -> bool:
        """Check if bot is blocked from reaching target."""
        bot_pos = self._get_bot_pos(controller, bot_id)
        
        if self._chebyshev_dist(bot_pos, target) <= 1:
            return False
        
        game_map = controller.get_map(team)
        occupied = set()
        other_pos = self._get_other_bot_pos(controller)
        if other_pos:
            occupied.add(other_pos)
        
        goal_adjacent = self._get_adjacent_walkable(target, game_map)
        if not goal_adjacent:
            return True
        
        path = self._bfs_path(bot_pos, goal_adjacent, occupied, game_map)
        return path is None
    
    def _get_nearest(self, pos: Tuple[int, int], 
                     locations: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get nearest location by Chebyshev distance."""
        if not locations:
            return None
        return min(locations, key=lambda p: self._chebyshev_dist(pos, p))
    
    def _get_free_counter(self, controller: RobotController, team: Team, 
                          pos: Tuple[int, int], 
                          exclude: Optional[Set[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """Find nearest free counter."""
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
        return self._get_nearest(pos, free)
    
    def _get_available_cooker(self, controller: RobotController, team: Team,
                               pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest cooker with empty pan."""
        available = []
        for c in self.cookers:
            tile = controller.get_tile(team, c[0], c[1])
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    available.append(c)
        return self._get_nearest(pos, available)
    
    def _item_signature(self, holding: Optional[Dict]) -> Optional[tuple]:
        """Get item signature for matching."""
        if holding is None:
            return None
        if holding.get('type') == 'Food':
            return ('Food', holding.get('food_name'), 
                    bool(holding.get('chopped')), int(holding.get('cooked_stage', 0)))
        if holding.get('type') == 'Plate':
            foods = []
            for f in holding.get('food', []):
                foods.append((f.get('food_name'), bool(f.get('chopped')), 
                             int(f.get('cooked_stage', 0))))
            return ('Plate', bool(holding.get('dirty')), tuple(sorted(foods)))
        if holding.get('type') == 'Pan':
            return ('Pan', self._item_signature(holding.get('food')))
        return (holding.get('type'),)
    
    def _can_place_on_tile(self, controller: RobotController, team: Team, 
                           tile_pos: Tuple[int, int], holding: Dict) -> bool:
        """Check if place() would succeed for this item on this tile."""
        tile = controller.get_tile(team, tile_pos[0], tile_pos[1])
        if tile is None:
            return False
        
        tile_name = tile.tile_name
        
        if tile_name == "COUNTER":
            return getattr(tile, 'item', None) is None
        
        if tile_name == "BOX":
            count = getattr(tile, 'count', 0)
            if count == 0:
                return True
            box_item = getattr(tile, 'item', None)
            if box_item is None:
                return True
            my_sig = self._item_signature(holding)
            box_sig = None
            if isinstance(box_item, Food):
                box_sig = ('Food', box_item.food_name, 
                          bool(box_item.chopped), int(box_item.cooked_stage))
            elif isinstance(box_item, Plate):
                foods = []
                for f in box_item.food:
                    foods.append((f.food_name, bool(f.chopped), int(f.cooked_stage)))
                box_sig = ('Plate', bool(box_item.dirty), tuple(sorted(foods)))
            elif isinstance(box_item, Pan):
                box_sig = ('Pan', None)
            return my_sig == box_sig
        
        if tile_name == "COOKER":
            if holding.get('type') == 'Pan':
                existing_pan = getattr(tile, 'item', None)
                if isinstance(existing_pan, Pan):
                    return existing_pan.food is None
                return True
            if holding.get('type') == 'Food':
                existing_pan = getattr(tile, 'item', None)
                if not isinstance(existing_pan, Pan):
                    return False
                if existing_pan.food is not None:
                    return False
                food_name = holding.get('food_name')
                return self.ingredient_info.get(food_name, {}).get('cook', False)
            return False
        
        if tile_name == "TRASH":
            return False  # Can't place() on trash, must use trash() action
        
        if tile_name == "SUBMIT":
            return holding.get('type') == 'Plate' and not holding.get('dirty')
        
        return False
    
    def _find_handoff_drop_tile(self, controller: RobotController, team: Team,
                                 holding: Dict) -> Optional[Tuple[int, int]]:
        """Find a drop tile reachable by BOTH bots."""
        if self.active_bot_id is None or self.inactive_bot_id is None:
            return None
        
        game_map = controller.get_map(team)
        active_pos = self._get_bot_pos(controller, self.active_bot_id)
        inactive_pos = self._get_bot_pos(controller, self.inactive_bot_id)
        
        active_reachable = self._bfs_reachable_positions(
            active_pos, {inactive_pos}, game_map)
        inactive_reachable = self._bfs_reachable_positions(
            inactive_pos, {active_pos}, game_map)
        
        self._log(f"    Handoff search: active@{active_pos} reachable={len(active_reachable)}, inactive@{inactive_pos} reachable={len(inactive_reachable)}")
        self._log(f"    All counters: {self.counters}")
        
        candidates = []
        # Only counters can be used for handoff drop (not trash - can't place() on it)
        all_placeable = self.counters
        
        for tile_pos in all_placeable:
            can_place = self._can_place_on_tile(controller, team, tile_pos, holding)
            adj_walkable = self._get_adjacent_walkable(tile_pos, game_map)
            active_can_reach = any(p in active_reachable for p in adj_walkable)
            inactive_can_reach = any(p in inactive_reachable for p in adj_walkable)
            
            self._log(f"    Counter {tile_pos}: can_place={can_place}, active_reach={active_can_reach}, inactive_reach={inactive_can_reach}")
            
            if not can_place:
                continue
            
            if active_can_reach and inactive_can_reach:
                dist = self._chebyshev_dist(active_pos, tile_pos)
                candidates.append((dist, tile_pos))
        
        if not candidates:
            self._log(f"    No valid handoff candidates found!")
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _swap_control(self) -> None:
        """Atomically swap active/inactive bot IDs."""
        self.active_bot_id, self.inactive_bot_id = self.inactive_bot_id, self.active_bot_id
    
    def _can_use_inactive_for_storage(self, controller: RobotController, team: Team) -> bool:
        """Check if inactive bot is free to hold an item as mobile storage."""
        if self.inactive_bot_id is None:
            return False
        
        # Check if inactive bot is already holding something
        inactive_bot = controller.get_bot_state(self.inactive_bot_id)
        if inactive_bot.get('holding') is not None:
            return False
        
        # Check if there's already a storage handoff in progress
        if self.inactive_bot_storage is not None:
            return False
        
        # Check if there's a regular handoff in progress
        if self.goal and self.goal.get('handoff_drop_tile') is not None:
            return False
        
        return True
    
    def _initiate_storage_handoff(self, controller: RobotController, team: Team,
                                   ingredient: str, order_id: int) -> bool:
        """
        Start a storage handoff: active bot places item, inactive bot will pick it up.
        Returns True if handoff was initiated (turn ends), False if not possible.
        """
        if not self._can_use_inactive_for_storage(controller, team):
            self._log(f"  STORAGE HANDOFF: Cannot use inactive bot for storage")
            return False
        
        bot = controller.get_bot_state(self.active_bot_id)
        holding = bot.get('holding')
        if holding is None:
            return False
        
        # Find a drop tile reachable by both bots
        drop_tile = self._find_handoff_drop_tile(controller, team, holding)
        if drop_tile is None:
            self._log(f"  STORAGE HANDOFF: No drop tile found")
            return False
        
        bot_pos = self._get_bot_pos(controller, self.active_bot_id)
        
        if self._chebyshev_dist(bot_pos, drop_tile) <= 1:
            if controller.place(self.active_bot_id, drop_tile[0], drop_tile[1]):
                self._log(f"  STORAGE HANDOFF: Placed {ingredient} at {drop_tile} for inactive bot to hold")
                
                # Set up the storage tracking - inactive bot needs to pick this up
                self.inactive_bot_storage = {
                    'ingredient': ingredient,
                    'for_order_id': order_id,
                    'pickup_tile': drop_tile,
                    'state': 'PICKUP'  # Inactive bot needs to pick up
                }
                
                # Swap to inactive bot to pick up
                self._swap_control()
                return True
            else:
                self._log(f"  STORAGE HANDOFF: place() failed at {drop_tile}")
                return False
        else:
            # Move toward drop tile
            self._log(f"  STORAGE HANDOFF: Moving toward {drop_tile}")
            self._move_toward(controller, self.active_bot_id, drop_tile, team)
            return True  # Turn was used for movement
    
    def _handle_inactive_bot_storage(self, controller: RobotController, team: Team) -> bool:
        """
        Handle inactive bot storage operations (when inactive bot is now active).
        Returns True if this consumed the turn.
        """
        if self.inactive_bot_storage is None:
            return False
        
        storage = self.inactive_bot_storage
        state = storage.get('state')
        pickup_tile = storage.get('pickup_tile')
        ingredient = storage.get('ingredient')
        
        bot = controller.get_bot_state(self.active_bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        if state == 'PICKUP':
            # We need to pick up the item from the drop tile
            if pickup_tile is None:
                self._log(f"  STORAGE: No pickup tile set, clearing")
                self.inactive_bot_storage = None
                return False
            
            if self._chebyshev_dist(bot_pos, pickup_tile) <= 1:
                if controller.pickup(self.active_bot_id, pickup_tile[0], pickup_tile[1]):
                    self._log(f"  STORAGE: Picked up {ingredient} from {pickup_tile}")
                    storage['state'] = 'HOLDING'
                    storage['pickup_tile'] = None
                    
                    # Move away from the tile to free up space, then swap back
                    # For now, just swap back - the inactive bot will hold it
                    self._swap_control()
                    return True
                else:
                    self._log(f"  STORAGE: pickup() failed at {pickup_tile}")
                    self.inactive_bot_storage = None
                    self._swap_control()
                    return True
            else:
                # Move toward pickup tile
                self._move_toward(controller, self.active_bot_id, pickup_tile, team)
                return True
        
        elif state == 'HOLDING':
            # Bot is holding the item - check if we need to give it back
            # This state means we're the storage bot and we've been swapped in
            # This shouldn't normally happen unless we need to use the item
            order_id = storage.get('for_order_id')
            
            # Check if current goal matches
            if self.goal and self.goal.get('order', {}).get('order_id') == order_id:
                # We need this ingredient! Place it for the active bot
                self._log(f"  STORAGE: Order #{order_id} needs {ingredient}, returning it")
                
                # Find a counter to place it on
                counter = self._get_free_counter(controller, team, bot_pos)
                if counter:
                    if self._chebyshev_dist(bot_pos, counter) <= 1:
                        if controller.place(self.active_bot_id, counter[0], counter[1]):
                            self._log(f"  STORAGE: Placed {ingredient} at {counter}")
                            # Update the goal's stored_items
                            if self.goal:
                                stored = self.goal.get('stored_items', {})
                                if ingredient not in stored:
                                    stored[ingredient] = []
                                stored[ingredient].append(counter)
                                self.goal['stored_items'] = stored
                            self.inactive_bot_storage = None
                            self._swap_control()
                            return True
                    else:
                        self._move_toward(controller, self.active_bot_id, counter, team)
                        return True
                else:
                    # No counter - swap back for now, try later
                    self._swap_control()
                    return True
            else:
                # Goal doesn't match - just swap back, keep holding
                self._swap_control()
                return True
        
        return False
    
    def _handle_blocked(self, controller: RobotController, team: Team) -> None:
        """Handle blocked state - handoff or swap."""
        bot = controller.get_bot_state(self.active_bot_id)
        holding = bot.get('holding')
        
        if holding is not None:
            drop_tile = self._find_handoff_drop_tile(controller, team, holding)
            self._log(f"  HANDOFF: holding={holding.get('type', 'item')}, drop_tile={drop_tile}")
            if drop_tile is not None:
                bot_pos = self._get_bot_pos(controller, self.active_bot_id)
                
                if self._chebyshev_dist(bot_pos, drop_tile) <= 1:
                    if controller.place(self.active_bot_id, drop_tile[0], drop_tile[1]):
                        self._log(f"  HANDOFF: Placed item at {drop_tile}, swapping control")
                        if self.goal is None:
                            self.goal = {}
                        self.goal['handoff_drop_tile'] = drop_tile
                        self._swap_control()
                    else:
                        self._log(f"  HANDOFF: place() FAILED at {drop_tile}")
                        self._swap_control()
                else:
                    self._log(f"  HANDOFF: Moving toward drop tile {drop_tile}")
                    self._move_toward(controller, self.active_bot_id, drop_tile, team)
                return
            else:
                self._log(f"  HANDOFF: No valid drop tile found, swapping anyway (item will be lost!)")
        
        self._swap_control()
    
    def _estimate_completion_turn(self, order: Dict, current_turn: int, 
                                    controller: RobotController, team: Team) -> int:
        """Estimate the earliest turn we can complete this order.
        
        Uses REALISTIC estimates including movement time.
        Returns a very high number only if truly impossible.
        """
        required = order['required']
        expires = order.get('expires_turn', current_turn + 9999)
        starts_at = order.get('created_turn', 0)
        
        # Can't submit before order is active
        earliest_submit = max(current_turn, starts_at)
        
        # Get bot position for movement estimates
        bot_pos = self._get_bot_pos(controller, self.active_bot_id) if self.active_bot_id else (0, 0)
        
        # Check current goal progress (if working on same order)
        current_goal = self.goal
        plate_ready = False
        plate_contents = []
        cooking_items = {}
        stored_items = {}
        plate_counter = None
        
        if current_goal and current_goal.get('order', {}).get('order_id') == order.get('order_id'):
            plate_contents = current_goal.get('plate_contents', [])
            cooking_items = current_goal.get('cooking_items', {})
            stored_items = current_goal.get('stored_items', {})
            plate_counter = current_goal.get('plate_counter')
            plate_ready = plate_counter is not None
        
        # Calculate time needed - REALISTIC estimates including movement
        total_time = 0
        max_cook_time = 0
        
        # Average movement cost between key locations (approximate)
        # On this map, features are spread out ~5-8 tiles apart
        AVG_MOVE = 5
        
        # If no plate yet, need time to get one
        if not plate_ready:
            # Move to sinktable/shop + buy/get plate + move to counter + place
            if self.sink_tables:
                move_to_plate = self._chebyshev_dist(bot_pos, self.sink_tables[0])
            elif self.shops:
                move_to_plate = self._chebyshev_dist(bot_pos, self.shops[0])
            else:
                move_to_plate = AVG_MOVE
            total_time += move_to_plate + 1  # Move + get plate
            total_time += AVG_MOVE + 1  # Move to counter + place
        
        plate_contents_copy = list(plate_contents)
        for ing in required:
            if ing in plate_contents_copy:
                plate_contents_copy.remove(ing)
                continue  # Already on plate
            
            info = INGREDIENT_INFO.get(ing, {})
            needs_chop = info.get('chop', False)
            needs_cook = info.get('cook', False)
            
            # Check if already cooking
            if ing in cooking_items:
                # Move to cooker + take + move to plate + add
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1
                continue
            
            # Check if already stored (chopped on counter)
            if ing in stored_items:
                # Move to stored + pickup + move to plate + add
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1
                continue
            
            # Need full processing - realistic estimates
            if needs_chop and needs_cook:
                # MEAT: move to shop + buy + move to counter + place + chop(6) + pickup
                #       + move to cooker + place + wait for cook + take + move to plate + add
                total_time += AVG_MOVE + 1  # Buy
                total_time += AVG_MOVE + 1 + 6 + 1  # Place, chop, pickup
                total_time += AVG_MOVE + 1  # Place in cooker
                max_cook_time = max(max_cook_time, 20)
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1  # Take from pan + add to plate
            elif needs_chop:
                # ONIONS: buy + move + place + chop + pickup + move to plate + add
                total_time += AVG_MOVE + 1  # Buy
                total_time += AVG_MOVE + 1 + 6 + 1  # Place, chop, pickup
                total_time += AVG_MOVE + 1  # Move to plate + add
            elif needs_cook:
                # EGG: buy + move to cooker + place + wait + take + move to plate + add
                total_time += AVG_MOVE + 1  # Buy
                total_time += AVG_MOVE + 1  # Place in cooker
                max_cook_time = max(max_cook_time, 20)
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1  # Take + add
            else:
                # NOODLES, SAUCE: buy + move to plate + add
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1
        
        # Pickup plate + move to submit + submit
        total_time += 1 + AVG_MOVE + 1
        
        # Cooking happens in parallel - only add if it's the bottleneck
        effective_cook_wait = max(0, max_cook_time - total_time)
        total_time += effective_cook_wait
        
        # Earliest completion turn
        completion_turn = earliest_submit + total_time
        
        # Only return impossible if WAY over (give broad buffer for luck/support)
        # On small_wall, orders are tight (50 turns), so we must try even if check says 51
        if completion_turn > expires + 25:
            return 99999
        
        return completion_turn
    
    def _get_order_difficulty(self, order: Dict) -> int:
        """Calculate difficulty score for an order (lower = easier).
        
        Scoring:
        - Simple ingredients (NOODLES, SAUCE): +1 each
        - Cook-only (EGG): +3 each
        - Chop-only (ONIONS): +3 each
        - Chop+cook (MEAT): +5 each
        """
        difficulty = 0
        for ing in order.get('required', []):
            info = self.ingredient_info.get(ing, {})
            needs_chop = info.get('chop', False)
            needs_cook = info.get('cook', False)
            
            if needs_chop and needs_cook:
                difficulty += 5  # MEAT
            elif needs_chop or needs_cook:
                difficulty += 3  # EGG, ONIONS
            else:
                difficulty += 1  # NOODLES, SAUCE
        return difficulty
    
    def _select_order(self, controller: RobotController, team: Team) -> Optional[Dict]:
        """Select the best order to work on.
        
        Strategy:
        1. Only consider completable orders (completion_turn < 99999)
        2. Among completable orders, prioritize by:
           a. Orders starting soon (within 30 turns) - urgency window
           b. Easier orders (fewer/simpler ingredients)
           c. Earlier completion time
           d. Higher reward as tiebreaker
        3. If no completable orders, pick backup with most time left
        """
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        candidates = []  # List of (score, order) tuples
        
        # Also track best "backup" order in case all seem impossible
        backup = None
        backup_time_left = 0
        
        for order in orders:
            order_id = order.get('order_id')
            expires = order.get('expires_turn', 0)
            starts_at = order.get('created_turn', 0)
            
            # Skip already completed orders (might still be in list briefly)
            if order_id in self.completed_order_ids:
                continue
            
            # Skip already expired orders
            if expires <= current_turn:
                continue
            
            # Skip orders too far in the future (more than 100 turns away)
            if starts_at > current_turn + 100:
                continue
            
            completion_turn = self._estimate_completion_turn(order, current_turn, controller, team)
            reward = order.get('reward', 0)
            time_left = expires - current_turn
            difficulty = self._get_order_difficulty(order)
            
            if completion_turn < 99999:
                # This order is completable - calculate priority score
                # Lower score = better priority
                
                # Urgency: orders starting within 30 turns get priority
                turns_until_start = max(0, starts_at - current_turn)
                urgency_bonus = 0 if turns_until_start <= 0 else (1 if turns_until_start <= 30 else 1000)
                
                # Primary: difficulty (easier orders first)
                # Secondary: completion time (faster orders first)
                # Tertiary: reward (higher reward as tiebreaker, negative so lower is better)
                score = (urgency_bonus, difficulty, completion_turn, -reward)
                candidates.append((score, order))
            else:
                # Track as backup - prefer orders with more time left
                if time_left > backup_time_left:
                    backup_time_left = time_left
                    backup = order
        
        # Sort by score (lower is better)
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_order = candidates[0][1]
            self._log(f"  ORDER SELECTION: Chose #{best_order.get('order_id')} with score {candidates[0][0]}")
            return best_order
        
        # Otherwise, try the backup (might get lucky, or at least make progress)
        return backup
    
    def _create_goal(self, order: Dict) -> Dict[str, Any]:
        """Create a new goal state for an order."""
        self.goal_counter += 1
        
        # Categorize ingredients
        chop_cook = []   # MEAT: chop then cook
        chop_only = []   # ONIONS: chop, store on counter
        cook_only = []   # EGG: cook
        simple = []      # NOODLES, SAUCE: just add to plate
        
        for ing in order['required']:
            info = self.ingredient_info.get(ing, {})
            needs_chop = info.get('chop', False)
            needs_cook = info.get('cook', False)
            
            if needs_chop and needs_cook:
                chop_cook.append(ing)
            elif needs_chop:
                chop_only.append(ing)
            elif needs_cook:
                cook_only.append(ing)
            else:
                simple.append(ing)
        
        return {
            'goal_id': self.goal_counter,
            'goal_type': 'deliver_order',
            'order': order,
            'state': 'INIT',
            'plate_counter': None,
            'work_counter': None,
            'handoff_drop_tile': None,
            # Ingredient queues
            'chop_cook_queue': chop_cook,      # Chop then cook
            'chop_only_queue': chop_only,      # Chop then store
            'cook_only_queue': cook_only,      # Just cook
            'simple_queue': simple,            # Just add to plate
            # Tracking
            'stored_items': {},                # {ingredient_name: counter_pos}
            'cooking_items': {},               # {ingredient_name: cooker_pos}
            'plate_contents': [],              # Items already on plate
            'current_ingredient': None,
        }
    
    def _execute_goal(self, controller: RobotController, team: Team) -> bool:
        """Execute the current goal pipeline. Returns True if blocked."""
        if self.goal is None or self.active_bot_id is None:
            return False
        
        bot = controller.get_bot_state(self.active_bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        order = self.goal.get('order')
        state = self.goal.get('state', 'INIT')
        old_state = state  # Track state changes
        
        if order is None:
            return False
        
        shop = self._get_nearest(bot_pos, self.shops)
        submit = self._get_nearest(bot_pos, self.submits)
        
        # Log current execution context
        holding_str = "nothing"
        if holding:
            if holding.get('type') == 'Food':
                holding_str = holding.get('food_name', 'food')
            else:
                holding_str = holding.get('type', 'item')
        self._log(f"  Executing state={state}, pos={bot_pos}, holding={holding_str}, money=${money}")
        
        # === STATE: INIT ===
        # Decide what to process next
        if state == 'INIT':
            chop_cook = self.goal.get('chop_cook_queue', [])
            chop_only = self.goal.get('chop_only_queue', [])
            cook_only = self.goal.get('cook_only_queue', [])
            
            # Check if inactive bot is holding an ingredient we need
            if self.inactive_bot_storage is not None:
                storage = self.inactive_bot_storage
                storage_ing = storage.get('ingredient')
                storage_order_id = storage.get('for_order_id')
                storage_state = storage.get('state')
                
                # If it's for this order and in HOLDING state
                if storage_order_id == order.get('order_id') and storage_state == 'HOLDING':
                    # Check if this ingredient is still needed
                    needed_ings = chop_cook + chop_only + cook_only + self.goal.get('simple_queue', [])
                    
                    if storage_ing in needed_ings:
                        # We need this ingredient! Swap to get it back
                        self._log(f"  INIT: Inactive bot holding {storage_ing}, swapping to retrieve it")
                        self._swap_control()
                        return False
            
            # Check stored items - if ingredient is already stored, go pick it up
            stored_items = self.goal.get('stored_items', {})
            
            # First priority: chop+cook items (MEAT)
            if chop_cook:
                ing = chop_cook[0]
                if ing in stored_items and stored_items[ing]:
                    # Already stored - go pick it up for cooking
                    # Pick the nearest one
                    
                    # Remove from stored list since we are picking it up
                    # We'll just remove the first one for now
                    counter = None
                    if stored_items[ing]:
                        counter = stored_items[ing].pop(0)
                        if not stored_items[ing]:
                            del stored_items[ing]
                    self.goal['stored_items'] = stored_items
                    
                    self._log(f"  INIT: {ing} already stored at {counter}, picking up")
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'BUY_INGREDIENT'
                    self.goal['next_after_buy'] = 'CHOP_THEN_COOK'
                return False
            
            # Second: chop-only items (ONIONS)
            if chop_only:
                ing = chop_only[0]
                if ing in stored_items and stored_items[ing]:
                    counter = stored_items[ing].pop(0)
                    if not stored_items[ing]:
                        del stored_items[ing]
                    self.goal['stored_items'] = stored_items

                    self._log(f"  INIT: {ing} already stored at {counter}, picking up")
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'BUY_INGREDIENT'
                    self.goal['next_after_buy'] = 'CHOP_THEN_STORE'
                return False
            
            # Third: cook-only items (EGG)
            if cook_only:
                ing = cook_only[0]
                if ing in stored_items and stored_items[ing]:
                    counter = stored_items[ing].pop(0)
                    if not stored_items[ing]:
                        del stored_items[ing]
                    self.goal['stored_items'] = stored_items

                    self._log(f"  INIT: {ing} already stored at {counter}, picking up")
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    self.goal['current_ingredient'] = ing
                    self.goal['state'] = 'BUY_INGREDIENT'
                    self.goal['next_after_buy'] = 'COOK_ONLY'
                return False
            
            # All pre-processing done - now handle plate and assembly
            self.goal['state'] = 'ENSURE_PLATE'
            return False
        
        # === STATE: BUY_INGREDIENT ===
        if state == 'BUY_INGREDIENT':
            ing = self.goal.get('current_ingredient')
            next_state = self.goal.get('next_after_buy', 'INIT')
            
            if ing is None:
                self.goal['state'] = 'INIT'
                return False
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                self.goal['state'] = next_state
                return False
            
            if holding:
                # Wrong item - trash it
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._is_blocked(controller, self.active_bot_id, trash, team):
                            return True
                        if self._move_toward(controller, self.active_bot_id, trash, team):
                            controller.trash(self.active_bot_id, trash[0], trash[1])
                return False
            
            if shop is None:
                return True
            if self._is_blocked(controller, self.active_bot_id, shop, team):
                return True
            if self._move_toward(controller, self.active_bot_id, shop, team):
                food_type = getattr(FoodType, ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(self.active_bot_id, food_type, shop[0], shop[1])
            return False
        
        # === STATE: CHOP_THEN_COOK (for MEAT) ===
        if state == 'CHOP_THEN_COOK':
            ing = self.goal.get('current_ingredient')
            work_counter = self.goal.get('work_counter')
            
            # If holding the ingredient, place it for chopping
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                if holding.get('chopped'):
                    # Already chopped - start cooking
                    self.goal['state'] = 'START_COOK'
                    return False
                
                # Need to place for chopping
                # Flatten exclude list
                exclude = set()
                for locs in self.goal.get('stored_items', {}).values():
                    if isinstance(locs, list):
                        exclude.update(locs)
                    else:
                        exclude.add(locs)
                if self.goal.get('plate_counter'):
                    exclude.add(self.goal['plate_counter'])
                
                counter = self._get_free_counter(controller, team, bot_pos, exclude)
                if counter is None:
                    return True
                
                if self._is_blocked(controller, self.active_bot_id, counter, team):
                    return True
                if self._move_toward(controller, self.active_bot_id, counter, team):
                    controller.place(self.active_bot_id, counter[0], counter[1])
                    self.goal['work_counter'] = counter
                return False
            
            # If work_counter is set, chop the item there
            if work_counter:
                tile = controller.get_tile(team, work_counter[0], work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    food = tile.item
                    if self._is_blocked(controller, self.active_bot_id, work_counter, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, work_counter, team):
                        if food.chopped:
                            controller.pickup(self.active_bot_id, work_counter[0], work_counter[1])
                            self.goal['work_counter'] = None
                        else:
                            controller.chop(self.active_bot_id, work_counter[0], work_counter[1])
                    return False
                else:
                    self.goal['work_counter'] = None
            
            # Don't have the ingredient - buy it
            self.goal['state'] = 'BUY_INGREDIENT'
            self.goal['next_after_buy'] = 'CHOP_THEN_COOK'
            return False
        
        # === STATE: PICKUP_STORED_FOR_COOK ===
        # Pick up a stored ingredient and go to START_COOK
        if state == 'PICKUP_STORED_FOR_COOK':
            ing = self.goal.get('current_ingredient')
            stored_items = self.goal.get('stored_items', {})
            
            if ing is None or ing not in stored_items:
                # No stored item - go back to INIT
                self._log(f"  PICKUP_STORED: {ing} not in stored items, resetting")
                self.goal['current_ingredient'] = None
                self.goal['state'] = 'INIT'
                return False
            
            stored_list = stored_items[ing]
            
            # Check if we already picked it up
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                # Already holding it - remove from stored and go to START_COOK
                # Remove one instance (the one we presumably picked up)
                if stored_list:
                    stored_list.pop(0)
                    if not stored_list:
                        del stored_items[ing]
                self.goal['stored_items'] = stored_items
                self.goal['state'] = 'START_COOK'
                self._log(f"  PICKUP_STORED: Already holding {ing}, going to START_COOK")
                return False
            
            # Check if item is still there
            tile = controller.get_tile(team, stored_pos[0], stored_pos[1])
            tile_item = getattr(tile, 'item', None) if tile else None
            
            if tile_item is None:
                # Item is gone - remove from stored and go back to INIT
                self._log(f"  PICKUP_STORED: {ing} no longer at {stored_pos}")
                del stored_items[ing]
                self.goal['stored_items'] = stored_items
                self.goal['current_ingredient'] = None
                self.goal['state'] = 'INIT'
                return False
            
            # Go pick it up
            if self._is_blocked(controller, self.active_bot_id, stored_pos, team):
               stored_pos = stored_list[0]
            
            # Navigate to it
            bot_pos = self._get_bot_pos(controller, self.active_bot_id)
            if self._chebyshev_dist(bot_pos, stored_pos) <= 1:
                # If it's a Box, and empty, we can't pickup?
                # But we only put in stored_items if we placed something.
                if controller.pickup(self.active_bot_id, stored_pos[0], stored_pos[1]):
                    self._log(f"  PICKUP_STORED: Picked up {ing} from {stored_pos}")
                    # Remove from stored
                    stored_list.pop(0)
                    if not stored_list:
                        del stored_items[ing]
                    self.goal['stored_items'] = stored_items
                    self.goal['state'] = 'START_COOK'
                    return True
                else:
                    self._log(f"  PICKUP_STORED: pickup() failed at {stored_pos}, removing from memory")
                    # Assume it's gone/taken
                    stored_list.pop(0)
                    if not stored_list:
                        del stored_items[ing]
                    self.goal['stored_items'] = stored_items
                    # Retry
                    return True
        
        # === STATE: START_COOK ===
        if state == 'START_COOK':
            ing = self.goal.get('current_ingredient')
            
            if not holding or holding.get('type') != 'Food':
                self.goal['state'] = 'INIT'
                return False
            
            cooker = self._get_available_cooker(controller, team, bot_pos)
            if cooker is None:
                # No available cooker - check if something we need is already cooking
                for cooker_pos in self.cookers:
                    tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food is not None:
                            cooking_name = pan.food.food_name
                            if cooking_name == ing:
                                # Our ingredient is already cooking! Track it and move on
                                self._log(f"  START_COOK: {ing} already cooking at {cooker_pos}, using that")
                                cooking_items = self.goal.get('cooking_items', {})
                                cooking_items[ing] = cooker_pos
                                self.goal['cooking_items'] = cooking_items
                                
                                # Remove from queue
                                chop_cook = self.goal.get('chop_cook_queue', [])
                                cook_only = self.goal.get('cook_only_queue', [])
                                if ing in chop_cook:
                                    chop_cook.remove(ing)
                                if ing in cook_only:
                                    cook_only.remove(ing)
                                
                                # Store the one we're holding since we don't need to cook it
                                counter = self._get_free_counter(controller, team, bot_pos)
                                if counter:
                                    if self._move_toward(controller, self.active_bot_id, counter, team):
                                        controller.place(self.active_bot_id, counter[0], counter[1])
                                        stored = self.goal.get('stored_items', {})
                                        stored[ing + '_extra'] = counter
                                        self.goal['stored_items'] = stored
                                else:
                                    # No counter - trash the extra ingredient
                                    if self.trashes:
                                        trash = self._get_nearest(bot_pos, self.trashes)
                                        if trash and self._move_toward(controller, self.active_bot_id, trash, team):
                                            controller.trash(self.active_bot_id, trash[0], trash[1])
                                
                                self.goal['current_ingredient'] = None
                                self.goal['state'] = 'INIT'
                                return False
                
                # Cooker is occupied - check if we can take the cooked food out
                for cooker_pos in self.cookers:
                    tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food is not None and pan.food.cooked_stage >= 1:
                            # Food is done - take it out
                            self._log(f"  START_COOK: Taking cooked food from {cooker_pos} to free cooker")
                            if self._move_toward(controller, self.active_bot_id, cooker_pos, team):
                                controller.take_from_pan(self.active_bot_id, cooker_pos[0], cooker_pos[1])
                            return False
                
                # Cooker still cooking, try to store our ingredient
                self._log(f"  START_COOK: Cooker busy, storing {ing}")
                counter = self._get_free_counter(controller, team, bot_pos)
                if counter:
                    if self._move_toward(controller, self.active_bot_id, counter, team):
                        if controller.place(self.active_bot_id, counter[0], counter[1]):
                            stored = self.goal.get('stored_items', {})
                            stored[ing] = counter
                            self.goal['stored_items'] = stored
                            self.goal['current_ingredient'] = None
                            self.goal['state'] = 'INIT'
                    return False
                
                # No counter available - try using inactive bot as storage
                order_id = order.get('order_id', 0)
                if self._can_use_inactive_for_storage(controller, team):
                    self._log(f"  START_COOK: No counter, using inactive bot as storage for {ing}")
                    if self._initiate_storage_handoff(controller, team, ing, order_id):
                        self.goal['current_ingredient'] = None
                        self.goal['state'] = 'INIT'
                        return False
                
                # Last resort - trash the ingredient
                self._log(f"  START_COOK: No counter or storage bot, trashing {ing}")
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, self.active_bot_id, trash, team):
                            controller.trash(self.active_bot_id, trash[0], trash[1])
                            self.goal['current_ingredient'] = None
                            self.goal['state'] = 'INIT'
                        return False
                return False
            
            if self._is_blocked(controller, self.active_bot_id, cooker, team):
                return True
            if self._move_toward(controller, self.active_bot_id, cooker, team):
                if controller.place(self.active_bot_id, cooker[0], cooker[1]):
                    self._log(f"  STARTED COOKING: {ing} at {cooker}")
                    # Track what's cooking where
                    cooking_items = self.goal.get('cooking_items', {})
                    cooking_items[ing] = cooker
                    self.goal['cooking_items'] = cooking_items
                    
                    # Remove from queue
                    chop_cook = self.goal.get('chop_cook_queue', [])
                    cook_only = self.goal.get('cook_only_queue', [])
                    if ing in chop_cook:
                        chop_cook.remove(ing)
                    if ing in cook_only:
                        cook_only.remove(ing)
                    
                    self.goal['current_ingredient'] = None
                    self.goal['state'] = 'INIT'
            return False
        
        # === STATE: CHOP_THEN_STORE (for ONIONS) ===
        if state == 'CHOP_THEN_STORE':
            ing = self.goal.get('current_ingredient')
            work_counter = self.goal.get('work_counter')
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                if holding.get('chopped'):
                    # Chopped - store on counter
                    exclude = set()
                    for locs in self.goal.get('stored_items', {}).values():
                        if isinstance(locs, list):
                            exclude.update(locs)
                        else:
                            exclude.add(locs)
                    if self.goal.get('plate_counter'):
                        exclude.add(self.goal['plate_counter'])
                    
                    counter = self._get_free_counter(controller, team, bot_pos, exclude)
                    if counter is None:
                        return True
                    
                    if self._is_blocked(controller, self.active_bot_id, counter, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, counter, team):
                        if controller.place(self.active_bot_id, counter[0], counter[1]):
                            # Track where we stored it
                            stored = self.goal.get('stored_items', {})
                            if ing not in stored:
                                stored[ing] = []
                            stored[ing].append(counter)
                            self.goal['stored_items'] = stored
                            
                            # Remove from queue
                            chop_only = self.goal.get('chop_only_queue', [])
                            if ing in chop_only:
                                chop_only.remove(ing)
                            
                            self.goal['current_ingredient'] = None
                            self.goal['work_counter'] = None
                            self.goal['state'] = 'INIT'
                    return False
                
                # Need to place for chopping
                exclude = set()
                for locs in self.goal.get('stored_items', {}).values():
                    if isinstance(locs, list):
                        exclude.update(locs)
                    else:
                        exclude.add(locs)
                if self.goal.get('plate_counter'):
                    exclude.add(self.goal['plate_counter'])
                
                counter = self._get_free_counter(controller, team, bot_pos, exclude)
                if counter is None:
                    return True
                
                if self._is_blocked(controller, self.active_bot_id, counter, team):
                    return True
                if self._move_toward(controller, self.active_bot_id, counter, team):
                    controller.place(self.active_bot_id, counter[0], counter[1])
                    self.goal['work_counter'] = counter
                return False
            
            if work_counter:
                tile = controller.get_tile(team, work_counter[0], work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    food = tile.item
                    if self._is_blocked(controller, self.active_bot_id, work_counter, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, work_counter, team):
                        if food.chopped:
                            controller.pickup(self.active_bot_id, work_counter[0], work_counter[1])
                            self.goal['work_counter'] = None
                        else:
                            controller.chop(self.active_bot_id, work_counter[0], work_counter[1])
                    return False
                else:
                    self.goal['work_counter'] = None
            
            self.goal['state'] = 'BUY_INGREDIENT'
            self.goal['next_after_buy'] = 'CHOP_THEN_STORE'
            return False
        
        # === STATE: COOK_ONLY (for EGG) ===
        if state == 'COOK_ONLY':
            ing = self.goal.get('current_ingredient')
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                self.goal['state'] = 'START_COOK'
                return False
            
            self.goal['state'] = 'BUY_INGREDIENT'
            self.goal['next_after_buy'] = 'COOK_ONLY'
            return False
        
        # === STATE: ENSURE_PLATE ===
        if state == 'ENSURE_PLATE':
            plate_counter = self.goal.get('plate_counter')
            
            # Check if we already have a plate on a counter
            if plate_counter:
                tile = controller.get_tile(team, plate_counter[0], plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    self.goal['state'] = 'ASSEMBLE'
                    return False
            
            # Need to get a plate
            if holding and holding.get('type') == 'Plate':
                self.goal['state'] = 'PLACE_PLATE'
                return False
            
            # If holding something else (not plate), check if we need it for this order
            if holding is not None:
                holding_type = holding.get('type')
                if holding_type == 'Food':
                    food_name = holding.get('food_name')
                    required = order.get('required', [])
                    if food_name not in required:
                        # Holding unwanted ingredient - trash it
                        self._log(f"  ENSURE_PLATE: Holding unwanted {food_name}, trashing")
                        if self.trashes:
                            trash = self._get_nearest(bot_pos, self.trashes)
                            if trash:
                                if self._chebyshev_dist(bot_pos, trash) <= 1:
                                    controller.trash(self.active_bot_id, trash[0], trash[1])
                                else:
                                    self._move_toward(controller, self.active_bot_id, trash, team)
                                return False
                    else:
                        # Holding something we need - store it on counter first
                        self._log(f"  ENSURE_PLATE: Holding needed {food_name}, storing first")
                        counter = self._get_free_counter(controller, team, bot_pos)
                        if counter:
                            if self._chebyshev_dist(bot_pos, counter) <= 1:
                                if controller.place(self.active_bot_id, counter[0], counter[1]):
                                    stored = self.goal.get('stored_items', {})
                                    if food_name not in stored:
                                        stored[food_name] = []
                                    stored[food_name].append(counter)
                                    self.goal['stored_items'] = stored
                            else:
                                self._move_toward(controller, self.active_bot_id, counter, team)
                            return False

                else:
                    # Holding something non-food, non-plate - trash it
                    self._log(f"  ENSURE_PLATE: Holding unknown {holding_type}, trashing")
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash:
                            if self._chebyshev_dist(bot_pos, trash) <= 1:
                                controller.trash(self.active_bot_id, trash[0], trash[1])
                            else:
                                self._move_toward(controller, self.active_bot_id, trash, team)
                            return False
            
            # Try sink table first (need empty hands)
            if self.sink_tables:
                for st in self.sink_tables:
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self._is_blocked(controller, self.active_bot_id, st, team):
                            return True
                        if self._move_toward(controller, self.active_bot_id, st, team):
                            controller.take_clean_plate(self.active_bot_id, st[0], st[1])
                        return False
            
            # Buy plate (need empty hands)
            if shop is None:
                return True
            if self._is_blocked(controller, self.active_bot_id, shop, team):
                return True
            if self._move_toward(controller, self.active_bot_id, shop, team):
                if money >= ShopCosts.PLATE.buy_cost:
                    controller.buy(self.active_bot_id, ShopCosts.PLATE, shop[0], shop[1])
            return False
        
        # === STATE: PLACE_PLATE ===
        if state == 'PLACE_PLATE':
            # Check if we're actually holding a plate
            if not holding or holding.get('type') != 'Plate':
                self._log(f"  PLACE_PLATE: Not holding plate (holding={holding}), going back to ENSURE_PLATE")
                self.goal['state'] = 'ENSURE_PLATE'
                return False
            
            exclude = set()
            for locs in self.goal.get('stored_items', {}).values():
                if isinstance(locs, list):
                    exclude.update(locs)
                else:
                    exclude.add(locs)
            counter = self._get_free_counter(controller, team, bot_pos, exclude)
            if counter is None:
                self._log(f"  PLACE_PLATE: No free counter found, trashing plate")
                # All counters are full - trash the plate and start fresh
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._chebyshev_dist(bot_pos, trash) <= 1:
                            if controller.trash(self.active_bot_id, trash[0], trash[1]):
                                self._log(f"  PLACE_PLATE: Trashed plate at {trash}")
                                self.goal['state'] = 'ENSURE_PLATE'
                                return False
                        else:
                            self._move_toward(controller, self.active_bot_id, trash, team)
                            return False
                self._log(f"  PLACE_PLATE: No trash available")
                return True
            
            if self._is_blocked(controller, self.active_bot_id, counter, team):
                self._log(f"  PLACE_PLATE: Blocked reaching counter {counter}")
                return True
            if self._move_toward(controller, self.active_bot_id, counter, team):
                if controller.place(self.active_bot_id, counter[0], counter[1]):
                    self._log(f"  PLACE_PLATE: Placed plate at {counter}")
                    self.goal['plate_counter'] = counter
                    self.goal['state'] = 'ASSEMBLE'
            return False
        
        # === STATE: ASSEMBLE ===
        # Add all ingredients to plate: simple items, stored items, cooked items
        if state == 'ASSEMBLE':
            plate_counter = self.goal.get('plate_counter')
            if plate_counter is None:
                self.goal['state'] = 'ENSURE_PLATE'
                return False
            
            # Verify plate is actually there
            plate_tile = controller.get_tile(team, plate_counter[0], plate_counter[1])
            if not plate_tile or not isinstance(getattr(plate_tile, 'item', None), Plate):
                # Plate is gone - need to get a new one
                self.goal['plate_counter'] = None
                self.goal['state'] = 'ENSURE_PLATE'
                return False
            
            plate_contents = self.goal.get('plate_contents', [])
            required = order['required']
            
            # Check what's still needed - handle duplicates correctly
            current_on_plate = list(plate_contents)
            missing = []
            for req in required:
                if req in current_on_plate:
                    current_on_plate.remove(req)
                else:
                    missing.append(req)
            
            if not missing:
                self.goal['state'] = 'PICKUP_PLATE'
                return False
            
            # If holding food that we need, add it
            if holding and holding.get('type') == 'Food':
                food_name = holding.get('food_name')
                if food_name in missing:
                    if self._is_blocked(controller, self.active_bot_id, plate_counter, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, plate_counter, team):
                        if controller.add_food_to_plate(self.active_bot_id, 
                                                         plate_counter[0], plate_counter[1]):
                            plate_contents.append(food_name)
                            self.goal['plate_contents'] = plate_contents
                    return False
                else:
                    # Holding wrong food - trash it
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash:
                            if self._is_blocked(controller, self.active_bot_id, trash, team):
                                return True
                            if self._move_toward(controller, self.active_bot_id, trash, team):
                                controller.trash(self.active_bot_id, trash[0], trash[1])
                            return False
                    return False
            
            # If holding something else (plate, pan, etc), we have a problem - trash it
            if holding and holding.get('type') != 'Food':
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, self.active_bot_id, trash, team):
                            controller.trash(self.active_bot_id, trash[0], trash[1])
                        return False
                return False
            
            # Now holding is None - we can pick up or buy things
            
            # Check for stored items we can pick up
            # Check for stored items we can pick up
            stored_items = self.goal.get('stored_items', {})
            for ing, locs in list(stored_items.items()):
                if ing in missing:
                    # Check all locations
                    # Use a copy since we might modify
                    for counter_pos in list(locs):
                        tile = controller.get_tile(team, counter_pos[0], counter_pos[1])
                        if tile and isinstance(getattr(tile, 'item', None), Food):
                            if self._is_blocked(controller, self.active_bot_id, counter_pos, team):
                                continue # Try next location
                            if self._move_toward(controller, self.active_bot_id, counter_pos, team):
                                controller.pickup(self.active_bot_id, counter_pos[0], counter_pos[1])
                                # Remove specific location
                                if counter_pos in locs:
                                    locs.remove(counter_pos)
                                if not locs:
                                    del stored_items[ing]
                                self.goal['stored_items'] = stored_items
                            return False
                        else:
                            # Item gone from counter - cleanup
                            if counter_pos in locs:
                                locs.remove(counter_pos)
                            if not locs:
                                if ing in stored_items:
                                    del stored_items[ing]
                            self.goal['stored_items'] = stored_items
            
            # Check for cooked items ready to take
            cooking_items = self.goal.get('cooking_items', {})
            has_cooking = False
            for ing, cooker_pos in list(cooking_items.items()):
                if ing in missing:
                    tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food:
                            if pan.food.cooked_stage >= 1:
                                # Ready to take!
                                if self._is_blocked(controller, self.active_bot_id, cooker_pos, team):
                                    return True
                                if self._move_toward(controller, self.active_bot_id, cooker_pos, team):
                                    if pan.food.cooked_stage == 2:
                                        # Burnt - take and trash
                                        controller.take_from_pan(self.active_bot_id, 
                                                                  cooker_pos[0], cooker_pos[1])
                                        self.goal['state'] = 'TRASH_BURNT'
                                        self.goal['burnt_ingredient'] = ing
                                    else:
                                        controller.take_from_pan(self.active_bot_id, 
                                                                  cooker_pos[0], cooker_pos[1])
                                        del cooking_items[ing]
                                return False
                            else:
                                # Still cooking - remember we're waiting
                                has_cooking = True
                        else:
                            # Pan is empty - item lost?
                            del cooking_items[ing]
                    else:
                        # No pan at cooker - item lost?
                        del cooking_items[ing]
            
            # Check for simple items we need to buy
            simple_queue = self.goal.get('simple_queue', [])
            for ing in list(simple_queue):
                if ing in missing:
                    if shop is None:
                        return True
                    if self._is_blocked(controller, self.active_bot_id, shop, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, shop, team):
                        food_type = getattr(FoodType, ing, None)
                        if food_type and money >= food_type.buy_cost:
                            controller.buy(self.active_bot_id, food_type, shop[0], shop[1])
                            simple_queue.remove(ing)
                        # If can't afford, we'll try again next turn
                    return False
            
            # If we're still missing items but nothing in our queues...
            if missing and not has_cooking and not stored_items and not simple_queue:
                # Check if missing items need to be re-processed
                for ing in missing:
                    info = self.ingredient_info.get(ing, {})
                    needs_chop = info.get('chop', False)
                    needs_cook = info.get('cook', False)
                    
                    if needs_chop and needs_cook:
                        self.goal.get('chop_cook_queue', []).append(ing)
                    elif needs_chop:
                        self.goal.get('chop_only_queue', []).append(ing)
                    elif needs_cook:
                        self.goal.get('cook_only_queue', []).append(ing)
                    else:
                        self.goal.get('simple_queue', []).append(ing)
                
                self.goal['state'] = 'INIT'
                return False
            
            # Still waiting for something to cook
            return False
        
        # === STATE: TRASH_BURNT ===
        if state == 'TRASH_BURNT':
            if self.trashes:
                trash = self._get_nearest(bot_pos, self.trashes)
                if trash:
                    if self._is_blocked(controller, self.active_bot_id, trash, team):
                        return True
                    if self._move_toward(controller, self.active_bot_id, trash, team):
                        controller.trash(self.active_bot_id, trash[0], trash[1])
                        # Need to re-cook this ingredient
                        ing = self.goal.get('burnt_ingredient')
                        if ing:
                            cooking_items = self.goal.get('cooking_items', {})
                            if ing in cooking_items:
                                del cooking_items[ing]
                            # Add back to appropriate queue
                            info = self.ingredient_info.get(ing, {})
                            if info.get('chop'):
                                self.goal.get('chop_cook_queue', []).append(ing)
                            else:
                                self.goal.get('cook_only_queue', []).append(ing)
                        self.goal['state'] = 'INIT'
                    return False
            self.goal['state'] = 'ASSEMBLE'
            return False
        
        # === STATE: PICKUP_PLATE ===
        if state == 'PICKUP_PLATE':
            plate_counter = self.goal.get('plate_counter')
            if plate_counter is None:
                self.goal['state'] = 'ENSURE_PLATE'
                return False
            
            if holding and holding.get('type') == 'Plate':
                self.goal['state'] = 'SUBMIT'
                return False
            
            if self._is_blocked(controller, self.active_bot_id, plate_counter, team):
                return True
            if self._move_toward(controller, self.active_bot_id, plate_counter, team):
                controller.pickup(self.active_bot_id, plate_counter[0], plate_counter[1])
                self.goal['state'] = 'SUBMIT'
            return False
        
        # === STATE: SUBMIT ===
        if state == 'SUBMIT':
            if submit is None:
                self._log(f"  SUBMIT: No submit tile found!")
                return True
            
            # Check if the order is active by checking the current orders list
            current_turn = controller.get_turn()
            order_id = order.get('order_id')
            expires = order.get('expires_turn', 0)
            starts_at = order.get('created_turn', 0)
            
            # Order should be active if: started <= current_turn < expires
            if current_turn < starts_at:
                self._log(f"  SUBMIT: Order #{order_id} not started yet (starts turn {starts_at}), waiting")
                self._move_toward(controller, self.active_bot_id, submit, team)
                return False
            
            if self._is_blocked(controller, self.active_bot_id, submit, team):
                self._log(f"  SUBMIT: Blocked reaching submit tile {submit}")
                return True
            
            is_adjacent = self._move_toward(controller, self.active_bot_id, submit, team)
            self._log(f"  SUBMIT: adjacent={is_adjacent}, bot_pos={bot_pos}, submit={submit}")
            
            if is_adjacent:
                # Try to submit
                submit_result = controller.submit(self.active_bot_id, submit[0], submit[1])
                self._log(f"  SUBMIT: controller.submit() returned {submit_result}")
                if submit_result:
                    self._log(f"  ORDER COMPLETED! Submitted order #{order.get('order_id')}")
                    self.completed_order_ids.add(order.get('order_id'))
                    self.goal = None
                else:
                    # Submit failed - check if there's a matching order we can submit
                    current_orders = controller.get_orders(team)
                    for o in current_orders:
                        if o.get('is_active', False) and o.get('required') == order.get('required'):
                            # Found a matching active order - try submitting for that one
                            self._log(f"  SUBMIT: Trying matching order #{o.get('order_id')}")
                            submit_result = controller.submit(self.active_bot_id, submit[0], submit[1])
                            if submit_result:
                                self._log(f"  ORDER COMPLETED! Submitted order #{o.get('order_id')}")
                                self.completed_order_ids.add(o.get('order_id'))
                                self.goal = None
                                break
                    if self.goal is not None:
                        # Still couldn't submit - log failure
                        holding_type = holding.get('type') if holding else None
                        self._log(f"  SUBMIT FAILED: holding={holding_type}, order_required={order.get('required')}")
                        
                        # Enhanced Debugging for Submit Failures
                        if holding and holding.get('type') == 'Plate':
                            plate_contents = []
                            for f in holding.get('food', []):
                                plate_contents.append(f"{f.get('food_name')}(chop={f.get('chopped')},cook={f.get('cooked_stage')})")
                            plate_contents.sort()
                            self._log(f"    PLATE CONTENTS: {plate_contents}")
                            
                            required_details = []
                            for r in order.get('required', []):
                                info = self.ingredient_info.get(r, {})
                                required_details.append(f"{r}(chop={info.get('chop')},cook={1 if info.get('cook') else 0})")
                            required_details.sort()
                            self._log(f"    ORDER REQUIRES: {required_details}")
            return False
        
        return False
    
    def _handle_handoff_pickup(self, controller: RobotController, team: Team) -> bool:
        """Handle picking up from handoff drop tile. Returns True if we should stop this turn."""
        if self.goal is None:
            return False
        
        drop_tile = self.goal.get('handoff_drop_tile')
        if drop_tile is None:
            return False
        
        bot = controller.get_bot_state(self.active_bot_id)
        holding = bot.get('holding')
        
        if holding is not None:
            # Already holding something - clear handoff and continue
            self.goal['handoff_drop_tile'] = None
            return False
        
        bot_pos = (bot['x'], bot['y'])
        
        tile = controller.get_tile(team, drop_tile[0], drop_tile[1])
        if tile is None or getattr(tile, 'item', None) is None:
            # Item gone - clear handoff and continue
            self.goal['handoff_drop_tile'] = None
            return False
        
        if self._chebyshev_dist(bot_pos, drop_tile) <= 1:
            controller.pickup(self.active_bot_id, drop_tile[0], drop_tile[1])
            self.goal['handoff_drop_tile'] = None
            return True  # FIX: Stop here, we used our action
        else:
            if self._is_blocked(controller, self.active_bot_id, drop_tile, team):
                self._swap_control()
                return True
            self._move_toward(controller, self.active_bot_id, drop_tile, team)
            return True
    
    def _get_kitchen_center(self) -> Tuple[int, int]:
        """Get approximate center of kitchen (shop/counter/cooker area)."""
        key_tiles = self.shops + self.counters + self.cookers + self.submits
        if not key_tiles:
            return (0, 0)
        avg_x = sum(t[0] for t in key_tiles) // len(key_tiles)
        avg_y = sum(t[1] for t in key_tiles) // len(key_tiles)
        return (avg_x, avg_y)
    
    def _choose_starting_bot(self, controller: RobotController, bots: List[int]) -> int:
        """Choose the bot that's closer to the kitchen area."""
        if len(bots) == 1:
            return bots[0]
        
        kitchen = self._get_kitchen_center()
        
        best_bot = bots[0]
        best_dist = float('inf')
        
        for bot_id in bots:
            pos = self._get_bot_pos(controller, bot_id)
            dist = self._chebyshev_dist(pos, kitchen)
            if dist < best_dist:
                best_dist = dist
                best_bot = bot_id
        
        return best_bot
    
    def _do_active_support(self, controller: RobotController, team: Team) -> None:
        """Have the inactive bot proactively fetch ingredients."""
        if self.inactive_bot_id is None or self.goal is None:
            return
            
        bot_id = self.inactive_bot_id
        bot = controller.get_bot_state(bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        # 1. If holding something useful, bring it close
        if holding:
            if holding.get('type') == 'Food':
                food_name = holding.get('food_name')
                # Check if needed
                needed = False
                for q in ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']:
                    if food_name in self.goal.get(q, []):
                        needed = True
                        break
                
                if needed:
                    # Drop off at nearest free counter to Kitchen Center
                    kitchen = self._get_kitchen_center()
                    exclude = set()
                    for locs in self.goal.get('stored_items', {}).values():
                        if isinstance(locs, list):
                            exclude.update(locs)
                        else:
                            exclude.add(locs)
                    if self.goal.get('plate_counter'):
                        exclude.add(self.goal['plate_counter'])
                        
                    counter = self._get_free_counter(controller, team, kitchen, exclude)
                    if counter:
                        if self._is_blocked(controller, bot_id, counter, team):
                            return
                        if self._move_toward(controller, bot_id, counter, team):
                            if controller.place(bot_id, counter[0], counter[1]):
                                # Track it properly - append to list
                                stored = self.goal.get('stored_items', {})
                                if food_name not in stored:
                                    stored[food_name] = []
                                # Only add if not already there (prevent duplicate entries for same location)
                                if counter not in stored[food_name]:
                                    stored[food_name].append(counter)
                                self.goal['stored_items'] = stored
                                self._log(f"SUPPORT: Delivered {food_name} to {counter}")
                else:
                    # Holding garbage - trash it
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash and self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
            return

        # 2. Check what's needed and go get it
        # Prioritize queues: simple > cook > chop
        candidates = []
        candidates.extend(self.goal.get('simple_queue', []))
        candidates.extend(self.goal.get('cook_only_queue', []))
        candidates.extend(self.goal.get('chop_only_queue', []))
        
        target_ing = None
        for ing in candidates:
            # Skip if active bot is already handling it
            if ing == self.goal.get('current_ingredient'):
                continue
            
            # Check how many we already have stored
            stored_list = self.goal.get('stored_items', {}).get(ing, [])
            needed_count = candidates.count(ing)
            if len(stored_list) >= needed_count:
                continue
                
            # Skip if already cooking (handled separately? or just treat as stored?)
            # Simplified: just check storage buffer
            
            target_ing = ing
            break
            
        if target_ing:
            # Go buy it
            shop = self._get_nearest(bot_pos, self.shops)
            if shop:
                if self._move_toward(controller, bot_id, shop, team):
                    ft = getattr(FoodType, target_ing, None)
                    if ft and controller.get_team_money(team) >= ft.buy_cost:
                        controller.buy(bot_id, ft, shop[0], shop[1])
                        self._log(f"SUPPORT: Bought {target_ing}")

    def _do_preparation(self, controller: RobotController, team: Team) -> None:
        """Do useful preparation work when no orders are available."""

        if self.active_bot_id is None:
            return
        
        bot = controller.get_bot_state(self.active_bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        # If holding something, just put it down or trash it
        if holding is not None:
            if self.trashes:
                trash = self._get_nearest(bot_pos, self.trashes)
                if trash:
                    if self._move_toward(controller, self.active_bot_id, trash, team):
                        controller.trash(self.active_bot_id, trash[0], trash[1])
            return
        
        # Check if cooker has burnt/cooked food that should be removed
        for cooker_pos in self.cookers:
            tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food is not None and pan.food.cooked_stage >= 1:
                    # Food is done or burnt - take it out and trash it
                    if self._move_toward(controller, self.active_bot_id, cooker_pos, team):
                        if controller.take_from_pan(self.active_bot_id, cooker_pos[0], cooker_pos[1]):
                            self._log(f"  PREP: Took cooked/burnt food from cooker")
                    return
        
        # Don't do any more preparation - just wait for orders
        # Buying ingredients speculatively wastes money
    
    def play_turn(self, controller: RobotController) -> None:
        """Main entry point - called each turn."""
        team = controller.get_team()
        self.team = team
        
        # Initialize logging on first turn
        if not self.initialized:
            self._init_log()
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Choose the bot closer to the kitchen for initial activation
        if self.active_bot_id is None:
            self.active_bot_id = self._choose_starting_bot(controller, bots)
            if len(bots) > 1:
                self.inactive_bot_id = [b for b in bots if b != self.active_bot_id][0]
            self._log(f"Initial bot selection: active={self.active_bot_id}, inactive={self.inactive_bot_id}")
        
        # Log state at start of turn
        self._log_state(controller, team)
        
        if self._handle_handoff_pickup(controller, team):
            self._log("ACTION: Handling handoff pickup, ending turn")
            return
        
        if self._handle_inactive_bot_storage(controller, team):
            self._log("ACTION: Handling inactive bot storage, ending turn")
            return

        # Run Active Support if goal exists
        if self.goal is not None and self.inactive_bot_id is not None:
             self._do_active_support(controller, team)

        
        # Check if current goal's order has expired
        if self.goal is not None:
            order = self.goal.get('order')
            if order is not None:
                current_turn = controller.get_turn()
                expires = order.get('expires_turn', 0)
                if current_turn > expires:
                    expired_order_id = order.get('order_id')
                    self._log(f"ORDER EXPIRED: Order #{expired_order_id} expired at turn {expires}")
                    # Clean up: if there's a plate on a counter, we need to handle it
                    plate_counter = self.goal.get('plate_counter')
                    if plate_counter:
                        self._log(f"  Cleanup: plate was at {plate_counter}, marking for cleanup")
                        # Store the plate position so we can clean it up
                        self.abandoned_plate = plate_counter
                    
                    # Clean up inactive bot storage if it was for this expired order
                    if self.inactive_bot_storage is not None:
                        storage_order_id = self.inactive_bot_storage.get('for_order_id')
                        if storage_order_id == expired_order_id:
                            self._log(f"  Cleanup: inactive bot was holding for expired order, clearing storage")
                            # The ingredient will still be held - it will get trashed during preparation
                            self.inactive_bot_storage = None
                    
                    self.goal = None
        
        # Handle abandoned plate - try to reuse it for another order
        if self.abandoned_plate is not None:
            bot = controller.get_bot_state(self.active_bot_id)
            bot_pos = (bot['x'], bot['y'])
            holding = bot.get('holding')
            
            # Check if the plate is still there
            plate_tile = controller.get_tile(team, self.abandoned_plate[0], self.abandoned_plate[1])
            plate_item = getattr(plate_tile, 'item', None) if plate_tile else None
            
            if plate_item and isinstance(plate_item, Plate):
                # Get what's on the plate
                plate_contents = [f.food_name for f in plate_item.foods] if hasattr(plate_item, 'foods') else []
                self._log(f"  Abandoned plate at {self.abandoned_plate} has: {plate_contents}")
                
                # Find an order that matches or can use these contents
                orders = controller.get_orders(team)
                current_turn = controller.get_turn()
                matching_order = None
                
                for order in orders:
                    if not order.get('is_active', False):
                        continue
                    required = order.get('required', [])
                    expires = order.get('expires_turn', 0)
                    if expires <= current_turn:
                        continue
                    
                    # Check if plate contents are a subset of required ingredients
                    required_copy = list(required)
                    is_valid = True
                    for item in plate_contents:
                        if item in required_copy:
                            required_copy.remove(item)
                        else:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # This order can use the plate!
                        matching_order = order
                        remaining = required_copy
                        self._log(f"  Found matching order #{order.get('order_id')}: needs {remaining} more")
                        break
                
                if matching_order:
                    # Adopt this plate for the matching order
                    self.goal = self._create_goal(matching_order)
                    self.goal['plate_counter'] = self.abandoned_plate
                    self.goal['plate_contents'] = plate_contents
                    # Remove already-on-plate items from queues
                    for item in plate_contents:
                        for queue_name in ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']:
                            queue = self.goal.get(queue_name, [])
                            if item in queue:
                                queue.remove(item)
                    self.goal['state'] = 'ASSEMBLE'
                    self._log(f"  Adopted plate for order #{matching_order.get('order_id')}")
                    self.abandoned_plate = None
                    # Continue with the goal execution below
                else:
                    # No matching order - trash the plate
                    self._log(f"  No matching order for plate contents {plate_contents}, trashing")
                    if holding is None:
                        # Pick up the plate first
                        if self._chebyshev_dist(bot_pos, self.abandoned_plate) <= 1:
                            controller.pickup(self.active_bot_id, self.abandoned_plate[0], self.abandoned_plate[1])
                            return
                        else:
                            self._move_toward(controller, self.active_bot_id, self.abandoned_plate, team)
                            return
                    else:
                        # Already holding - trash it
                        if self.trashes:
                            trash = self._get_nearest(bot_pos, self.trashes)
                            if trash:
                                if self._chebyshev_dist(bot_pos, trash) <= 1:
                                    controller.trash(self.active_bot_id, trash[0], trash[1])
                                    self._log(f"  Trashed plate at {trash}")
                                    self.abandoned_plate = None
                                    return
                                else:
                                    self._move_toward(controller, self.active_bot_id, trash, team)
                                    return
            else:
                # Plate is gone
                self._log(f"  Abandoned plate at {self.abandoned_plate} is gone")
                self.abandoned_plate = None
        
        if self.goal is None:
            order = self._select_order(controller, team)
            if order is None:
                self._log("NO ORDERS: Doing preparation work")
                self._do_preparation(controller, team)
                return
            self._log(f"NEW ORDER SELECTED: #{order.get('order_id')} - {order.get('required')}, expires turn {order.get('expires_turn')}")
            self.goal = self._create_goal(order)
        
        blocked = self._execute_goal(controller, team)
        
        if blocked:
            self._log(f"BLOCKED in state {self.goal.get('state') if self.goal else 'N/A'}")
            if self.inactive_bot_id is not None:
                self._log("Attempting handoff/swap")
                self._handle_blocked(controller, team)
