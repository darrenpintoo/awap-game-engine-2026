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
        # Clear existing lists to avoid duplicates on re-scan (Turn 101)
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sinks = []
        self.sink_tables = []
        
        m = controller.get_map(team)
        for x in range(m.width):
            for y in range(m.height):
                tile = controller.get_tile(team, x, y)
                if not tile: continue
                pos = (x, y)
                name = tile.tile_name
                
                if name == "SHOP":
                    self.shops.append(pos)
                elif name == "COOKER":
                    self.cookers.append(pos)
                elif name == "COUNTER" or name == "BOX":
                    self.counters.append(pos)
                elif name == "SINKTABLE":
                    self.sink_tables.append(pos)
                elif name == "SUBMIT":
                    self.submits.append(pos)
                elif name == "TRASH":
                    self.trashes.append(pos)
                elif name == "SINK":
                    self.sinks.append(pos)
                
                if name not in ["FLOOR", "WALL", "VOID"]:
                    self._log(f"MAP DIAGNOSTIC: Tile at {pos} is '{name}'")
                    
        self.initialized = True
        self._log(f"MAP INIT: counters={self.counters}")
        self._log(f"MAP INIT: cookers={self.cookers}")
        self._log(f"MAP INIT: shops={self.shops}")
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
                          exclude: Optional[Set[Tuple[int, int]]] = None,
                          for_chopping: bool = False) -> Optional[Tuple[int, int]]:
        """Find nearest free counter. If for_chopping=False, prioritizing sink_tables."""
        free = []
        
        # Check sink tables first if not for chopping (preserve true counters for chopping)
        if not for_chopping:
            for st in self.sink_tables:
                if exclude and st in exclude:
                    continue
                tile = controller.get_tile(team, st[0], st[1])
                if tile and getattr(tile, 'item', None) is None:
                    free.append(st)
            
            # If we found a sink table, use it!
            if free:
                return self._get_nearest(pos, free)

        # Check standard counters/boxes
        for c in self.counters:
            if exclude and c in exclude:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
                    
        return self._get_nearest(pos, free)
    
    def _count_free_staging_slots(self, controller: RobotController, team: Team) -> int:
        """Count total free counters and sink tables."""
        count = 0
        for c in (self.counters + self.sink_tables):
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                count += 1
        return count

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
                self._log(f"  HANDOFF: No valid drop tile found. WAITING.")
                # LEAN STRATEGY: Do not swap anyway - that causes item loss/duplication.
                # Just wait for a counter to open or for the blocking bot to move.
                return True
        
        # If not holding anything, or if handoff was rejected, we can swap to see if partner can do something
        if holding is None:
            self._swap_control()
        return
    
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
        # Need full processing - realistic estimates
        # Check if item exists in world (cooked/chopped)
        # Scan cookers for cooked items
        found_in_cooker = False
        if needs_cook:
            for c_pos in self.cookers:
                tile = controller.get_tile(team, c_pos[0], c_pos[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                     pan = tile.item
                     if pan.food and pan.food.food_name == ing and pan.food.cooked_stage == 1:
                         found_in_cooker = True
                         break
        
        # Scan counters for chopped items (or raw if needed)
        found_on_counter = False
        for c_pos in self.counters:
             tile = controller.get_tile(team, c_pos[0], c_pos[1])
             if tile and hasattr(tile, 'item') and isinstance(tile.item, Food):
                 f = tile.item
                 if f.food_name == ing:
                     if needs_chop and f.chopped:
                         found_on_counter = True
                         break
                     if not needs_chop and not needs_cook: # Simple item
                         found_on_counter = True
                         break

        if needs_chop and needs_cook:
            if found_in_cooker:
                 # Already cooked! Just pickup from pan
                 total_time += AVG_MOVE + 1 + AVG_MOVE + 1
            elif found_on_counter: # Chopped but not cooked
                 # Move to counter + pickup + move to cooker + place + cook + ...
                 ops = (AVG_MOVE + 1) + (AVG_MOVE + 1) + (AVG_MOVE + 1 + AVG_MOVE + 1)
                 # Optimistic: helper handles the cooking wait or parallel steps
                 if self.inactive_bot_id: ops *= 0.6
                 total_time += ops
                 max_cook_time = max(max_cook_time, 10)
            else:
                # Full prep
                # Optimistic: parallel prep
                ops = (AVG_MOVE + 1) + (AVG_MOVE + 1 + 6 + 1) + (AVG_MOVE + 1) + (AVG_MOVE + 1 + AVG_MOVE + 1)
                if self.inactive_bot_id: ops *= 0.5 # High optimism for Meat!
                total_time += ops
                max_cook_time = max(max_cook_time, 10)
        elif needs_chop:
            if found_on_counter:
                # Already chopped
                total_time += AVG_MOVE + 1 + AVG_MOVE + 1
            else:
                # Full prep
                ops = (AVG_MOVE + 1) + (AVG_MOVE + 1 + 6 + 1) + (AVG_MOVE + 1)
                if self.inactive_bot_id: ops *= 0.7
                total_time += ops
        elif needs_cook:
            if found_in_cooker:
                 total_time += AVG_MOVE + 1 + AVG_MOVE + 1
            else:
                 # Full prep
                ops = (AVG_MOVE + 1) + (AVG_MOVE + 1) + (AVG_MOVE + 1 + AVG_MOVE + 1)
                if self.inactive_bot_id: ops *= 0.7
                total_time += ops
                max_cook_time = max(max_cook_time, 10)
        else:
            # NOODLES, SAUCE
            if found_on_counter:
                  total_time += AVG_MOVE + 1 + AVG_MOVE + 1
            else:
                  ops = AVG_MOVE + 1 + AVG_MOVE + 1
                  if self.inactive_bot_id: ops *= 0.5 # Fast fetch
                  total_time += ops
        
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
            
            # Economics: Skip orders that start after the typical game end (250)
            # OR orders that start way too late to be finished.
            if starts_at > 230:
                continue
            
            completion_turn = self._estimate_completion_turn(order, current_turn, controller, team)
            reward = order.get('reward', 0)
            time_left = expires - current_turn
            difficulty = self._get_order_difficulty(order)
            


            if completion_turn < 99999:
                # This order is completable - calculate priority score
                # Lower score = better priority
                
                # Urgency: orders starting within 30 turns get priority
                # Urgency: orders starting within 30 turns get priority
                turns_until_start = max(0, starts_at - current_turn)
                # SNIPING FIX: Treat "starting soon" (<=30) exactly same as "started" (0)
                # This allows us to pick easy future orders over hard current orders
                urgency_bonus = 0 if turns_until_start <= 30 else 1000
                
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
    

    def _create_goal(self, order: Dict, controller: RobotController) -> Dict[str, Any]:
        """Create a new goal state for an order."""
        self.goal_counter += 1
        
        # Categorize ingredients
        chop_cook = []   # MEAT: chop then cook
        chop_only = []   # ONIONS: chop, store on counter
        cook_only = []   # EGG: cook
        simple = []      # NOODLES, SAUCE: just add to plate
        
        # Helper lists for scanning
        missing_ingredients = list(order['required'])
        stored_items = {}
        cooking_items = {}
        
        # SCAN MAP first to find existing items
        # 1. Check Cooked Items on Counters (Prepped by _do_preparation)
        for c_pos in self.counters:
            tile = controller.get_tile(self.team, c_pos[0], c_pos[1])
            if tile and isinstance(getattr(tile, 'item', None), Food):
                f = tile.item
                # If we need this item and it's cooked (MEAT) or chopped (ONION) or simple
                if f.food_name in missing_ingredients:
                     # Check if it satisfies the requirement
                     info = self.ingredient_info.get(f.food_name, {})
                     needs_chop = info.get('chop', False)
                     needs_cook = info.get('cook', False)
                     
                     is_ready = True
                     if needs_cook and f.cooked_stage < 1: is_ready = False
                     if needs_chop and not f.chopped: is_ready = False
                     
                     if is_ready:
                         # Found it!
                         if f.food_name not in stored_items: stored_items[f.food_name] = []
                         stored_items[f.food_name].append(c_pos)
                         # Remove one instance from missing
                         missing_ingredients.remove(f.food_name)
                         
        # 2. Check Items in Cookers
        for c_pos in self.cookers:
            tile = controller.get_tile(self.team, c_pos[0], c_pos[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food and pan.food.food_name in missing_ingredients:
                     info = self.ingredient_info.get(pan.food.food_name, {})
                     # If cooked or cooking, we can use it
                     # Logic usually adds to cooking_items
                     cooking_items[pan.food.food_name] = c_pos
                     missing_ingredients.remove(pan.food.food_name)

        # Now populate queues with REMAINDER
        for ing in missing_ingredients:
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
            'stored_items': stored_items,      # {ingredient_name: counter_pos}
            'cooking_items': cooking_items,    # {ingredient_name: cooker_pos}
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
            # SAFETY LOCK: Don't start chop_only on 1-counter maps if plate isn't ready
            low_resource = (len(self.counters) <= 1)
            plate_ready = (self.goal.get('plate_counter') is not None)
            
            if chop_only and (not low_resource or plate_ready):
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
                # Wrong item - trash it (or place it anywhere to free hands)
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._chebyshev_dist(bot_pos, trash) <= 1:
                             if controller.trash(self.active_bot_id, trash[0], trash[1]):
                                 self._log(f"  BUY_INGREDIENT: Trashed unwanted {holding} at {trash}")
                                 return False # Action taken
                        else:
                             if self._move_toward(controller, self.active_bot_id, trash, team):
                                 return False # Moved
                
                # Fallback: if trash failed or unreachable, put it on any counter
                self._log(f"  BUY_INGREDIENT: Could not trash {holding}, trying to place on counter")
                counter = self._get_free_counter(controller, team, bot_pos)
                if counter:
                    if self._chebyshev_dist(bot_pos, counter) <= 1:
                        if controller.place(self.active_bot_id, counter[0], counter[1]):
                            self._log(f"  BUY_INGREDIENT: Placed unwanted {holding} at {counter}")
                            return False
                    else:
                        self._move_toward(controller, self.active_bot_id, counter, team)
                        return False
                
                return False
            
            if shop is None:
                return True

            # SYNC: Check if partner is holding this item!
            if self.inactive_bot_id is not None:
                inactive_bot = controller.get_bot_state(self.inactive_bot_id)
                ib_hold = inactive_bot.get('holding')
                if ib_hold and ib_hold.get('type') == 'Food' and ib_hold.get('food_name') == ing:
                    self._log(f"  BUY_INGREDIENT: Partner is already holding {ing}! Swapping to them.")
                    self._swap_control()
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
                
                counter = self._get_free_counter(controller, team, bot_pos, exclude, for_chopping=True)
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
            
            # SYNC: Check if inactive bot is already fetching/staging a plate
            if self.inactive_bot_id is not None:
                ib_state = controller.get_bot_state(self.inactive_bot_id)
                ib_hold = ib_state.get('holding')
                if ib_hold and ib_hold.get('type') == 'Plate':
                    self._log(f"  ENSURE_PLATE: Inactive bot is already bringing a plate. Waiting.")
                    return True

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
                self._log(f"  PLACE_PLATE: No free counter found. WAITING.")
                # LEAN STRATEGY: Do not trash the plate! 
                # Just wait for a counter to open up or for the support bot to move.
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
            
            # SYNC: Update our knowledge of what's on the plate (in case Support bot added something)
            real_contents = [f.food_name for f in plate_tile.item.food]
            self.goal['plate_contents'] = real_contents
            
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

            # SYNC: Check if inactive bot is holding something we need
            # If so, WAIT for them to deliver it (don't go buy it yourself!)
            if self.inactive_bot_id is not None:
                inactive_bot = controller.get_bot_state(self.inactive_bot_id)
                ib_holding = inactive_bot.get('holding')
                if ib_holding and ib_holding.get('type') == 'Food':
                    fname = ib_holding.get('food_name')
                    if fname in missing:
                        self._log(f"  ASSEMBLE: Inactive bot holding needed {fname}, waiting for delivery")
                        return True # End turn (wait)
            
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
                    # Submit failed - log failure and consume turn
                    holding_type = holding.get('type') if holding else None
                    self._log(f"  SUBMIT FAILED for order #{order.get('order_id')}: holding={holding_type}, required={order.get('required')}")
                    
                    if holding and holding.get('type') == 'Plate':
                        plate_contents = [f.get('food_name') for f in holding.get('food', [])]
                        self._log(f"    PLATE CONTENTS: {plate_contents}")
                        self._log(f"    ORDER REQUIRES: {order.get('required')}")
                    
                    # IMPORTANT: return True here because we used an action (submit)
                    return True
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
        order = self.goal.get('order')
        
        # 1. If holding something useful, bring it close
        if holding:
            if holding.get('type') == 'Food':
                food_name = holding.get('food_name')
                # Check if needed
                needed_active = False
                for q in ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']:
                    q_list = self.goal.get(q, [])
                    if food_name in q_list:
                        # SYNC CHECK: Is active bot already holding this? 
                        # Or is it already cooking/stored? 
                        # (Queues usually handle storage/cooking, but not hands)
                        active_bot = controller.get_bot_state(self.active_bot_id)
                        ab_hold = active_bot.get('holding')
                        if ab_hold and ab_hold.get('type') == 'Food' and ab_hold.get('food_name') == food_name:
                             # Partner already has it - we don't need to add ours to plate
                             continue
                        
                        needed_active = True
                        break
                
                if needed_active:
                    # Priority: Deliver directly to plate if it exists
                    plate_counter = self.goal.get('plate_counter')
                    if plate_counter:
                        # Check if this item is actually needed on the PHYSICAL plate
                        tile = controller.get_tile(team, plate_counter[0], plate_counter[1])
                        on_plate = False
                        if tile and isinstance(getattr(tile, 'item', None), Plate):
                            plate = tile.item
                            contents = [f.food_name for f in plate.food]
                            # Simple check: does it already have what we are holding?
                            # (Queues handle total counts, we just need to avoid duplicates)
                            if food_name in contents:
                                # Count required total vs on plate
                                req_total = order.get('required', []).count(food_name)
                                if contents.count(food_name) >= req_total:
                                    on_plate = True
                        
                        if not on_plate and self._move_toward(controller, bot_id, plate_counter, team):
                            # Try to add to plate
                            if self._chebyshev_dist(bot_pos, plate_counter) <= 1:
                                if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                                    self._log(f"SUPPORT: Added {food_name} directly to plate at {plate_counter}")
                                    # Update queues
                                    for q in ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']:
                                        queue = self.goal.get(q, [])
                                        if food_name in queue:
                                            queue.remove(food_name)
                                    return
                    
                    # Drop off at nearest free counter to Kitchen Center
                    # SAFETY: On extremely cramped maps, check total storage slots (COUNTER + SINKTABLE)
                    if (len(self.counters) + len(self.sink_tables)) <= 1:
                        # Just hold it and wait
                        return

                    kitchen = self._get_kitchen_center()
                    
                    # LEAN STRATEGY: "Leave-One-Free" rule for cramped maps.
                    # If placing this would leave 0 free slots, and no plate is staged, WAIT.
                    is_plate_on_counter = (self.goal.get('plate_counter') is not None)
                    free_slots = self._count_free_staging_slots(controller, team)
                    if free_slots <= 1 and not is_plate_on_counter:
                         # Support bot should wait to ensure active bot can place plate
                         return

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
                                return
            if holding.get('type') == 'Plate':
                # STAGING: Place plate on counter/sinktable near kitchen if goal doesn't have one yet
                plate_counter = self.goal.get('plate_counter') if self.goal else None
                if plate_counter is None:
                    kitchen = self._get_kitchen_center()
                    counter = self._get_free_counter(controller, team, kitchen)
                    if counter:
                        if self._move_toward(controller, bot_id, counter, team):
                            if controller.place(bot_id, counter[0], counter[1]):
                                self._log(f"SUPPORT: Staged plate at {counter}")
                                if self.goal: self.goal['plate_counter'] = counter
                        return
                    
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
            
            # Check if active bot is holding it
            active_bot = controller.get_bot_state(self.active_bot_id)
            ab_holding = active_bot.get('holding')
            if ab_holding and ab_holding.get('type') == 'Food' and ab_holding.get('food_name') == ing:
                continue

            # Check if active bot is AT A SHOP (implies they are buying it)
            ab_pos = (active_bot['x'], active_bot['y'])
            for s in self.shops:
                if self._chebyshev_dist(ab_pos, s) <= 1:
                     # Active bot at shop - almost certainly buying something. 
                     # To be safe, don't interfere.
                     # (Refined: Only skip if it's a simple ingredient we typically buy)
                     if ing in ['NOODLES', 'SAUCE']:  # Simple items only
                         continue

            # Check how many we already have stored/delivered
            target_orders = [self.goal.get('order')] if self.goal else []
            
            # EXPAND HORIZON: Fetch for upcoming orders but never corrupt the plate
            all_orders = controller.get_orders(team)
            current_turn = controller.get_turn()
            
            # ECONOMY: Stop pre-fetching for later orders near the end of the game (total turns 250)
            horizon_limit = 220 if current_turn > 200 else 235
            
            for o in all_orders:
                starts_at = o.get('created_turn', 0)
                # NEW: fetch for upcoming orders too
                if (o.get('is_active', False) or (starts_at > 0 and starts_at < current_turn + 50)) and o not in target_orders:
                    if starts_at > horizon_limit: continue # Don't fetch for impossible orders
                    target_orders.append(o)
                    if len(target_orders) >= 2: break # Max 2 orders ahead
            
            candidates = []
            for o in target_orders:
                if o: candidates.extend(o.get('required', []))
                
            plate_contents = self.goal.get('plate_contents', []) if self.goal else []
            needed_count = candidates.count(ing) - plate_contents.count(ing)
            
            # ACCOUNT FOR HANDS: Don't buy if either bot is already holding it
            already_holding = 0
            for b_id in [self.active_bot_id, self.inactive_bot_id]:
                if b_id is None: continue
                b_state = controller.get_bot_state(b_id)
                h = b_state.get('holding')
                if h and h.get('type') == 'Food' and h.get('food_name') == ing:
                    already_holding += 1
            
            needed_count -= already_holding

            # SUPPORT BOT RESTRICTION: Only buy basic items (NOODLES, SAUCE)
            # This prevents it from buying Meat and clogging counters/hands
            is_simple = ing in ['NOODLES', 'SAUCE', 'EGG', 'ONIONS']
            if not is_simple:
                continue

            stored_list = self.goal.get('stored_items', {}).get(ing, [])
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
                return

        # 3. If nothing to do, move to Shop to be ready
        shop = self._get_nearest(bot_pos, self.shops)
        if shop:
            if self._chebyshev_dist(bot_pos, shop) > 1:
                self._move_toward(controller, bot_id, shop, team)
                self._log("SUPPORT: Idling - Moving to Shop")
    
    def _do_preparation(self, controller: RobotController, team: Team) -> None:
        """Do useful preparation work when no orders are available."""
        # LEAN STRATEGY: Disable speculative prep to prevent deadlocks on cramped maps
        # Eric doesn't prep, neither should we.
        return
        if self.active_bot_id is None:
            return
            
        # Only do prep if we have no active goal
        if self.goal is not None:
             return
        
        bot = controller.get_bot_state(self.active_bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        # 1. Handle Holding Items
        if holding is not None:
            # If holding useful food (Meat, Onion), process it or place on counter
            if holding.get('type') == 'Food':
                food_name = holding.get('food_name')
                
                # Logic for MEAT: Chop -> Cook
                if food_name == 'MEAT':
                     if holding.get('cooked_stage', 0) >= 1:
                         # Already cooked! Store on counter for pickup
                         counter = self._get_free_counter(controller, team, bot_pos)
                         if counter:
                             if self._chebyshev_dist(bot_pos, counter) <= 1:
                                 if controller.place(self.active_bot_id, counter[0], counter[1]):
                                     self._log(f"PREP: Stored COOKED MEAT at {counter}")
                             else:
                                 self._move_toward(controller, self.active_bot_id, counter, team)
                             return
                     
                     if not holding.get('chopped'):
                         # Go chop
                         # Find work counter? Any empty counter works for chopping if we have knife?
                         # Actually just 'place on counter' and it becomes chopped? No, need action?
                         # AWAP engine: Place on counter -> use Chop action? Or just Place and it's done?
                         # Usually: Place on counter -> wait (if auto-chop) or interact.
                         # Assuming standard counters allow work.
                         counter = self._get_free_counter(controller, team, bot_pos)
                         if counter:
                             if self._chebyshev_dist(bot_pos, counter) <= 1:
                                 if controller.place(self.active_bot_id, counter[0], counter[1]):
                                     # We assume placing it starts chopping/allows chopping?
                                     # Actually, betterbot usually handles this in CHOP state by placing.
                                     # Wait, does the map have specific Chopping Stations? or assume all counters?
                                     # "C" is counter.
                                     self._log(f"PREP: Placed MEAT for chopping at {counter}")
                             else:
                                 self._move_toward(controller, self.active_bot_id, counter, team)
                             return
                     else:
                         # It is chopped - COOK IT
                         cooker = self._get_available_cooker(controller, team, bot_pos)
                         if cooker:
                             if self._chebyshev_dist(bot_pos, cooker) <= 1:
                                 controller.place(self.active_bot_id, cooker[0], cooker[1])
                                 self._log(f"PREP: Placed Chopped MEAT in cooker at {cooker}")
                             else:
                                 self._move_toward(controller, self.active_bot_id, cooker, team)
                             return

                # Default: Place on counter
                counter = self._get_free_counter(controller, team, bot_pos)
                if counter:
                    if self._chebyshev_dist(bot_pos, counter) <= 1:
                        if controller.place(self.active_bot_id, counter[0], counter[1]):
                            self._log(f"PREP: Stored speculative {food_name} at {counter}")
                    else:
                        self._move_toward(controller, self.active_bot_id, counter, team)
                    return

            # Else trash it
            if self.trashes:
                trash = self._get_nearest(bot_pos, self.trashes)
                if trash:
                    if self._move_toward(controller, self.active_bot_id, trash, team):
                        controller.trash(self.active_bot_id, trash[0], trash[1])
            return
        
        # 2. Speculative Cooking (Meat)
        # Check if meat is cooking
        meat_cooking = False
        for c_pos in self.cookers:
            tile = controller.get_tile(team, c_pos[0], c_pos[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan and pan.food and pan.food.food_name == 'MEAT':
                    meat_cooking = True
                    break
        
        # Disable Prep only if truly no space (check counters AND sink tables)
        if (len(self.counters) + len(self.sink_tables)) <= 1:
             return
             
        # ECONOMY: Stop speculative prep near the end of the game
        if controller.get_turn() > 220:
             return

        if not meat_cooking and money > 300:
            # We want to cook meat.
            # Strategy: Buy -> Chop -> Cook
            meat_count = 0
            target_meat = None
            target_state = 'BUY' # BUY, CHOP, COOK
            
            # Count meats in hands
            for b_id in [self.active_bot_id, self.inactive_bot_id]:
                if b_id is None: continue
                b_state = controller.get_bot_state(b_id)
                h = b_state.get('holding')
                if h and h.get('type') == 'Food' and h.get('food_name') == 'MEAT':
                    meat_count += 1
            
            # Count meats on all staging surfaces
            for c_pos in (self.counters + self.sink_tables):
                 tile = controller.get_tile(team, c_pos[0], c_pos[1])
                 if tile and isinstance(getattr(tile, 'item', None), Food):
                     f = tile.item
                     if f.food_name == 'MEAT':
                         meat_count += 1
                         if target_meat is None:
                             target_meat = c_pos
                             if f.chopped: target_state = 'COOK'
                             else: target_state = 'CHOP'
            
            # If no meat found (or all cooked), and count < 2, buy more
            # RESTORED BUFFER to 2 now that we have SINKTABLE for storage
            if meat_count < 2:
                shop = self._get_nearest(bot_pos, self.shops)
                if shop:
                    if self._move_toward(controller, self.active_bot_id, shop, team):
                        ft = getattr(FoodType, 'MEAT', None)
                        if ft:
                            controller.buy(self.active_bot_id, ft, shop[0], shop[1])
            return

    
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

        # RE-SCAN MAP: Turn 101 (Map Switch happens at 100)
        # This handles the layout change specified in game constants
        if controller.get_turn() == 101:
            self._log("ACTION: Map Switch Detected (Turn 101) - Re-scanning Map")
            self._init_map(controller, team)
        
        goal_state = self.goal.get('state') if self.goal else "NO_GOAL"
        goal_id = self.goal.get('order', {}).get('order_id') if self.goal else "-"
        self._log(f"TRACE T{controller.get_turn()}: State={goal_state}, Order={goal_id}, Holding={controller.get_bot_state(self.active_bot_id).get('holding')}")
        
        if self._handle_handoff_pickup(controller, team):
            self._log("ACTION: Handling handoff pickup, ending turn")
            return
        
        if self._handle_inactive_bot_storage(controller, team):
            self._log("ACTION: Handling inactive bot storage, ending turn")
            return



        
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
                    self.goal = self._create_goal(matching_order, controller)
                    self.goal['plate_counter'] = self.abandoned_plate
                    self.goal['plate_contents'] = plate_contents
                    # Remove already-on-plate items from queues
                    for item in plate_contents:
                        for queue_name in ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']:
                            queue = self.goal.get(queue_name, [])
                            if item in queue:
                                queue.remove(item)
                    
                    # Safety checks for low-resource maps (like map5_grind, 1 counter)
                    # If we have limited counters and NO plate on counter, do NOT work on 'simple' or 'chop_only' items.
                    # Prioritize 'chop_cook' (uses cooker, frees counter) or Plating.
                    low_resource_mode = (len(self.counters) <= 1)
                    plate_ready = (self.goal.get('plate_counter') is not None)
                    
                    queues_to_check = ['chop_cook_queue', 'chop_only_queue', 'cook_only_queue', 'simple_queue']
                    
                    for q_name in queues_to_check:
                        queue = self.goal.get(q_name, [])
                        if not queue:
                            continue
                        
                        # Check locks
                        if low_resource_mode and not plate_ready:
                            if q_name in ['simple_queue', 'chop_only_queue']:
                                self._log(f"  INIT: Low resource mode, skipping {q_name} until plate is placed")
                                # Clear the queue to effectively block it
                                self.goal[q_name] = []
                                continue
                    
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
            self.goal = self._create_goal(order, controller)
        
        blocked = self._execute_goal(controller, team)
        
        # Run Active Support AFTER goal execution so it reacts to state changes (e.g. Plate Placed) immediately
        if self.goal is not None and self.inactive_bot_id is not None:
             self._do_active_support(controller, team)

        if blocked:
            self._log(f"BLOCKED in state {self.goal.get('state') if self.goal else 'N/A'}")
            if self.inactive_bot_id is not None:
                self._log("Attempting handoff/swap")
                self._handle_blocked(controller, team)
