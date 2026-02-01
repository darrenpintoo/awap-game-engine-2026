"""
Single-Active-Bot Controller
=============================

STRICT RULE: Only ONE bot issues any actions per turn.
When blocked, control swaps to the other bot with goal state preserved.

Handoff Protocol:
- If active bot is blocked AND holding an item needed for the goal:
  1. Find candidate drop tiles where place() would succeed
  2. Filter to tiles reachable by BOTH bots via BFS
  3. Move adjacent, call place(), set handoff_drop_tile
  4. Swap control atomically
  5. New active bot picks up and resumes goal

Pipeline States (0-11):
0: Ensure pan on cooker if order needs cooking
1: Get plate (sink table or buy)
2: Place plate on counter
3: Process ingredients (check what's missing, buy next)
4: Place ingredient for chopping
5: Chop and pickup
6: Start cook (place food in pan)
7: Wait for cook to complete
8: Take from pan
9: Add food to plate
10: Pickup completed plate
11: Submit order
"""

from collections import deque
from typing import Optional, List, Dict, Tuple, Set, Any

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
        """BFS from start to ANY position in goal_adjacent set.
        
        Returns list of (dx, dy) moves, or None if no path.
        Empty list means already adjacent.
        """
        if start in goal_adjacent:
            return []  # Already adjacent
        
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
        return None  # No path
    
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
        
        # Get occupied positions (other bot)
        occupied = set()
        other_pos = self._get_other_bot_pos(controller)
        if other_pos:
            occupied.add(other_pos)
        
        # Find adjacent walkable tiles to target
        goal_adjacent = self._get_adjacent_walkable(target, game_map)
        if not goal_adjacent:
            return False  # No accessible positions adjacent to target
        
        # BFS to find path
        path = self._bfs_path(bot_pos, goal_adjacent, occupied, game_map)
        if path is None:
            return False  # No path found
        if not path:
            return True  # Already adjacent
        
        # Take first step
        dx, dy = path[0]
        if controller.can_move(bot_id, dx, dy):
            controller.move(bot_id, dx, dy)
        
        return False
    
    def _is_blocked(self, controller: RobotController, bot_id: int, 
                    target: Tuple[int, int], team: Team) -> bool:
        """Check if bot is blocked from reaching target."""
        bot_pos = self._get_bot_pos(controller, bot_id)
        
        if self._chebyshev_dist(bot_pos, target) <= 1:
            return False  # Already adjacent
        
        game_map = controller.get_map(team)
        occupied = set()
        other_pos = self._get_other_bot_pos(controller)
        if other_pos:
            occupied.add(other_pos)
        
        goal_adjacent = self._get_adjacent_walkable(target, game_map)
        if not goal_adjacent:
            return True  # No accessible positions
        
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
        
        # COUNTER: must be empty
        if tile_name == "COUNTER":
            return getattr(tile, 'item', None) is None
        
        # BOX: accepts if empty, or matching signature
        if tile_name == "BOX":
            count = getattr(tile, 'count', 0)
            if count == 0:
                return True
            box_item = getattr(tile, 'item', None)
            if box_item is None:
                return True
            # Check signature match (simplified)
            my_sig = self._item_signature(holding)
            # Convert tile item to holding format
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
                box_sig = ('Pan', None)  # Simplified
            return my_sig == box_sig
        
        # COOKER: Special rules
        if tile_name == "COOKER":
            if holding.get('type') == 'Pan':
                # Pan swap only if existing pan has no food
                existing_pan = getattr(tile, 'item', None)
                if isinstance(existing_pan, Pan):
                    return existing_pan.food is None
                return True
            if holding.get('type') == 'Food':
                # Food only if pan exists, is empty, and food is cookable
                existing_pan = getattr(tile, 'item', None)
                if not isinstance(existing_pan, Pan):
                    return False
                if existing_pan.food is not None:
                    return False
                food_name = holding.get('food_name')
                return INGREDIENT_INFO.get(food_name, {}).get('cook', False)
            return False
        
        # TRASH: always works
        if tile_name == "TRASH":
            return True
        
        # SUBMIT: only clean plates
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
        
        # Get reachable positions for active bot (inactive is obstacle)
        active_reachable = self._bfs_reachable_positions(
            active_pos, {inactive_pos}, game_map)
        
        # Get reachable positions for inactive bot (active is obstacle)
        inactive_reachable = self._bfs_reachable_positions(
            inactive_pos, {active_pos}, game_map)
        
        # Find candidate drop tiles
        candidates = []
        all_placeable = self.counters + self.trashes
        
        for tile_pos in all_placeable:
            # Check if place would succeed
            if not self._can_place_on_tile(controller, team, tile_pos, holding):
                continue
            
            # Check if ADJACENT positions exist that both bots can reach
            adj_walkable = self._get_adjacent_walkable(tile_pos, game_map)
            
            # Need at least one adj position reachable by active AND one by inactive
            active_can_reach = any(p in active_reachable for p in adj_walkable)
            inactive_can_reach = any(p in inactive_reachable for p in adj_walkable)
            
            if active_can_reach and inactive_can_reach:
                dist = self._chebyshev_dist(active_pos, tile_pos)
                candidates.append((dist, tile_pos))
        
        if not candidates:
            return None
        
        # Pick closest by Chebyshev distance
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _swap_control(self) -> None:
        """Atomically swap active/inactive bot IDs."""
        self.active_bot_id, self.inactive_bot_id = self.inactive_bot_id, self.active_bot_id
    
    def _handle_blocked(self, controller: RobotController, team: Team) -> None:
        """Handle blocked state - handoff or swap."""
        bot = controller.get_bot_state(self.active_bot_id)
        holding = bot.get('holding')
        
        if holding is not None:
            # Find drop tile for handoff
            drop_tile = self._find_handoff_drop_tile(controller, team, holding)
            if drop_tile is not None:
                bot_pos = self._get_bot_pos(controller, self.active_bot_id)
                
                if self._chebyshev_dist(bot_pos, drop_tile) <= 1:
                    # Already adjacent - drop and swap
                    if controller.place(self.active_bot_id, drop_tile[0], drop_tile[1]):
                        if self.goal is None:
                            self.goal = {}
                        self.goal['handoff_drop_tile'] = drop_tile
                        self._swap_control()
                else:
                    # Move toward drop tile
                    self._move_toward(controller, self.active_bot_id, drop_tile, team)
                return
        
        # Either not holding anything, or no suitable drop tile found
        self._swap_control()
    
    def _select_order(self, controller: RobotController, team: Team) -> Optional[Dict]:
        """Select best order to work on."""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 40:  # Need enough time to complete
                continue
            
            required = order['required']
            
            # Score: prefer simpler orders with higher reward
            score = 200 - len(required) * 30  # Fewer ingredients = higher score
            score += order['reward'] / 5
            
            # Penalize orders requiring cooking (more complex)
            needs_cooking = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in required)
            if needs_cooking:
                score -= 20
            
            # Penalize orders requiring chopping
            needs_chopping = any(INGREDIENT_INFO.get(i, {}).get('chop') for i in required)
            if needs_chopping:
                score -= 10
            
            if score > best_score:
                best_score = score
                best = order
        
        return best
    
    def _create_goal(self, order: Dict) -> Dict[str, Any]:
        """Create a new goal state for an order."""
        self.goal_counter += 1
        
        # Check if order needs cooking
        needs_cooking = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in order['required'])
        
        return {
            'goal_id': self.goal_counter,
            'goal_type': 'deliver_order',
            'order': order,
            'state': 0 if needs_cooking else 1,  # Skip pan check if no cooking
            'plate_counter': None,
            'work_counter': None,
            'current_ingredient': None,
            'handoff_drop_tile': None,
            'processed_ingredients': [],
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
        state = self.goal.get('state', 0)
        
        if order is None:
            return False
        
        shop = self._get_nearest(bot_pos, self.shops)
        submit = self._get_nearest(bot_pos, self.submits)
        
        # STATE 0: Ensure pan on cooker if needed
        if state == 0:
            if not self.cookers:
                self.goal['state'] = 1
                return False
            
            cooker = self.cookers[0]
            tile = controller.get_tile(team, cooker[0], cooker[1])
            
            # Check if cooker already has pan
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                self.goal['state'] = 1
                return False
            
            # Need to get a pan
            if holding and holding.get('type') == 'Pan':
                # Have pan, place it
                if self._is_blocked(controller, self.active_bot_id, cooker, team):
                    return True
                if self._move_toward(controller, self.active_bot_id, cooker, team):
                    controller.place(self.active_bot_id, cooker[0], cooker[1])
                    self.goal['state'] = 1
            else:
                # Buy pan
                if shop is None:
                    return True
                if self._is_blocked(controller, self.active_bot_id, shop, team):
                    return True
                if self._move_toward(controller, self.active_bot_id, shop, team):
                    if money >= ShopCosts.PAN.buy_cost:
                        controller.buy(self.active_bot_id, ShopCosts.PAN, shop[0], shop[1])
            return False
        
        # STATE 1: Get plate
        if state == 1:
            if holding and holding.get('type') == 'Plate':
                self.goal['state'] = 2
                return False
            
            # Try sink table first
            if self.sink_tables:
                for st in self.sink_tables:
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self._is_blocked(controller, self.active_bot_id, st, team):
                            return True
                        if self._move_toward(controller, self.active_bot_id, st, team):
                            controller.take_clean_plate(self.active_bot_id, st[0], st[1])
                        return False
            
            # Buy plate
            if shop is None:
                return True
            if self._is_blocked(controller, self.active_bot_id, shop, team):
                return True
            if self._move_toward(controller, self.active_bot_id, shop, team):
                if money >= ShopCosts.PLATE.buy_cost:
                    controller.buy(self.active_bot_id, ShopCosts.PLATE, shop[0], shop[1])
            return False
        
        # STATE 2: Place plate on counter
        if state == 2:
            exclude = set()
            if self.goal.get('work_counter'):
                exclude.add(self.goal['work_counter'])
            
            counter = self.goal.get('plate_counter')
            if counter is None:
                counter = self._get_free_counter(controller, team, bot_pos, exclude)
                if counter is None:
                    return True  # No free counter
                self.goal['plate_counter'] = counter
            
            if self._is_blocked(controller, self.active_bot_id, counter, team):
                return True
            if self._move_toward(controller, self.active_bot_id, counter, team):
                controller.place(self.active_bot_id, counter[0], counter[1])
                self.goal['state'] = 3
            return False
        
        # STATE 3: Process ingredients - check what's missing, buy next
        if state == 3:
            plate_counter = self.goal.get('plate_counter')
            if plate_counter is None:
                self.goal['state'] = 1
                return False
            
            # Get plate contents
            plate_tile = controller.get_tile(team, plate_counter[0], plate_counter[1])
            plate_contents = []
            if plate_tile and isinstance(getattr(plate_tile, 'item', None), Plate):
                for f in plate_tile.item.food:
                    plate_contents.append(f.food_name)
            
            # Check processed ingredients
            processed = self.goal.get('processed_ingredients', [])
            
            # Find missing ingredients
            required = order['required']
            missing = []
            for ing in required:
                if ing not in plate_contents and ing not in processed:
                    missing.append(ing)
            
            if not missing:
                # All ingredients on plate, pickup and submit
                self.goal['state'] = 10
                return False
            
            # Pick next ingredient (prefer non-cooking first for simplicity)
            next_ing = None
            for ing in missing:
                info = INGREDIENT_INFO.get(ing, {})
                if not info.get('cook'):
                    next_ing = ing
                    break
            if next_ing is None:
                next_ing = missing[0]
            
            self.goal['current_ingredient'] = next_ing
            
            # Check what we're holding
            if holding:
                if holding.get('type') == 'Food' and holding.get('food_name') == next_ing:
                    info = INGREDIENT_INFO.get(next_ing, {})
                    if info.get('chop') and not holding.get('chopped'):
                        self.goal['state'] = 4  # Need to chop
                    elif info.get('cook') and holding.get('cooked_stage', 0) == 0:
                        self.goal['state'] = 6  # Need to cook
                    else:
                        self.goal['state'] = 9  # Ready to add to plate
                else:
                    # Holding wrong item - need to trash or place somewhere
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash:
                            if self._move_toward(controller, self.active_bot_id, trash, team):
                                controller.trash(self.active_bot_id, trash[0], trash[1])
                    return False
            else:
                # Buy ingredient
                if shop is None:
                    return True
                if self._is_blocked(controller, self.active_bot_id, shop, team):
                    return True
                if self._move_toward(controller, self.active_bot_id, shop, team):
                    food_type = getattr(FoodType, next_ing, None)
                    if food_type and money >= food_type.buy_cost:
                        controller.buy(self.active_bot_id, food_type, shop[0], shop[1])
            return False
        
        # STATE 4: Place ingredient for chopping
        if state == 4:
            exclude = set()
            if self.goal.get('plate_counter'):
                exclude.add(self.goal['plate_counter'])
            
            work_counter = self.goal.get('work_counter')
            if work_counter is None:
                work_counter = self._get_free_counter(controller, team, bot_pos, exclude)
                if work_counter is None:
                    return True
                self.goal['work_counter'] = work_counter
            
            if self._is_blocked(controller, self.active_bot_id, work_counter, team):
                return True
            if self._move_toward(controller, self.active_bot_id, work_counter, team):
                controller.place(self.active_bot_id, work_counter[0], work_counter[1])
                self.goal['state'] = 5
            return False
        
        # STATE 5: Chop and pickup
        if state == 5:
            work_counter = self.goal.get('work_counter')
            if work_counter is None:
                self.goal['state'] = 3
                return False
            
            if self._is_blocked(controller, self.active_bot_id, work_counter, team):
                return True
            
            if holding is None:
                # Check if food is chopped
                tile = controller.get_tile(team, work_counter[0], work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    food = tile.item
                    if self._move_toward(controller, self.active_bot_id, work_counter, team):
                        if food.chopped:
                            controller.pickup(self.active_bot_id, work_counter[0], work_counter[1])
                            self.goal['work_counter'] = None
                            ing = self.goal.get('current_ingredient')
                            info = INGREDIENT_INFO.get(ing, {})
                            if info.get('cook'):
                                self.goal['state'] = 6
                            else:
                                self.goal['state'] = 9
                        else:
                            controller.chop(self.active_bot_id, work_counter[0], work_counter[1])
                else:
                    # Food gone? Go back to state 3
                    self.goal['work_counter'] = None
                    self.goal['state'] = 3
            else:
                # Already holding chopped food
                ing = self.goal.get('current_ingredient')
                info = INGREDIENT_INFO.get(ing, {})
                if info.get('cook'):
                    self.goal['state'] = 6
                else:
                    self.goal['state'] = 9
            return False
        
        # STATE 6: Start cook (place food in pan)
        if state == 6:
            if not self.cookers:
                # No cooker, skip cooking (shouldn't happen if we selected right order)
                self.goal['state'] = 9
                return False
            
            cooker = self.cookers[0]
            
            if self._is_blocked(controller, self.active_bot_id, cooker, team):
                return True
            
            if holding and holding.get('type') == 'Food':
                if self._move_toward(controller, self.active_bot_id, cooker, team):
                    controller.place(self.active_bot_id, cooker[0], cooker[1])
                    self.goal['state'] = 7
            else:
                # Not holding food anymore, go back
                self.goal['state'] = 3
            return False
        
        # STATE 7: Wait for cook to complete
        if state == 7:
            if not self.cookers:
                self.goal['state'] = 9
                return False
            
            cooker = self.cookers[0]
            tile = controller.get_tile(team, cooker[0], cooker[1])
            
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food:
                    if pan.food.cooked_stage >= 1:
                        self.goal['state'] = 8
                        return False
            # Just wait
            return False
        
        # STATE 8: Take from pan
        if state == 8:
            if not self.cookers:
                self.goal['state'] = 3
                return False
            
            cooker = self.cookers[0]
            
            if holding:
                # Check if it's cooked food
                if holding.get('type') == 'Food' and holding.get('cooked_stage', 0) >= 1:
                    if holding.get('cooked_stage') == 2:
                        # Burnt - trash it
                        if self.trashes:
                            trash = self._get_nearest(bot_pos, self.trashes)
                            if trash:
                                if self._move_toward(controller, self.active_bot_id, trash, team):
                                    controller.trash(self.active_bot_id, trash[0], trash[1])
                                    self.goal['state'] = 3
                        return False
                    else:
                        # Properly cooked
                        self.goal['state'] = 9
                return False
            
            if self._is_blocked(controller, self.active_bot_id, cooker, team):
                return True
            
            if self._move_toward(controller, self.active_bot_id, cooker, team):
                controller.take_from_pan(self.active_bot_id, cooker[0], cooker[1])
            return False
        
        # STATE 9: Add food to plate
        if state == 9:
            plate_counter = self.goal.get('plate_counter')
            if plate_counter is None:
                self.goal['state'] = 3
                return False
            
            if self._is_blocked(controller, self.active_bot_id, plate_counter, team):
                return True
            
            if holding and holding.get('type') == 'Food':
                if self._move_toward(controller, self.active_bot_id, plate_counter, team):
                    if controller.add_food_to_plate(self.active_bot_id, 
                                                     plate_counter[0], plate_counter[1]):
                        # Mark this ingredient as processed
                        ing = self.goal.get('current_ingredient')
                        if ing:
                            processed = self.goal.get('processed_ingredients', [])
                            processed.append(ing)
                            self.goal['processed_ingredients'] = processed
                        self.goal['current_ingredient'] = None
                        self.goal['state'] = 3  # Check for more ingredients
            else:
                # Not holding food, go back
                self.goal['state'] = 3
            return False
        
        # STATE 10: Pickup completed plate
        if state == 10:
            plate_counter = self.goal.get('plate_counter')
            if plate_counter is None:
                self.goal['state'] = 1
                return False
            
            if holding and holding.get('type') == 'Plate':
                self.goal['state'] = 11
                return False
            
            if self._is_blocked(controller, self.active_bot_id, plate_counter, team):
                return True
            
            if self._move_toward(controller, self.active_bot_id, plate_counter, team):
                controller.pickup(self.active_bot_id, plate_counter[0], plate_counter[1])
                self.goal['state'] = 11
            return False
        
        # STATE 11: Submit order
        if state == 11:
            if submit is None:
                return True
            
            if self._is_blocked(controller, self.active_bot_id, submit, team):
                return True
            
            if self._move_toward(controller, self.active_bot_id, submit, team):
                if controller.submit(self.active_bot_id, submit[0], submit[1]):
                    # Order complete!
                    self.goal = None
            return False
        
        return False
    
    def _handle_handoff_pickup(self, controller: RobotController, team: Team) -> bool:
        """Handle picking up from handoff drop tile. Returns True if still handling."""
        if self.goal is None:
            return False
        
        drop_tile = self.goal.get('handoff_drop_tile')
        if drop_tile is None:
            return False
        
        bot = controller.get_bot_state(self.active_bot_id)
        holding = bot.get('holding')
        
        if holding is not None:
            # Already picked up, clear handoff state
            self.goal['handoff_drop_tile'] = None
            return False
        
        bot_pos = (bot['x'], bot['y'])
        
        # Check if item is still there
        tile = controller.get_tile(team, drop_tile[0], drop_tile[1])
        if tile is None or getattr(tile, 'item', None) is None:
            # Item gone, clear handoff
            self.goal['handoff_drop_tile'] = None
            return False
        
        # Navigate to drop tile and pickup
        if self._chebyshev_dist(bot_pos, drop_tile) <= 1:
            controller.pickup(self.active_bot_id, drop_tile[0], drop_tile[1])
            self.goal['handoff_drop_tile'] = None
            return False
        else:
            if self._is_blocked(controller, self.active_bot_id, drop_tile, team):
                # Can't reach drop tile - swap back
                self._swap_control()
                return True
            self._move_toward(controller, self.active_bot_id, drop_tile, team)
            return True
    
    def play_turn(self, controller: RobotController) -> None:
        """Main entry point - called each turn."""
        team = controller.get_team()
        self.team = team
        
        # Initialize map features on first turn
        if not self.initialized:
            self._init_map(controller, team)
        
        # Get bot IDs
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Initialize active/inactive bot IDs
        if self.active_bot_id is None:
            self.active_bot_id = bots[0]
            if len(bots) > 1:
                self.inactive_bot_id = bots[1]
        
        # Handle handoff pickup first
        if self._handle_handoff_pickup(controller, team):
            return
        
        # Ensure we have a goal
        if self.goal is None:
            order = self._select_order(controller, team)
            if order is None:
                return  # No suitable orders
            self.goal = self._create_goal(order)
        
        # Execute goal pipeline
        blocked = self._execute_goal(controller, team)
        
        # Handle blocked state
        if blocked and self.inactive_bot_id is not None:
            self._handle_blocked(controller, team)
