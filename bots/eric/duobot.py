"""
DuoBot Controller - Dual Active Bot System
============================================

Both bots work simultaneously with independent goals.
Features:
- Dynamic task assignment (proximity + holding + progress scoring)
- Resource reservation to prevent conflicts
- Deadlock detection and resolution
- Parallel cooking (do other tasks while waiting for cook)
- Handoff support for isolated/disjoint map zones
- Zone detection at startup
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
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "duobot_debug.log")


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
        
        # Bot management - both bots active
        self.bot_ids: List[int] = []
        self.bot_goals: Dict[int, Optional[Dict[str, Any]]] = {}  # {bot_id: goal}
        self.goal_counter: int = 0
        
        # Resource reservations
        self.reserved_counters: Dict[Tuple[int, int], int] = {}  # {pos: bot_id}
        self.reserved_cookers: Dict[Tuple[int, int], int] = {}   # {pos: bot_id}
        
        # Order tracking
        self.orders_in_progress: Dict[int, int] = {}  # {order_id: bot_id}
        self.completed_order_ids: Set[int] = set()
        
        # Zone detection (for isolated maps)
        self.same_zone: bool = True
        self.bot_zones: Dict[int, Set[Tuple[int, int]]] = {}  # {bot_id: reachable_positions}
        self.handoff_counters: List[Tuple[int, int]] = []  # Counters bridging zones
        
        # Cooking tracking for parallel tasks
        self.cooking_wait: Dict[int, Dict] = {}  # {bot_id: {'cooker': pos, 'ingredient': str, 'done_turn': int}}
        
        # Handoff state (for isolated maps)
        self.pending_handoffs: Dict[int, Dict] = {}  # {bot_id: handoff_info}
        
        # Blocked tracking - to detect bots stuck for too long
        self.blocked_turns: Dict[int, int] = {}  # {bot_id: consecutive_blocked_turns}
        
        # Debug logging
        self.log_initialized = False
    
    # ========== LOGGING ==========
    
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
                f.write("=== DuoBot Debug Log ===\n\n")
            self.log_initialized = True
        except Exception:
            pass
    
    def _log_state(self, controller: RobotController, team: Team) -> None:
        """Log comprehensive state information."""
        if not DEBUG_LOG_ENABLED:
            return
        
        turn = controller.get_turn()
        money = controller.get_team_money(team)
        
        lines = [
            f"\n{'='*60}",
            f"TURN {turn} | Money: ${money} | Same Zone: {self.same_zone}",
            f"{'='*60}",
        ]
        
        # Bot positions and goals
        for bot_id in self.bot_ids:
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
            
            goal = self.bot_goals.get(bot_id)
            goal_str = "None"
            if goal:
                order = goal.get('order', {})
                goal_str = f"Order #{order.get('order_id')} state={goal.get('state')}"
            
            lines.append(f"  Bot {bot_id}: pos={pos}, holding={holding_str}")
            lines.append(f"    Goal: {goal_str}")
            
            # Cooking wait status
            if bot_id in self.cooking_wait:
                cw = self.cooking_wait[bot_id]
                lines.append(f"    Cooking: {cw.get('ingredient')} at {cw.get('cooker')}, done turn {cw.get('done_turn')}")
        
        # Reservations
        if self.reserved_counters:
            lines.append(f"\nReserved counters: {self.reserved_counters}")
        if self.reserved_cookers:
            lines.append(f"Reserved cookers: {self.reserved_cookers}")
        
        # Orders in progress
        if self.orders_in_progress:
            lines.append(f"Orders in progress: {self.orders_in_progress}")
        
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
        
        self._log('\n'.join(lines))
    
    # ========== MAP INITIALIZATION ==========
    
    def _init_map(self, controller: RobotController, team: Team) -> None:
        """Cache map feature positions."""
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
        
        self._log(f"MAP INIT: counters={self.counters}")
        self._log(f"MAP INIT: cookers={self.cookers}")
        self._log(f"MAP INIT: shops={self.shops}")
        self._log(f"MAP INIT: submits={self.submits}")
        self._log(f"MAP INIT: trashes={self.trashes}")
        self._log(f"MAP INIT: sink_tables={self.sink_tables}")
    
    def _detect_zones(self, controller: RobotController, team: Team) -> None:
        """Detect if bots are in separate zones (disjoint areas)."""
        if len(self.bot_ids) < 2:
            self.same_zone = True
            return
        
        game_map = controller.get_map(team)
        
        bot1_pos = self._get_bot_pos(controller, self.bot_ids[0])
        bot2_pos = self._get_bot_pos(controller, self.bot_ids[1])
        
        # Get reachable positions for each bot (ignoring the other bot)
        zone1 = self._bfs_reachable_positions(bot1_pos, set(), game_map)
        zone2 = self._bfs_reachable_positions(bot2_pos, set(), game_map)
        
        self.bot_zones[self.bot_ids[0]] = zone1
        self.bot_zones[self.bot_ids[1]] = zone2
        
        # Check if bots can reach each other
        self.same_zone = bot2_pos in zone1
        
        if not self.same_zone:
            # Find handoff counters (counters that bridge the two zones)
            self.handoff_counters = []
            for counter in self.counters:
                adj = self._get_adjacent_walkable(counter, game_map)
                zone1_adj = any(p in zone1 for p in adj)
                zone2_adj = any(p in zone2 for p in adj)
                if zone1_adj and zone2_adj:
                    self.handoff_counters.append(counter)
            
            self._log(f"ZONES: Bots in SEPARATE zones! Handoff counters: {self.handoff_counters}")
        else:
            self._log(f"ZONES: Bots in same zone")
    
    # ========== UTILITY FUNCTIONS ==========
    
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
    
    def _get_other_bot_id(self, bot_id: int) -> Optional[int]:
        """Get the other bot's ID."""
        for bid in self.bot_ids:
            if bid != bot_id:
                return bid
        return None
    
    def _get_other_bot_pos(self, controller: RobotController, bot_id: int) -> Optional[Tuple[int, int]]:
        """Get position of the other bot."""
        other_id = self._get_other_bot_id(bot_id)
        if other_id is None:
            return None
        return self._get_bot_pos(controller, other_id)
    
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
        other_pos = self._get_other_bot_pos(controller, bot_id)
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
        other_pos = self._get_other_bot_pos(controller, bot_id)
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
    
    # ========== RESOURCE MANAGEMENT ==========
    
    def _get_free_counter(self, controller: RobotController, team: Team, 
                          bot_id: int, pos: Tuple[int, int], 
                          exclude: Optional[Set[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """Find nearest free counter, respecting reservations."""
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            # Check reservations - skip if reserved by other bot
            reserved_by = self.reserved_counters.get(c)
            if reserved_by is not None and reserved_by != bot_id:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
        return self._get_nearest(pos, free)
    
    def _get_available_cooker(self, controller: RobotController, team: Team,
                               bot_id: int, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest cooker with empty pan, respecting reservations."""
        available = []
        for c in self.cookers:
            # Check reservations - skip if reserved by other bot
            reserved_by = self.reserved_cookers.get(c)
            if reserved_by is not None and reserved_by != bot_id:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    available.append(c)
        return self._get_nearest(pos, available)
    
    def _reserve_counter(self, pos: Tuple[int, int], bot_id: int) -> None:
        """Reserve a counter for a bot."""
        self.reserved_counters[pos] = bot_id
    
    def _reserve_cooker(self, pos: Tuple[int, int], bot_id: int) -> None:
        """Reserve a cooker for a bot."""
        self.reserved_cookers[pos] = bot_id
    
    def _release_reservations(self, bot_id: int) -> None:
        """Release all reservations held by a bot."""
        self.reserved_counters = {k: v for k, v in self.reserved_counters.items() if v != bot_id}
        self.reserved_cookers = {k: v for k, v in self.reserved_cookers.items() if v != bot_id}
    
    def _can_bot_reach(self, controller: RobotController, bot_id: int, 
                       target: Tuple[int, int], team: Team) -> bool:
        """Check if bot can reach a target (in its zone)."""
        if self.same_zone:
            return not self._is_blocked(controller, bot_id, target, team)
        
        # Check if target is in bot's zone
        game_map = controller.get_map(team)
        target_adj = self._get_adjacent_walkable(target, game_map)
        bot_zone = self.bot_zones.get(bot_id, set())
        return any(p in bot_zone for p in target_adj)
    
    # ========== DEADLOCK DETECTION AND RESOLUTION ==========
    
    def _detect_deadlock(self, controller: RobotController, team: Team) -> Optional[Tuple[int, int]]:
        """Detect if both bots are blocking each other. Returns (yielding_bot, priority_bot) or None."""
        if len(self.bot_ids) < 2:
            return None
        
        bot1, bot2 = self.bot_ids[0], self.bot_ids[1]
        pos1 = self._get_bot_pos(controller, bot1)
        pos2 = self._get_bot_pos(controller, bot2)
        
        goal1 = self.bot_goals.get(bot1)
        goal2 = self.bot_goals.get(bot2)
        
        if goal1 is None or goal2 is None:
            return None
        
        # Check if bot1 is blocked by bot2 and vice versa
        game_map = controller.get_map(team)
        
        # Get target for each bot based on their current state
        target1 = self._get_current_target(controller, bot1, goal1, team)
        target2 = self._get_current_target(controller, bot2, goal2, team)
        
        if target1 is None or target2 is None:
            return None
        
        # Check if they're blocking each other
        # Bot1 blocked by bot2: path to target1 goes through pos2
        path1_without_bot2 = self._bfs_path(pos1, self._get_adjacent_walkable(target1, game_map), set(), game_map)
        path1_with_bot2 = self._bfs_path(pos1, self._get_adjacent_walkable(target1, game_map), {pos2}, game_map)
        
        path2_without_bot1 = self._bfs_path(pos2, self._get_adjacent_walkable(target2, game_map), set(), game_map)
        path2_with_bot1 = self._bfs_path(pos2, self._get_adjacent_walkable(target2, game_map), {pos1}, game_map)
        
        bot1_blocked_by_2 = path1_with_bot2 is None and path1_without_bot2 is not None
        bot2_blocked_by_1 = path2_with_bot1 is None and path2_without_bot1 is not None
        
        if bot1_blocked_by_2 and bot2_blocked_by_1:
            # Deadlock! Determine priority
            # Priority based on order urgency (expires sooner = higher priority)
            order1 = goal1.get('order', {})
            order2 = goal2.get('order', {})
            expires1 = order1.get('expires_turn', 99999)
            expires2 = order2.get('expires_turn', 99999)
            
            if expires1 < expires2:
                return (bot2, bot1)  # bot2 yields, bot1 has priority
            elif expires2 < expires1:
                return (bot1, bot2)  # bot1 yields, bot2 has priority
            else:
                # Same urgency - lower bot_id has priority
                return (bot2, bot1) if bot1 < bot2 else (bot1, bot2)
        
        return None
    
    def _get_current_target(self, controller: RobotController, bot_id: int, 
                            goal: Dict, team: Team) -> Optional[Tuple[int, int]]:
        """Get the current target position for a bot based on its goal state."""
        if goal is None:
            return None
        
        state = goal.get('state', 'INIT')
        bot_pos = self._get_bot_pos(controller, bot_id)
        
        if state == 'BUY_INGREDIENT':
            return self._get_nearest(bot_pos, self.shops)
        elif state == 'ENSURE_PLATE':
            if self.sink_tables:
                return self._get_nearest(bot_pos, self.sink_tables)
            return self._get_nearest(bot_pos, self.shops)
        elif state in ['CHOP_THEN_COOK', 'CHOP_THEN_STORE']:
            work_counter = goal.get('work_counter')
            if work_counter:
                return work_counter
            return self._get_free_counter(controller, team, bot_id, bot_pos)
        elif state in ['START_COOK', 'COOK_ONLY', 'PICKUP_STORED_FOR_COOK']:
            return self._get_available_cooker(controller, team, bot_id, bot_pos)
        elif state == 'SUBMIT':
            return self._get_nearest(bot_pos, self.submits)
        elif state == 'ASSEMBLE':
            plate_counter = goal.get('plate_counter')
            if plate_counter:
                return plate_counter
        elif state == 'PLACE_PLATE':
            return self._get_free_counter(controller, team, bot_id, bot_pos)
        
        return None
    
    def _resolve_deadlock(self, controller: RobotController, team: Team, 
                          yielding_bot: int, priority_bot: int) -> None:
        """Make the yielding bot move aside to let priority bot pass."""
        yielding_pos = self._get_bot_pos(controller, yielding_bot)
        priority_pos = self._get_bot_pos(controller, priority_bot)
        
        game_map = controller.get_map(team)
        
        # Find a position for yielding bot to move to that's not in priority bot's path
        priority_goal = self.bot_goals.get(priority_bot)
        priority_target = self._get_current_target(controller, priority_bot, priority_goal, team)
        
        if priority_target is None:
            return
        
        # Get priority bot's path
        priority_path_tiles = set()
        path = self._bfs_path(priority_pos, self._get_adjacent_walkable(priority_target, game_map), 
                              {yielding_pos}, game_map)
        if path:
            curr = priority_pos
            for dx, dy in path:
                curr = (curr[0] + dx, curr[1] + dy)
                priority_path_tiles.add(curr)
        
        # Find nearest walkable tile not in priority bot's path
        best_escape = None
        best_dist = 99999
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = yielding_pos[0] + dx, yielding_pos[1] + dy
                if not game_map.is_tile_walkable(nx, ny):
                    continue
                if (nx, ny) == priority_pos:
                    continue
                if (nx, ny) in priority_path_tiles:
                    continue
                # Good escape position
                dist = self._chebyshev_dist((nx, ny), yielding_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_escape = (nx, ny)
        
        if best_escape:
            dx = best_escape[0] - yielding_pos[0]
            dy = best_escape[1] - yielding_pos[1]
            if controller.can_move(yielding_bot, dx, dy):
                self._log(f"  DEADLOCK: Bot {yielding_bot} moving aside to {best_escape}")
                controller.move(yielding_bot, dx, dy)
    
    def _move_out_of_way(self, controller: RobotController, bot_id: int, 
                          other_bot: int, team: Team) -> None:
        """Move bot to a position where it won't block the other bot."""
        my_pos = self._get_bot_pos(controller, bot_id)
        other_pos = self._get_bot_pos(controller, other_bot)
        other_goal = self.bot_goals.get(other_bot)
        
        if other_goal is None:
            return
        
        # Get other bot's target
        other_target = self._get_current_target(controller, other_bot, other_goal, team)
        if other_target is None:
            return
        
        game_map = controller.get_map(team)
        
        # Check if I'm blocking the other bot's path
        path_with_me = self._bfs_path(other_pos, self._get_adjacent_walkable(other_target, game_map), 
                                       {my_pos}, game_map)
        path_without_me = self._bfs_path(other_pos, self._get_adjacent_walkable(other_target, game_map), 
                                          set(), game_map)
        
        if path_with_me is not None:
            # I'm not blocking - no need to move
            return
        
        if path_without_me is None:
            # Path doesn't exist anyway
            return
        
        # I'm blocking! Find a position to move to
        # Get tiles in other bot's path
        path_tiles = set()
        curr = other_pos
        for dx, dy in path_without_me:
            curr = (curr[0] + dx, curr[1] + dy)
            path_tiles.add(curr)
        
        # Find adjacent walkable tile not in path
        best_escape = None
        best_dist = 99999
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = my_pos[0] + dx, my_pos[1] + dy
                if not game_map.is_tile_walkable(nx, ny):
                    continue
                if (nx, ny) == other_pos:
                    continue
                if (nx, ny) in path_tiles:
                    continue
                # Good escape position - prefer positions far from other bot's path
                min_dist_to_path = min(self._chebyshev_dist((nx, ny), p) for p in path_tiles) if path_tiles else 99
                if min_dist_to_path > best_dist or best_escape is None:
                    best_dist = min_dist_to_path
                    best_escape = (nx, ny)
        
        if best_escape:
            dx = best_escape[0] - my_pos[0]
            dy = best_escape[1] - my_pos[1]
            if controller.can_move(bot_id, dx, dy):
                self._log(f"  Bot {bot_id}: Moving out of way to {best_escape}")
                controller.move(bot_id, dx, dy)
    
    # ========== ORDER SELECTION ==========
    
    def _get_order_difficulty(self, order: Dict) -> int:
        """Calculate difficulty score for an order (lower = easier)."""
        difficulty = 0
        for ing in order.get('required', []):
            info = INGREDIENT_INFO.get(ing, {})
            needs_chop = info.get('chop', False)
            needs_cook = info.get('cook', False)
            
            if needs_chop and needs_cook:
                difficulty += 5  # MEAT
            elif needs_chop or needs_cook:
                difficulty += 3  # EGG, ONIONS
            else:
                difficulty += 1  # NOODLES, SAUCE
        return difficulty
    
    def _estimate_completion_turns(self, controller: RobotController, bot_id: int, 
                                    order: Dict, team: Team) -> int:
        """Estimate how many turns it will take to complete an order.
        
        Accounts for current bot state (holding plate, plates on counters, etc.)
        """
        required = order.get('required', [])
        
        # Dynamic AVG_MOVE based on map size - be optimistic
        game_map = controller.get_map(team)
        map_diag = max(game_map.width, game_map.height)
        AVG_MOVE = max(2, map_diag // 5)
        
        # Check current state
        bot = controller.get_bot_state(bot_id)
        holding = bot.get('holding') if bot else None
        
        plate_time = 0
        
        # Check if bot already has a plate
        if holding and holding.get('type') == 'Plate':
            plate_time = AVG_MOVE + 1  # Just need to place it
        else:
            # Check if there's a plate on a counter we can use
            plate_on_counter = False
            for c in self.counters:
                tile = controller.get_tile(team, c[0], c[1])
                if tile and hasattr(tile, 'item') and tile.item is not None:
                    item = tile.item
                    if hasattr(item, 'food') and hasattr(item, 'dirty'):  # It's a Plate
                        if not getattr(item, 'dirty', True):
                            plate_on_counter = True
                            break
            
            if plate_on_counter:
                plate_time = 0  # Plate already placed
            else:
                plate_time = AVG_MOVE + 1 + AVG_MOVE + 1  # Get + place
        
        # Track cooking time (can overlap with prep)
        max_cook_time = 0
        prep_time = 0
        
        for ing in required:
            info = INGREDIENT_INFO.get(ing, {})
            needs_chop = info.get('chop', False)
            needs_cook = info.get('cook', False)
            
            # Buy ingredient
            prep_time += AVG_MOVE + 1
            
            if needs_chop and needs_cook:  # MEAT
                prep_time += AVG_MOVE + 1 + 5 + 1  # Place, chop, pickup
                prep_time += AVG_MOVE + 1  # To cooker
                max_cook_time = max(max_cook_time, 10)
                prep_time += 1 + AVG_MOVE + 1  # Take + add to plate
            elif needs_chop:  # ONIONS
                prep_time += AVG_MOVE + 1 + 5 + 1  # Place, chop, pickup
                prep_time += AVG_MOVE + 1  # Add to plate
            elif needs_cook:  # EGG
                prep_time += AVG_MOVE + 1  # To cooker
                max_cook_time = max(max_cook_time, 10)
                prep_time += 1 + AVG_MOVE + 1  # Take + add to plate
            else:  # NOODLES, SAUCE
                prep_time += AVG_MOVE + 1  # Add to plate
        
        # Pickup plate and submit
        prep_time += 1 + AVG_MOVE + 1
        
        # Total: plate_time + prep_time, cook overlaps with prep
        total = plate_time + prep_time + max(0, max_cook_time - prep_time // 2)
        
        return total
    
    def _can_complete_order(self, controller: RobotController, bot_id: int,
                            order: Dict, team: Team) -> bool:
        """Check if bot can realistically complete order before it expires."""
        current_turn = controller.get_turn()
        expires = order.get('expires_turn', 0)
        starts_at = order.get('created_turn', 0)
        
        # Order hasn't started yet - count from when it starts
        effective_start = max(current_turn, starts_at)
        time_available = expires - effective_start
        
        estimated_time = self._estimate_completion_turns(controller, bot_id, order, team)
        
        # Be lenient - no buffer, just check if estimate fits
        return estimated_time <= time_available
    
    def _get_order_profitability(self, order: Dict) -> float:
        """Calculate reward/difficulty ratio (higher = more profitable)."""
        reward = order.get('reward', 0)
        difficulty = self._get_order_difficulty(order)
        if difficulty == 0:
            difficulty = 1
        return reward / difficulty
    
    def _has_enough_space(self, controller: RobotController, order: Dict, team: Team) -> bool:
        """Check if there's enough counter/cooker space for this order."""
        required = order.get('required', [])
        
        # Count resources needed simultaneously
        # 1 counter for plate, 1 counter for chopping, 1 cooker per cooking item
        counters_needed = 1  # For plate
        cookers_needed = 0
        
        for ing in required:
            info = INGREDIENT_INFO.get(ing, {})
            if info.get('chop', False):
                counters_needed += 1  # Need counter to place during chop
            if info.get('cook', False):
                cookers_needed += 1
        
        # Check available resources
        available_counters = 0
        for c in self.counters:
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                available_counters += 1
        
        available_cookers = len(self.cookers)  # Assume all cookers can be used
        
        return available_counters >= counters_needed and available_cookers >= cookers_needed
    
    def _get_order_resource_needs(self, order: Dict) -> Tuple[int, int]:
        """Calculate (counters_needed, cookers_needed) for an order."""
        required = order.get('required', [])
        
        counters_needed = 1  # Plate always needs a counter
        cookers_needed = 0
        
        for ing in required:
            info = INGREDIENT_INFO.get(ing, {})
            if info.get('chop', False):
                counters_needed += 1  # Counter for chopping work
            if info.get('cook', False):
                cookers_needed += 1
        
        return (counters_needed, cookers_needed)
    
    def _get_available_resources(self, controller: RobotController, team: Team) -> Tuple[int, int]:
        """Count currently available (free_counters, free_cookers)."""
        free_counters = 0
        for c in self.counters:
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free_counters += 1
        
        free_cookers = 0
        for c in self.cookers:
            tile = controller.get_tile(team, c[0], c[1])
            if tile:
                pan = getattr(tile, 'pan', None)
                if pan:
                    food = getattr(pan, 'food', None)
                    if food is None:
                        free_cookers += 1
                else:
                    free_cookers += 1  # No pan - can still use
        
        return (free_counters, free_cookers)
    
    def _can_both_bots_work_independently(self, controller: RobotController, bot_id: int, 
                                           new_order: Dict, team: Team) -> bool:
        """Check if this bot can take new_order while other bot continues its order."""
        other_bot = self._get_other_bot_id(bot_id)
        other_goal = self.bot_goals.get(other_bot) if other_bot else None
        
        # If other bot has no goal, we can work freely
        if not other_goal:
            return True
        
        other_order = other_goal.get('order', {})
        
        # Calculate combined resource needs
        my_needs = self._get_order_resource_needs(new_order)
        other_needs = self._get_order_resource_needs(other_order)
        
        total_counters_needed = my_needs[0] + other_needs[0]
        total_cookers_needed = my_needs[1] + other_needs[1]
        
        # Check against TOTAL resources (not just free - we need capacity for both)
        total_counters = len(self.counters)
        total_cookers = len(self.cookers)
        
        # Allow some slack - counters can be reused after items are added to plate
        # But cookers are blocking while cooking (10 turns)
        if total_cookers_needed > total_cookers:
            self._log(f"    Bot {bot_id}: Can't work independently - need {total_cookers_needed} cookers, have {total_cookers}")
            return False
        
        # For counters, be more lenient - items move through quickly
        # But if both need lots of counters simultaneously, it's a problem
        if total_counters_needed > total_counters + 1:  # +1 slack
            self._log(f"    Bot {bot_id}: Can't work independently - need {total_counters_needed} counters, have {total_counters}")
            return False
        
        return True
    
    def _should_force_collaboration(self, controller: RobotController, team: Team) -> bool:
        """Determine if bots should collaborate on a single order instead of independent orders.
        
        Force collaboration when:
        - Only 1 cooker and multiple cooking orders
        - Very limited counter space
        """
        # If only 1 cooker, both bots working on cooking orders is wasteful
        if len(self.cookers) == 1:
            return True
        
        # If very limited counters (0-1 free), collaboration might help
        free_counters, free_cookers = self._get_available_resources(controller, team)
        
        if free_counters <= 1 and len(self.counters) <= 2:
            return True
        
        return False
    
    def _cleanup_clutter(self, controller: RobotController, bot_id: int, team: Team) -> bool:
        """Clean up burnt items on cookers or unused items on counters.
        
        Returns True if we performed a cleanup action (moved/trashed item).
        """
        bot = controller.get_bot_state(bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        
        # If holding something, trash it first
        if holding:
            trash = self._get_nearest(bot_pos, self.trashes)
            if trash:
                if self._move_toward(controller, bot_id, trash, team):
                    controller.trash(bot_id, trash[0], trash[1])
                    self._log(f"  Bot {bot_id}: Cleaned up held item")
                return True
            return False
        
        # Check for burnt items on cookers - priority cleanup
        for cooker_pos in self.cookers:
            tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
            if tile:
                pan = getattr(tile, 'pan', None)
                if pan:
                    food = getattr(pan, 'food', None)
                    if food:
                        cooked_stage = getattr(food, 'cooked_stage', 0)
                        if cooked_stage >= 2:  # Burnt
                            self._log(f"  Bot {bot_id}: Cleaning up BURNT {food.food_name} at cooker {cooker_pos}")
                            # Take from pan
                            if self._move_toward(controller, bot_id, cooker_pos, team):
                                if controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1]):
                                    return True
                            return True
        
        # Check for items on counters that are not being used by any active goal
        active_plate_counters = set()
        for bid, goal in self.bot_goals.items():
            if goal:
                pc = goal.get('plate_counter')
                if pc:
                    active_plate_counters.add(pc)
                wc = goal.get('work_counter')
                if wc:
                    active_plate_counters.add(wc)
        
        # Check counters for items not part of current work
        for counter_pos in self.counters:
            if counter_pos in active_plate_counters:
                continue  # In use by a goal
            if counter_pos in self.reserved_counters:
                continue  # Reserved
            
            tile = controller.get_tile(team, counter_pos[0], counter_pos[1])
            if tile:
                item = getattr(tile, 'item', None)
                if item:
                    # Check if this item is useful for any order we could work on
                    # For now, just clean up any item that's been sitting around
                    item_type = getattr(item, 'type', None) or type(item).__name__
                    self._log(f"  Bot {bot_id}: Cleaning up clutter ({item_type}) at counter {counter_pos}")
                    if self._move_toward(controller, bot_id, counter_pos, team):
                        if controller.pickup(bot_id, counter_pos[0], counter_pos[1]):
                            return True
                    return True
        
        return False
    
    def _do_preparation(self, controller: RobotController, bot_id: int, team: Team) -> bool:
        """Prepare for upcoming orders when no current orders are available.
        
        Returns True if doing something useful, False otherwise.
        """
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        bot = controller.get_bot_state(bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        # Find upcoming orders (within next 30 turns)
        upcoming_orders = []
        for order in orders:
            order_id = order.get('order_id')
            starts_at = order.get('created_turn', 0)
            expires = order.get('expires_turn', 0)
            
            # Skip completed/expired
            if order_id in self.completed_order_ids:
                continue
            if expires <= current_turn:
                continue
            # Skip orders in progress by other bot
            working_bot = self.orders_in_progress.get(order_id)
            if working_bot is not None and working_bot != bot_id:
                continue
            
            # Include orders starting within 30 turns
            if starts_at <= current_turn + 30:
                upcoming_orders.append(order)
        
        if not upcoming_orders:
            return False
        
        self._log(f"  Bot {bot_id}: Preparing for {len(upcoming_orders)} upcoming orders")
        
        # Analyze what ingredients will be needed
        ingredient_counts = {}
        for order in upcoming_orders:
            for ing in order.get('required', []):
                ingredient_counts[ing] = ingredient_counts.get(ing, 0) + 1
        
        # Priority 1: If holding nothing, get a plate ready
        if holding is None:
            # Check if any sink table has plates
            for sink_table in self.sink_tables:
                tile = controller.get_tile(team, sink_table[0], sink_table[1])
                if tile and hasattr(tile, 'count') and tile.count > 0:
                    if self._move_toward(controller, bot_id, sink_table, team):
                        if controller.take_clean_plate(bot_id, sink_table[0], sink_table[1]):
                            self._log(f"  Bot {bot_id}: Prepared - got plate from sink")
                            return True
                    return True
            
            # Buy plate from shop if affordable
            if self.shops and money >= ShopCosts.PLATE.buy_cost:
                shop = self._get_nearest(bot_pos, self.shops)
                if shop:
                    if self._move_toward(controller, bot_id, shop, team):
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
                        self._log(f"  Bot {bot_id}: Prepared - bought plate")
                    return True
        
        # Priority 2: If holding plate, place it on a counter
        if holding and holding.get('type') == 'Plate':
            counter = self._get_free_counter(controller, team, bot_id, bot_pos)
            if counter:
                if self._move_toward(controller, bot_id, counter, team):
                    if controller.place(bot_id, counter[0], counter[1]):
                        self._log(f"  Bot {bot_id}: Prepared - placed plate at {counter}")
                return True
        
        # Priority 3: Buy a common ingredient (NOODLES is cheap and common)
        if holding is None:
            # Find most common simple ingredient (no chop/cook needed)
            best_ing = None
            best_count = 0
            for ing, count in ingredient_counts.items():
                info = INGREDIENT_INFO.get(ing, {})
                # Prefer simple ingredients (no chop, no cook)
                if not info.get('chop', False) and not info.get('cook', False):
                    if count > best_count:
                        best_count = count
                        best_ing = ing
            
            if best_ing and self.shops:
                ing_info = INGREDIENT_INFO.get(best_ing, {})
                cost = ing_info.get('cost', 100)
                if money >= cost:
                    shop = self._get_nearest(bot_pos, self.shops)
                    if shop:
                        if self._move_toward(controller, bot_id, shop, team):
                            # Convert ingredient name to FoodType
                            food_type = None
                            for ft in [FoodType.NOODLES, FoodType.SAUCE, FoodType.EGG, FoodType.ONIONS, FoodType.MEAT]:
                                if ft.food_name == best_ing:
                                    food_type = ft
                                    break
                            if food_type and controller.buy(bot_id, food_type, shop[0], shop[1]):
                                self._log(f"  Bot {bot_id}: Prepared - bought {best_ing}")
                        return True
        
        # Priority 4: If holding a simple ingredient, store it on a counter/box
        if holding and holding.get('type') == 'Food':
            food_name = holding.get('food_name')
            info = INGREDIENT_INFO.get(food_name, {})
            if not info.get('chop', False) and not info.get('cook', False):
                # Store on a free counter
                counter = self._get_free_counter(controller, team, bot_id, bot_pos)
                if counter:
                    if self._move_toward(controller, bot_id, counter, team):
                        if controller.place(bot_id, counter[0], counter[1]):
                            self._log(f"  Bot {bot_id}: Prepared - stored {food_name} at {counter}")
                    return True
        
        # Priority 5: Position near shop for quick access
        if self.shops:
            shop = self._get_nearest(bot_pos, self.shops)
            if shop and self._chebyshev_dist(bot_pos, shop) > 2:
                self._move_toward(controller, bot_id, shop, team)
                self._log(f"  Bot {bot_id}: Prepared - moving toward shop")
                return True
        
        return False
    
    def _help_other_bot(self, controller: RobotController, helper_id: int, 
                        primary_id: int, team: Team) -> bool:
        """Helper bot assists primary bot with its order.
        
        Returns True if helper is doing something useful, False otherwise.
        """
        primary_goal = self.bot_goals.get(primary_id)
        if not primary_goal:
            return False
        
        order = primary_goal.get('order', {})
        required = order.get('required', [])
        primary_state = primary_goal.get('state', 'INIT')
        
        helper = controller.get_bot_state(helper_id)
        helper_pos = (helper['x'], helper['y'])
        helper_holding = helper.get('holding')
        money = controller.get_team_money(team)
        
        # Task 1: Get plate if primary doesn't have one yet
        plate_counter = primary_goal.get('plate_counter')
        if plate_counter is None and primary_state not in ['ENSURE_PLATE', 'PLACE_PLATE']:
            if helper_holding and helper_holding.get('type') == 'Plate':
                # Place the plate
                counter = self._get_free_counter(controller, team, helper_id, helper_pos)
                if counter:
                    if self._move_toward(controller, helper_id, counter, team):
                        if controller.place(helper_id, counter[0], counter[1]):
                            primary_goal['plate_counter'] = counter
                            self._reserve_counter(counter, primary_id)
                            self._log(f"  Bot {helper_id}: Placed plate at {counter} for Bot {primary_id}")
                            return True
                return True  # Still moving
            else:
                # Get a plate
                for sink_table in self.sink_tables:
                    tile = controller.get_tile(team, sink_table[0], sink_table[1])
                    if tile and hasattr(tile, 'count') and tile.count > 0:
                        if self._move_toward(controller, helper_id, sink_table, team):
                            controller.take_clean_plate(helper_id, sink_table[0], sink_table[1])
                            self._log(f"  Bot {helper_id}: Getting plate for Bot {primary_id}")
                            return True
                        return True
                # Try buying from shop
                if self.shops and money >= ShopCosts.PLATE.buy_cost:
                    shop = self._get_nearest(helper_pos, self.shops)
                    if shop:
                        if self._move_toward(controller, helper_id, shop, team):
                            controller.buy(helper_id, ShopCosts.PLATE, shop[0], shop[1])
                            self._log(f"  Bot {helper_id}: Buying plate for Bot {primary_id}")
                            return True
                        return True
        
        # Task 2: Buy/fetch simple ingredients if needed and not already being processed
        processed_ingredients = primary_goal.get('processed_ingredients', set())
        current_ingredient = primary_goal.get('current_ingredient')
        
        for ing in required:
            if ing in processed_ingredients:
                continue
            if ing == current_ingredient:
                continue  # Primary is handling this one
            
            info = INGREDIENT_INFO.get(ing, {})
            
            # Only help with simple ingredients (no chop/cook) to avoid conflicts
            if not info.get('chop', False) and not info.get('cook', False):
                cost = info.get('cost', 0)
                if money >= cost and self.shops:
                    # Check if helper already has this ingredient
                    if helper_holding and helper_holding.get('type') == 'Food':
                        if helper_holding.get('food_name') == ing:
                            # Deliver to plate
                            if plate_counter:
                                if self._move_toward(controller, helper_id, plate_counter, team):
                                    if controller.add_food_to_plate(helper_id, plate_counter[0], plate_counter[1]):
                                        self._log(f"  Bot {helper_id}: Added {ing} to plate for Bot {primary_id}")
                                        if 'processed_ingredients' not in primary_goal:
                                            primary_goal['processed_ingredients'] = set()
                                        primary_goal['processed_ingredients'].add(ing)
                                        return True
                                return True
                    elif not helper_holding:
                        # Buy the ingredient
                        shop = self._get_nearest(helper_pos, self.shops)
                        if shop:
                            food_type = getattr(FoodType, ing, None)
                            if food_type:
                                if self._move_toward(controller, helper_id, shop, team):
                                    controller.buy(helper_id, food_type, shop[0], shop[1])
                                    self._log(f"  Bot {helper_id}: Buying {ing} for Bot {primary_id}")
                                    return True
                                return True
        
        # Task 3: Position near submit if order is almost ready
        if primary_state in ['ASSEMBLE', 'SUBMIT']:
            # Wait near submit but don't block
            submit = self._get_nearest(helper_pos, self.submits)
            if submit:
                # Stay 2 tiles away from submit
                primary_bot = controller.get_bot_state(primary_id)
                primary_pos = (primary_bot['x'], primary_bot['y'])
                
                dist_to_submit = self._chebyshev_dist(helper_pos, submit)
                if dist_to_submit > 3:
                    self._move_toward(controller, helper_id, submit, team)
                    self._log(f"  Bot {helper_id}: Positioning near submit")
                    return True
        
        # Nothing specific to help with - stay out of the way
        return False
    
    def _score_bot_for_order(self, controller: RobotController, bot_id: int, 
                              order: Dict, team: Team) -> int:
        """Score how suitable a bot is for an order (lower = better)."""
        bot = controller.get_bot_state(bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        required = order.get('required', [])
        
        score = 0
        
        # Factor 1: Distance to shop
        if self.shops:
            score += self._chebyshev_dist(bot_pos, self._get_nearest(bot_pos, self.shops)) * 2
        
        # Factor 2: Already holding something useful for this order?
        if holding and holding.get('type') == 'Food':
            food_name = holding.get('food_name')
            if food_name in required:
                score -= 30  # Big bonus: already has needed ingredient
            else:
                score += 15  # Penalty: holding wrong thing
        elif holding and holding.get('type') == 'Plate':
            score -= 20  # Bonus: already has plate
        elif holding:
            score += 10  # Penalty: holding something unneeded
        
        # Factor 3: Distance to submit station
        if self.submits:
            score += self._chebyshev_dist(bot_pos, self._get_nearest(bot_pos, self.submits))
        
        # Factor 4: Current progress on this order (if any)
        current_goal = self.bot_goals.get(bot_id)
        if current_goal and current_goal.get('order', {}).get('order_id') == order.get('order_id'):
            score -= 50  # Big bonus: already working on it
        
        # Factor 5: Order difficulty
        score += self._get_order_difficulty(order)
        
        return score
    
    def _order_needs_cooking(self, order: Dict) -> bool:
        """Check if an order requires cooking."""
        for ing in order.get('required', []):
            info = INGREDIENT_INFO.get(ing, {})
            if info.get('cook', False):
                return True
        return False
    
    def _order_needs_chopping(self, order: Dict) -> bool:
        """Check if an order requires chopping."""
        for ing in order.get('required', []):
            info = INGREDIENT_INFO.get(ing, {})
            if info.get('chop', False):
                return True
        return False
    
    def _count_available_cookers(self, controller: RobotController, bot_id: int, team: Team) -> int:
        """Count cookers available to this bot (not reserved by others)."""
        count = 0
        for c in self.cookers:
            reserved_by = self.reserved_cookers.get(c)
            if reserved_by is None or reserved_by == bot_id:
                count += 1
        return count
    
    def _count_available_counters(self, controller: RobotController, bot_id: int, team: Team) -> int:
        """Count counters available to this bot (not reserved by others and empty)."""
        count = 0
        for c in self.counters:
            reserved_by = self.reserved_counters.get(c)
            if reserved_by is not None and reserved_by != bot_id:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                count += 1
        return count
    
    def _select_order_for_bot(self, controller: RobotController, bot_id: int, team: Team) -> Optional[Dict]:
        """Select the best order for a specific bot.
        
        Includes both active orders and future orders we can prepare for.
        Future orders are selected if no active orders are feasible.
        """
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        # Check resource availability
        available_cookers = self._count_available_cookers(controller, bot_id, team)
        available_counters = self._count_available_counters(controller, bot_id, team)
        
        active_candidates = []
        future_candidates = []
        
        for order in orders:
            order_id = order.get('order_id')
            expires = order.get('expires_turn', 0)
            starts_at = order.get('created_turn', 0)
            reward = order.get('reward', 0)
            
            # Skip completed orders
            if order_id in self.completed_order_ids:
                continue
            
            # Skip expired orders
            if expires <= current_turn:
                continue
            
            # Skip orders already being worked on by another bot
            working_bot = self.orders_in_progress.get(order_id)
            if working_bot is not None and working_bot != bot_id:
                continue
            
            # Skip orders too far in future (more than 60 turns)
            if starts_at > current_turn + 60:
                continue
            
            is_active = starts_at <= current_turn
            
            # For active orders, check feasibility strictly
            if is_active:
                if not self._can_complete_order(controller, bot_id, order, team):
                    self._log(f"    Bot {bot_id}: Skipping order #{order_id} - cannot complete in time")
                    continue
            
            # SPACE CHECK - skip orders that need more resources than available
            if not self._has_enough_space(controller, order, team):
                self._log(f"    Bot {bot_id}: Skipping order #{order_id} - not enough space/resources")
                continue
            
            # COMBINED RESOURCE CHECK - can both bots work independently?
            if not self._can_both_bots_work_independently(controller, bot_id, order, team):
                self._log(f"    Bot {bot_id}: Skipping order #{order_id} - would conflict with other bot's order")
                continue
            
            # Check if bot can reach submit station (for isolated maps)
            if not self.same_zone:
                submit = self._get_nearest(self._get_bot_pos(controller, bot_id), self.submits)
                if submit and not self._can_bot_reach(controller, bot_id, submit, team):
                    continue
            
            # Check resource constraints
            needs_cooking = self._order_needs_cooking(order)
            needs_chopping = self._order_needs_chopping(order)
            
            # If no cookers available and order needs cooking, skip
            if needs_cooking and available_cookers == 0:
                self._log(f"    Bot {bot_id}: Skipping order #{order_id} - no cooker available")
                continue
            
            # If no counters available and order needs chopping, penalty
            resource_penalty = 0
            if needs_chopping and available_counters == 0:
                resource_penalty = 500
            
            # Calculate scores
            profitability = self._get_order_profitability(order)
            profit_score = -profitability
            bot_score = self._score_bot_for_order(controller, bot_id, order, team)
            bot_score += resource_penalty
            
            if is_active:
                # For active orders: prioritize by urgency then profitability
                time_left = expires - current_turn
                score = (time_left, profit_score, bot_score)
                active_candidates.append((score, order))
            else:
                # For future orders: prioritize by start time (soonest first) then easiness
                turns_until_start = starts_at - current_turn
                difficulty = self._get_order_difficulty(order)
                score = (turns_until_start, difficulty, profit_score)
                future_candidates.append((score, order))
        
        # Prefer active orders first
        if active_candidates:
            active_candidates.sort(key=lambda x: x[0])
            selected = active_candidates[0][1]
            self._log(f"    Bot {bot_id}: Order selection from {len(active_candidates)} active candidates")
            return selected
        
        # If no active orders, select a future order to prepare for
        if future_candidates:
            future_candidates.sort(key=lambda x: x[0])
            selected = future_candidates[0][1]
            self._log(f"    Bot {bot_id}: Preparing for future order #{selected.get('order_id')} (starts turn {selected.get('created_turn')})")
            return selected
        
        return None
    
    # ========== GOAL CREATION ==========
    
    def _create_goal(self, order: Dict, bot_id: int) -> Dict[str, Any]:
        """Create a new goal state for an order."""
        self.goal_counter += 1
        
        # Categorize ingredients
        chop_cook = []   # MEAT: chop then cook
        chop_only = []   # ONIONS: chop, store on counter
        cook_only = []   # EGG: cook
        simple = []      # NOODLES, SAUCE: just add to plate
        
        for ing in order['required']:
            info = INGREDIENT_INFO.get(ing, {})
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
        
        # Register order as in progress
        self.orders_in_progress[order['order_id']] = bot_id
        
        return {
            'goal_id': self.goal_counter,
            'goal_type': 'deliver_order',
            'order': order,
            'state': 'INIT',
            'plate_counter': None,
            'work_counter': None,
            'handoff_drop_tile': None,
            # Ingredient queues
            'chop_cook_queue': chop_cook,
            'chop_only_queue': chop_only,
            'cook_only_queue': cook_only,
            'simple_queue': simple,
            # Tracking
            'stored_items': {},
            'cooking_items': {},
            'plate_contents': [],
            'current_ingredient': None,
        }
    
    def _clear_goal(self, bot_id: int) -> None:
        """Clear a bot's goal and release resources."""
        goal = self.bot_goals.get(bot_id)
        if goal:
            order = goal.get('order', {})
            order_id = order.get('order_id')
            if order_id in self.orders_in_progress:
                del self.orders_in_progress[order_id]
        
        self._release_reservations(bot_id)
        self.bot_goals[bot_id] = None
        
        # Clear cooking wait if any
        if bot_id in self.cooking_wait:
            del self.cooking_wait[bot_id]
    
    # ========== PARALLEL COOKING ==========
    
    def _check_cooking_done(self, controller: RobotController, bot_id: int, team: Team) -> bool:
        """Check if a bot's cooking is done. Returns True if cooking completed."""
        if bot_id not in self.cooking_wait:
            return False
        
        cw = self.cooking_wait[bot_id]
        cooker_pos = cw.get('cooker')
        ingredient = cw.get('ingredient')
        
        if cooker_pos is None:
            del self.cooking_wait[bot_id]
            return False
        
        tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
        if tile is None:
            del self.cooking_wait[bot_id]
            return False
        
        pan = getattr(tile, 'item', None)
        if not isinstance(pan, Pan) or pan.food is None:
            del self.cooking_wait[bot_id]
            return False
        
        # Check if cooked
        if pan.food.cooked_stage >= 1:
            self._log(f"  Bot {bot_id}: Cooking done! {ingredient} at {cooker_pos}")
            del self.cooking_wait[bot_id]
            return True
        
        return False
    
    def _estimate_task_time(self, task_type: str, bot_pos: Tuple[int, int]) -> int:
        """Estimate turns needed for a parallel task."""
        AVG_MOVE = 5
        
        if task_type == 'get_plate':
            if self.sink_tables:
                return self._chebyshev_dist(bot_pos, self.sink_tables[0]) + 1
            elif self.shops:
                return self._chebyshev_dist(bot_pos, self.shops[0]) + 1
            return AVG_MOVE + 1
        elif task_type == 'buy_ingredient':
            if self.shops:
                return self._chebyshev_dist(bot_pos, self.shops[0]) + 1
            return AVG_MOVE + 1
        
        return AVG_MOVE
    
    def _can_do_parallel_task(self, controller: RobotController, bot_id: int, team: Team) -> Optional[str]:
        """Check if bot can do something useful while cooking. Returns task type or None."""
        if bot_id not in self.cooking_wait:
            return None
        
        cw = self.cooking_wait[bot_id]
        cooker_pos = cw.get('cooker')
        current_turn = controller.get_turn()
        done_turn = cw.get('done_turn', current_turn)
        
        turns_left = done_turn - current_turn
        if turns_left <= 3:
            # Not enough time for parallel task
            return None
        
        bot_pos = self._get_bot_pos(controller, bot_id)
        goal = self.bot_goals.get(bot_id)
        
        if goal is None:
            return None
        
        # Check if we need a plate
        if goal.get('plate_counter') is None:
            task_time = self._estimate_task_time('get_plate', bot_pos)
            return_time = self._chebyshev_dist(bot_pos, cooker_pos) if cooker_pos else 5
            if task_time + return_time < turns_left:
                return 'get_plate'
        
        # Check if we need to buy another ingredient
        chop_cook = goal.get('chop_cook_queue', [])
        chop_only = goal.get('chop_only_queue', [])
        cook_only = goal.get('cook_only_queue', [])
        simple = goal.get('simple_queue', [])
        
        remaining = chop_cook + chop_only + cook_only + simple
        if remaining:
            task_time = self._estimate_task_time('buy_ingredient', bot_pos)
            return_time = self._chebyshev_dist(bot_pos, cooker_pos) if cooker_pos else 5
            if task_time + return_time < turns_left:
                return 'buy_ingredient'
        
        return None
    
    # ========== HANDOFF FOR ISOLATED MAPS ==========
    
    def _needs_handoff(self, controller: RobotController, bot_id: int, 
                       target: Tuple[int, int], team: Team) -> bool:
        """Check if handoff is needed to reach target."""
        if self.same_zone:
            return False
        
        return not self._can_bot_reach(controller, bot_id, target, team)
    
    def _find_handoff_counter(self, controller: RobotController, team: Team,
                               from_bot: int, holding: Dict) -> Optional[Tuple[int, int]]:
        """Find a counter that bridges zones for handoff."""
        if not self.handoff_counters:
            return None
        
        from_pos = self._get_bot_pos(controller, from_bot)
        
        # Find nearest handoff counter that's empty
        for counter in sorted(self.handoff_counters, 
                             key=lambda c: self._chebyshev_dist(from_pos, c)):
            tile = controller.get_tile(team, counter[0], counter[1])
            if tile and getattr(tile, 'item', None) is None:
                return counter
        
        return None
    
    # ========== GOAL EXECUTION (State Machine) ==========
    
    def _execute_goal(self, controller: RobotController, bot_id: int, team: Team) -> bool:
        """Execute a bot's goal pipeline. Returns True if blocked."""
        goal = self.bot_goals.get(bot_id)
        if goal is None:
            return False
        
        bot = controller.get_bot_state(bot_id)
        bot_pos = (bot['x'], bot['y'])
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        order = goal.get('order')
        state = goal.get('state', 'INIT')
        
        if order is None:
            return False
        
        # Get nearest resources
        shop = self._get_nearest(bot_pos, self.shops)
        submit = self._get_nearest(bot_pos, self.submits)
        
        # Log execution context
        holding_str = "nothing"
        if holding:
            if holding.get('type') == 'Food':
                holding_str = holding.get('food_name', 'food')
            else:
                holding_str = holding.get('type', 'item')
        self._log(f"  Bot {bot_id}: state={state}, pos={bot_pos}, holding={holding_str}")
        
        # === STATE: INIT ===
        if state == 'INIT':
            chop_cook = goal.get('chop_cook_queue', [])
            chop_only = goal.get('chop_only_queue', [])
            cook_only = goal.get('cook_only_queue', [])
            stored_items = goal.get('stored_items', {})
            
            # First priority: chop+cook items (MEAT)
            if chop_cook:
                ing = chop_cook[0]
                if ing in stored_items:
                    self._log(f"    {ing} already stored, picking up")
                    goal['current_ingredient'] = ing
                    goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    goal['current_ingredient'] = ing
                    goal['state'] = 'BUY_INGREDIENT'
                    goal['next_after_buy'] = 'CHOP_THEN_COOK'
                return False
            
            # Second: chop-only items (ONIONS)
            if chop_only:
                ing = chop_only[0]
                if ing in stored_items:
                    goal['current_ingredient'] = ing
                    goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    goal['current_ingredient'] = ing
                    goal['state'] = 'BUY_INGREDIENT'
                    goal['next_after_buy'] = 'CHOP_THEN_STORE'
                return False
            
            # Third: cook-only items (EGG)
            if cook_only:
                ing = cook_only[0]
                if ing in stored_items:
                    goal['current_ingredient'] = ing
                    goal['state'] = 'PICKUP_STORED_FOR_COOK'
                else:
                    goal['current_ingredient'] = ing
                    goal['state'] = 'BUY_INGREDIENT'
                    goal['next_after_buy'] = 'COOK_ONLY'
                return False
            
            # All pre-processing done - now handle plate and assembly
            goal['state'] = 'ENSURE_PLATE'
            return False
        
        # === STATE: BUY_INGREDIENT ===
        if state == 'BUY_INGREDIENT':
            ing = goal.get('current_ingredient')
            next_state = goal.get('next_after_buy', 'INIT')
            
            if ing is None:
                goal['state'] = 'INIT'
                return False
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                goal['state'] = next_state
                return False
            
            if holding:
                # Wrong item - trash it
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._is_blocked(controller, bot_id, trash, team):
                            return True
                        if self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                return False
            
            if shop is None:
                return True
            if self._is_blocked(controller, bot_id, shop, team):
                return True
            if self._move_toward(controller, bot_id, shop, team):
                food_type = getattr(FoodType, ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
            return False
        
        # === STATE: CHOP_THEN_COOK ===
        if state == 'CHOP_THEN_COOK':
            ing = goal.get('current_ingredient')
            work_counter = goal.get('work_counter')
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                if holding.get('chopped'):
                    goal['state'] = 'START_COOK'
                    return False
                
                # Need to place for chopping
                exclude = set(goal.get('stored_items', {}).values())
                if goal.get('plate_counter'):
                    exclude.add(goal['plate_counter'])
                
                counter = self._get_free_counter(controller, team, bot_id, bot_pos, exclude)
                if counter is None:
                    return True
                
                if self._is_blocked(controller, bot_id, counter, team):
                    return True
                if self._move_toward(controller, bot_id, counter, team):
                    if controller.place(bot_id, counter[0], counter[1]):
                        goal['work_counter'] = counter
                        self._reserve_counter(counter, bot_id)
                return False
            
            if work_counter:
                tile = controller.get_tile(team, work_counter[0], work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    food = tile.item
                    if self._is_blocked(controller, bot_id, work_counter, team):
                        return True
                    if self._move_toward(controller, bot_id, work_counter, team):
                        if food.chopped:
                            controller.pickup(bot_id, work_counter[0], work_counter[1])
                            goal['work_counter'] = None
                        else:
                            controller.chop(bot_id, work_counter[0], work_counter[1])
                    return False
                else:
                    goal['work_counter'] = None
            
            goal['state'] = 'BUY_INGREDIENT'
            goal['next_after_buy'] = 'CHOP_THEN_COOK'
            return False
        
        # === STATE: PICKUP_STORED_FOR_COOK ===
        if state == 'PICKUP_STORED_FOR_COOK':
            ing = goal.get('current_ingredient')
            stored_items = goal.get('stored_items', {})
            
            if ing is None or ing not in stored_items:
                goal['current_ingredient'] = None
                goal['state'] = 'INIT'
                return False
            
            stored_pos = stored_items[ing]
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                del stored_items[ing]
                goal['stored_items'] = stored_items
                goal['state'] = 'START_COOK'
                return False
            
            tile = controller.get_tile(team, stored_pos[0], stored_pos[1])
            if getattr(tile, 'item', None) is None:
                del stored_items[ing]
                goal['stored_items'] = stored_items
                goal['current_ingredient'] = None
                goal['state'] = 'INIT'
                return False
            
            if self._is_blocked(controller, bot_id, stored_pos, team):
                return True
            if self._move_toward(controller, bot_id, stored_pos, team):
                if controller.pickup(bot_id, stored_pos[0], stored_pos[1]):
                    del stored_items[ing]
                    goal['stored_items'] = stored_items
                    goal['state'] = 'START_COOK'
            return False
        
        # === STATE: START_COOK ===
        if state == 'START_COOK':
            ing = goal.get('current_ingredient')
            
            if not holding or holding.get('type') != 'Food':
                goal['state'] = 'INIT'
                return False
            
            cooker = self._get_available_cooker(controller, team, bot_id, bot_pos)
            if cooker is None:
                # No available cooker - check if we can take cooked food
                for cooker_pos in self.cookers:
                    reserved_by = self.reserved_cookers.get(cooker_pos)
                    if reserved_by is not None and reserved_by != bot_id:
                        continue
                    tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food is not None and pan.food.cooked_stage >= 1:
                            self._log(f"    Taking cooked food from {cooker_pos}")
                            if self._move_toward(controller, bot_id, cooker_pos, team):
                                controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1])
                            return False
                
                # Store ingredient and wait
                counter = self._get_free_counter(controller, team, bot_id, bot_pos)
                if counter:
                    if self._move_toward(controller, bot_id, counter, team):
                        if controller.place(bot_id, counter[0], counter[1]):
                            stored = goal.get('stored_items', {})
                            stored[ing] = counter
                            goal['stored_items'] = stored
                            goal['current_ingredient'] = None
                            goal['state'] = 'INIT'
                    return False
                
                # No counter - just wait
                return False
            
            if self._is_blocked(controller, bot_id, cooker, team):
                return True
            if self._move_toward(controller, bot_id, cooker, team):
                if controller.place(bot_id, cooker[0], cooker[1]):
                    self._log(f"    Started cooking {ing} at {cooker}")
                    self._reserve_cooker(cooker, bot_id)
                    
                    # Track cooking for parallel tasks
                    current_turn = controller.get_turn()
                    self.cooking_wait[bot_id] = {
                        'cooker': cooker,
                        'ingredient': ing,
                        'done_turn': current_turn + 10  # Cooking takes 10 turns
                    }
                    
                    cooking_items = goal.get('cooking_items', {})
                    cooking_items[ing] = cooker
                    goal['cooking_items'] = cooking_items
                    
                    # Remove from queue
                    for queue_name in ['chop_cook_queue', 'cook_only_queue']:
                        queue = goal.get(queue_name, [])
                        if ing in queue:
                            queue.remove(ing)
                    
                    goal['current_ingredient'] = None
                    goal['state'] = 'INIT'
            return False
        
        # === STATE: COOK_ONLY ===
        if state == 'COOK_ONLY':
            goal['state'] = 'START_COOK'
            return False
        
        # === STATE: CHOP_THEN_STORE ===
        if state == 'CHOP_THEN_STORE':
            ing = goal.get('current_ingredient')
            work_counter = goal.get('work_counter')
            
            if holding and holding.get('type') == 'Food' and holding.get('food_name') == ing:
                if holding.get('chopped'):
                    # Store it
                    counter = self._get_free_counter(controller, team, bot_id, bot_pos)
                    if counter is None:
                        return True
                    if self._is_blocked(controller, bot_id, counter, team):
                        return True
                    if self._move_toward(controller, bot_id, counter, team):
                        if controller.place(bot_id, counter[0], counter[1]):
                            stored = goal.get('stored_items', {})
                            stored[ing] = counter
                            goal['stored_items'] = stored
                            self._reserve_counter(counter, bot_id)
                            
                            chop_only = goal.get('chop_only_queue', [])
                            if ing in chop_only:
                                chop_only.remove(ing)
                            
                            goal['current_ingredient'] = None
                            goal['state'] = 'INIT'
                    return False
                
                # Need to place for chopping
                exclude = set(goal.get('stored_items', {}).values())
                counter = self._get_free_counter(controller, team, bot_id, bot_pos, exclude)
                if counter is None:
                    return True
                if self._is_blocked(controller, bot_id, counter, team):
                    return True
                if self._move_toward(controller, bot_id, counter, team):
                    if controller.place(bot_id, counter[0], counter[1]):
                        goal['work_counter'] = counter
                return False
            
            if work_counter:
                tile = controller.get_tile(team, work_counter[0], work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    food = tile.item
                    if self._is_blocked(controller, bot_id, work_counter, team):
                        return True
                    if self._move_toward(controller, bot_id, work_counter, team):
                        if food.chopped:
                            controller.pickup(bot_id, work_counter[0], work_counter[1])
                            goal['work_counter'] = None
                        else:
                            controller.chop(bot_id, work_counter[0], work_counter[1])
                    return False
                else:
                    goal['work_counter'] = None
            
            goal['state'] = 'BUY_INGREDIENT'
            goal['next_after_buy'] = 'CHOP_THEN_STORE'
            return False
        
        # === STATE: ENSURE_PLATE ===
        if state == 'ENSURE_PLATE':
            required = order.get('required', [])
            
            # If holding unwanted item, trash it
            if holding is not None:
                holding_type = holding.get('type')
                if holding_type == 'Plate':
                    goal['state'] = 'PLACE_PLATE'
                    return False
                elif holding_type == 'Food':
                    food_name = holding.get('food_name')
                    if food_name not in required:
                        if self.trashes:
                            trash = self._get_nearest(bot_pos, self.trashes)
                            if trash:
                                if self._move_toward(controller, bot_id, trash, team):
                                    controller.trash(bot_id, trash[0], trash[1])
                        return False
                    else:
                        # Store needed ingredient
                        counter = self._get_free_counter(controller, team, bot_id, bot_pos)
                        if counter:
                            if self._move_toward(controller, bot_id, counter, team):
                                if controller.place(bot_id, counter[0], counter[1]):
                                    stored = goal.get('stored_items', {})
                                    stored[food_name] = counter
                                    goal['stored_items'] = stored
                        return False
                else:
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash:
                            if self._move_toward(controller, bot_id, trash, team):
                                controller.trash(bot_id, trash[0], trash[1])
                    return False
            
            # Try sink table first - but check if it has plates
            sink_attempts = goal.get('sink_attempts', 0)
            if self.sink_tables and sink_attempts < 1:  # Only try once
                # Find a sink table that might have plates
                best_sink = None
                for sink_table in self.sink_tables:
                    tile = controller.get_tile(team, sink_table[0], sink_table[1])
                    # Check if sink table has plates (count > 0)
                    if tile and hasattr(tile, 'count') and tile.count > 0:
                        if best_sink is None or self._chebyshev_dist(bot_pos, sink_table) < self._chebyshev_dist(bot_pos, best_sink):
                            best_sink = sink_table
                
                if best_sink:
                    if self._is_blocked(controller, bot_id, best_sink, team):
                        return True
                    if self._move_toward(controller, bot_id, best_sink, team):
                        result = controller.take_clean_plate(bot_id, best_sink[0], best_sink[1])
                        goal['sink_attempts'] = sink_attempts + 1
                        if not result:
                            self._log(f"    take_clean_plate failed, attempt {sink_attempts + 1}")
                        # Whether success or fail, we tried - move on
                    else:
                        return False
                else:
                    # No sink table with plates - skip to shop
                    goal['sink_attempts'] = 3  # Mark as exhausted
            
            # Buy plate from shop
            if shop:
                if self._is_blocked(controller, bot_id, shop, team):
                    return True
                if self._move_toward(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
            return False
        
        # === STATE: PLACE_PLATE ===
        if state == 'PLACE_PLATE':
            if not holding or holding.get('type') != 'Plate':
                goal['state'] = 'ENSURE_PLATE'
                return False
            
            counter = self._get_free_counter(controller, team, bot_id, bot_pos)
            if counter is None:
                # No free counter - trash plate
                if self.trashes:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                            goal['state'] = 'ENSURE_PLATE'
                return False
            
            if self._is_blocked(controller, bot_id, counter, team):
                return True
            if self._move_toward(controller, bot_id, counter, team):
                if controller.place(bot_id, counter[0], counter[1]):
                    goal['plate_counter'] = counter
                    self._reserve_counter(counter, bot_id)
                    goal['state'] = 'ASSEMBLE'
            return False
        
        # === STATE: ASSEMBLE ===
        if state == 'ASSEMBLE':
            plate_counter = goal.get('plate_counter')
            cooking_items = goal.get('cooking_items', {})
            stored_items = goal.get('stored_items', {})
            plate_contents = goal.get('plate_contents', [])
            simple_queue = goal.get('simple_queue', [])
            required = order.get('required', [])
            
            # Check what's still needed
            needed = [ing for ing in required if ing not in plate_contents]
            
            if not needed:
                goal['state'] = 'SUBMIT'
                return False
            
            # If holding food, add it to plate
            if holding and holding.get('type') == 'Food':
                food_name = holding.get('food_name')
                if food_name in needed:
                    if plate_counter:
                        if self._is_blocked(controller, bot_id, plate_counter, team):
                            return True
                        if self._move_toward(controller, bot_id, plate_counter, team):
                            if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                                plate_contents.append(food_name)
                                goal['plate_contents'] = plate_contents
                        return False
                else:
                    # Wrong food - trash it
                    if self.trashes:
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash:
                            if self._move_toward(controller, bot_id, trash, team):
                                controller.trash(bot_id, trash[0], trash[1])
                    return False
            
            # If holding plate, put it down first
            if holding and holding.get('type') == 'Plate':
                if plate_counter is None:
                    counter = self._get_free_counter(controller, team, bot_id, bot_pos)
                    if counter:
                        if self._move_toward(controller, bot_id, counter, team):
                            if controller.place(bot_id, counter[0], counter[1]):
                                goal['plate_counter'] = counter
                                self._reserve_counter(counter, bot_id)
                    return False
                else:
                    if self._move_toward(controller, bot_id, plate_counter, team):
                        controller.place(bot_id, plate_counter[0], plate_counter[1])
                    return False
            
            # Check cooking items
            for ing, cooker_pos in list(cooking_items.items()):
                if ing not in needed:
                    continue
                tile = controller.get_tile(team, cooker_pos[0], cooker_pos[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage >= 1:
                        # Take from pan
                        if self._is_blocked(controller, bot_id, cooker_pos, team):
                            return True
                        if self._move_toward(controller, bot_id, cooker_pos, team):
                            if controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1]):
                                del cooking_items[ing]
                                goal['cooking_items'] = cooking_items
                        return False
                else:
                    del cooking_items[ing]
                    goal['cooking_items'] = cooking_items
            
            # Check stored items
            for ing, stored_pos in list(stored_items.items()):
                if ing not in needed:
                    continue
                tile = controller.get_tile(team, stored_pos[0], stored_pos[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    if self._is_blocked(controller, bot_id, stored_pos, team):
                        return True
                    if self._move_toward(controller, bot_id, stored_pos, team):
                        if controller.pickup(bot_id, stored_pos[0], stored_pos[1]):
                            del stored_items[ing]
                            goal['stored_items'] = stored_items
                    return False
                else:
                    del stored_items[ing]
                    goal['stored_items'] = stored_items
            
            # Buy simple ingredients
            for ing in simple_queue:
                if ing not in needed:
                    continue
                if shop:
                    if self._is_blocked(controller, bot_id, shop, team):
                        return True
                    if self._move_toward(controller, bot_id, shop, team):
                        food_type = getattr(FoodType, ing, None)
                        if food_type and money >= food_type.buy_cost:
                            controller.buy(bot_id, food_type, shop[0], shop[1])
                            simple_queue.remove(ing)
                            goal['simple_queue'] = simple_queue
                    return False
            
            # Nothing to do - wait for cooking
            return False
        
        # === STATE: SUBMIT ===
        if state == 'SUBMIT':
            plate_counter = goal.get('plate_counter')
            current_turn = controller.get_turn()
            starts_at = order.get('created_turn', 0)
            
            if current_turn < starts_at:
                # Wait for order to become active
                if submit:
                    self._move_toward(controller, bot_id, submit, team)
                return False
            
            # Pick up plate if not holding
            if not holding or holding.get('type') != 'Plate':
                if plate_counter:
                    if self._is_blocked(controller, bot_id, plate_counter, team):
                        return True
                    if self._move_toward(controller, bot_id, plate_counter, team):
                        controller.pickup(bot_id, plate_counter[0], plate_counter[1])
                    return False
                else:
                    # No plate counter - something went wrong
                    goal['state'] = 'ENSURE_PLATE'
                    return False
            
            # Submit
            if submit is None:
                return True
            if self._is_blocked(controller, bot_id, submit, team):
                return True
            if self._move_toward(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self._log(f"  Bot {bot_id}: ORDER COMPLETED #{order.get('order_id')}")
                    self.completed_order_ids.add(order.get('order_id'))
                    self._clear_goal(bot_id)
            return False
        
        return False
    
    # ========== MAIN TURN EXECUTION ==========
    
    def play_turn(self, controller: RobotController) -> None:
        """Main entry point - execute both bots."""
        team = controller.get_team()
        self.team = team
        
        # Initialize on first turn
        if not self.initialized:
            self._init_log()
            self._init_map(controller, team)
        
        # Get bot IDs
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Initialize bot tracking
        if not self.bot_ids:
            self.bot_ids = list(bots)
            for bot_id in self.bot_ids:
                self.bot_goals[bot_id] = None
            
            # Detect zones
            self._detect_zones(controller, team)
        
        # Log state
        self._log_state(controller, team)
        
        current_turn = controller.get_turn()
        
        # Check for deadlock
        deadlock = self._detect_deadlock(controller, team)
        if deadlock:
            yielding_bot, priority_bot = deadlock
            self._log(f"DEADLOCK detected: Bot {yielding_bot} yields to Bot {priority_bot}")
            self._resolve_deadlock(controller, team, yielding_bot, priority_bot)
            # Priority bot still gets to act
            self._execute_single_bot_turn(controller, priority_bot, team)
            return
        
        # Execute each bot
        for bot_id in self.bot_ids:
            self._execute_single_bot_turn(controller, bot_id, team)
    
    def _execute_single_bot_turn(self, controller: RobotController, bot_id: int, team: Team) -> None:
        """Execute a single bot's turn."""
        current_turn = controller.get_turn()
        
        # Check if cooking is done
        if self._check_cooking_done(controller, bot_id, team):
            # Cooking done - goal state machine will handle pickup
            pass
        
        # Check for expired goal
        goal = self.bot_goals.get(bot_id)
        if goal:
            order = goal.get('order')
            if order:
                expires = order.get('expires_turn', 0)
                if current_turn >= expires:
                    self._log(f"  Bot {bot_id}: Order #{order.get('order_id')} EXPIRED - cleaning up")
                    self._clear_goal(bot_id)
                    goal = None
                    # Immediately trigger cleanup for any items from this order
                    if self._cleanup_clutter(controller, bot_id, team):
                        return
        
        # Select new goal if needed
        if goal is None:
            # Check if resources are too limited - wait for other bot to finish
            other_bot = self._get_other_bot_id(bot_id)
            other_goal = self.bot_goals.get(other_bot) if other_bot else None
            
            # Check if we should wait for other bot to finish
            # With limited cookers, it's better to let one bot finish completely
            # before the second bot starts, to avoid blocking delays
            total_cookers = len(self.cookers)
            available_cookers = self._count_available_cookers(controller, bot_id, team)
            
            if total_cookers == 1 and other_goal is not None:
                # Only 1 cooker on map - try to help other bot instead of waiting idle
                self._log(f"  Bot {bot_id}: Only 1 cooker, trying to help Bot {other_bot}")
                
                # Try to help with the other bot's order
                if self._help_other_bot(controller, bot_id, other_bot, team):
                    return
                
                # If can't help, trash any held items and stay out of the way
                bot = controller.get_bot_state(bot_id)
                bot_pos = (bot['x'], bot['y'])
                holding = bot.get('holding')
                if holding:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                    return
                
                # Move out of the way
                self._move_out_of_way(controller, bot_id, other_bot, team)
                return
            elif available_cookers == 0 and other_goal is not None:
                # Multiple cookers but all reserved - wait
                self._log(f"  Bot {bot_id}: No cooker available, waiting for Bot {other_bot}")
                
                bot = controller.get_bot_state(bot_id)
                bot_pos = (bot['x'], bot['y'])
                holding = bot.get('holding')
                if holding:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                    return
                
                self._move_out_of_way(controller, bot_id, other_bot, team)
                return
            
            order = self._select_order_for_bot(controller, bot_id, team)
            if order:
                starts_at = order.get('created_turn', 0)
                current_turn = controller.get_turn()
                if starts_at > current_turn:
                    self._log(f"  Bot {bot_id}: Preparing for future order #{order.get('order_id')} - {order.get('required')} (starts turn {starts_at})")
                else:
                    self._log(f"  Bot {bot_id}: Selected order #{order.get('order_id')} - {order.get('required')}")
                self.bot_goals[bot_id] = self._create_goal(order, bot_id)
            else:
                # No independent orders available
                # Try helping the other bot with their order (resource-efficient collaboration)
                if other_goal:
                    self._log(f"  Bot {bot_id}: No independent orders, helping Bot {other_bot}")
                    if self._help_other_bot(controller, bot_id, other_bot, team):
                        return  # Helping other bot
                
                # No one to help - maybe cleanup is needed
                if self._cleanup_clutter(controller, bot_id, team):
                    self._log(f"  Bot {bot_id}: Cleaning up clutter")
                    return  # Doing cleanup
                
                # Nothing to do - stay out of the way
                self._log(f"  Bot {bot_id}: Nothing to do, staying out of the way")
                
                bot = controller.get_bot_state(bot_id)
                bot_pos = (bot['x'], bot['y'])
                holding = bot.get('holding')
                if holding:
                    trash = self._get_nearest(bot_pos, self.trashes)
                    if trash:
                        if self._move_toward(controller, bot_id, trash, team):
                            controller.trash(bot_id, trash[0], trash[1])
                    return
                
                if other_goal:
                    self._move_out_of_way(controller, bot_id, other_bot, team)
                return
        
        # Check for parallel task opportunity while cooking
        if bot_id in self.cooking_wait:
            parallel_task = self._can_do_parallel_task(controller, bot_id, team)
            if parallel_task:
                self._log(f"  Bot {bot_id}: Doing parallel task '{parallel_task}' while cooking")
                # Execute parallel task through goal system
                # For simplicity, just let the goal system handle it
        
        # Execute goal
        blocked = self._execute_goal(controller, bot_id, team)
        if blocked:
            # Track consecutive blocked turns
            self.blocked_turns[bot_id] = self.blocked_turns.get(bot_id, 0) + 1
            blocked_count = self.blocked_turns[bot_id]
            self._log(f"  Bot {bot_id}: BLOCKED (turn {blocked_count})")
            
            # If blocked for too long, consider dropping the order
            if blocked_count >= 5:
                goal = self.bot_goals.get(bot_id)
                if goal:
                    order = goal.get('order', {})
                    self._log(f"  Bot {bot_id}: Blocked too long, dropping order #{order.get('order_id')}")
                    
                    # Trash any held item
                    bot = controller.get_bot_state(bot_id)
                    holding = bot.get('holding')
                    if holding:
                        bot_pos = (bot['x'], bot['y'])
                        trash = self._get_nearest(bot_pos, self.trashes)
                        if trash and self._chebyshev_dist(bot_pos, trash) <= 1:
                            controller.trash(bot_id, trash[0], trash[1])
                        elif trash:
                            self._move_toward(controller, bot_id, trash, team)
                    
                    self._clear_goal(bot_id)
                    self.blocked_turns[bot_id] = 0
        else:
            # Not blocked - reset counter
            self.blocked_turns[bot_id] = 0
