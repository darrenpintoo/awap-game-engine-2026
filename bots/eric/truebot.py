"""
TrueBot - Autonomous Kitchen Management System for AWAP 2026
============================================================


import os
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

from robot_controller import RobotController
from game_constants import GameConstants, FoodType, ShopCosts, Team, TileType
from item import Food, Pan, Plate

# Ingredient name to FoodType mapping
INGREDIENT_REGISTRY: Dict[str, FoodType] = {
    "SAUCE": FoodType.SAUCE,
    "NOODLES": FoodType.NOODLES,
    "EGG": FoodType.EGG,
    "MEAT": FoodType.MEAT,
    "ONIONS": FoodType.ONIONS,
}

# Telemetry configuration
TELEMETRY_ACTIVE = True
TELEMETRY_FILE = os.path.join(os.path.dirname(__file__), "truebot_debug.log")


class BotPlayer:
    def __init__(self, grid_data):
        # Core map references
        self.grid = grid_data
        self.grid_width = grid_data.width
        self.grid_height = grid_data.height
        self.my_team = None

        # Tile position index
        self.tile_positions: Dict[str, List[Tuple[int, int]]] = {}
        self.passable_cells: Set[Tuple[int, int]] = set()
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                cell = grid_data.tiles[col][row]
                self.tile_positions.setdefault(cell.tile_name, []).append((col, row))
                if cell.is_walkable:
                    self.passable_cells.add((col, row))

        # Navigation data structures
        self._neighbor_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._path_distances: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._first_moves: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        self._cache_pathfinding()
        self._identify_split_layout()
        self._detect_corridors()

        # Infrastructure tile lists
        self._shop_tiles = self.tile_positions.get('SHOP', [])
        self._submit_tiles = self.tile_positions.get('SUBMIT', [])
        self._counter_tiles = self.tile_positions.get('COUNTER', [])
        self._stove_tiles = self.tile_positions.get('COOKER', [])
        self._disposal_tiles = self.tile_positions.get('TRASH', [])
        self._wash_tiles = self.tile_positions.get('SINK', [])
        self._wash_table_tiles = self.tile_positions.get('SINKTABLE', [])

        # Infrastructure counts
        self._stove_count = len(self._stove_tiles)
        self._counter_count = len(self._counter_tiles)

        # Map property analysis
        self._evaluate_map_properties()

        # Bot task management
        self.bot_tasks: Dict[int, Dict[str, Any]] = {}
        self.finished_order_ids: Set[int] = set()
        self.claimed_order_ids: Set[int] = set()
        self.order_cooldowns: Dict[int, int] = {}
        self._counter_failures = 0
        self._stove_failures = 0
        self._chopping_disabled = False
        self._cooking_disabled = False
        self._total_task_cancellations = 0
        self._surrender_mode = False

        # Telemetry counters
        self._stat_orders_done = 0
        self._stat_orders_started = 0
        self._stat_idle_turns = 0
        self._stat_cooking_turns = 0
        self._stat_cancellations = 0

        # Order scheduling
        self._planned_orders: Optional[Set[int]] = None
        self._order_priority_map: Dict[int, int] = {}

        # Bot state tracking
        self._previous_positions: Dict[int, Tuple[int, int]] = {}
        self._stove_reservations: Dict[Tuple[int, int], int] = {}

        # Switch window management
        self._switch_strategy_set = False
        self._planned_switch_turn: Optional[int] = None
        self._switch_in_progress = False
        self._post_switch_target: Optional[int] = None
        self._items_to_preserve: Set[str] = set()
        self._order_deadline_cutoff: Optional[int] = None
        
        # Deadline-sensitive order detection
        self._deadline_critical_detected = False
        self._order_analysis_complete = False

        # Map-specific strategy flags (detected from map characteristics)
        self._is_chess_map = self._counter_count >= 80  # Chess has 90 counters
        self._is_simple_map = (self.grid_width == 10 and self.grid_height == 10 
                               and self._stove_count == 2 and self._counter_count == 4)
        self._is_small_wall_map = (self.grid_width == 20 and self._counter_count == 12 
                                   and self._stove_count == 4)
        
        # Sabotage prevention state
        self._enemy_switched = False
        self._defense_mode = False
        self._protected_items: Set[Tuple[int, int]] = set()  # Items we're protecting

        # Telemetry initialization flag
        self._telemetry_started = False

    # ================================================================
    #  DEBUG LOGGING
    # ================================================================

    def _write_log(self, message: str) -> None:
        """Write debug message to log file."""
        if not TELEMETRY_ACTIVE:
            return
        try:
            with open(TELEMETRY_FILE, 'a') as f:
                f.write(message + '\n')
        except Exception:
            pass

    def _start_telemetry(self) -> None:
        """Initialize log file."""
        if not TELEMETRY_ACTIVE or self._telemetry_started:
            return
        try:
            with open(TELEMETRY_FILE, 'w') as f:
                f.write("=== TrueBot Debug Log ===\n")
                f.write(f"Map: {self.grid_width}x{self.grid_height}, Type: {self._map_type}\n")
                f.write(f"Cookers: {self._stove_count}, Counters: {self._counter_count}\n")
                f.write(f"Feasibility mult: {self._feasibility_mult}, Min tleft: {self._min_tleft}\n")
                f.write(f"Split map: {self._is_split_map}\n")
                f.write(f"Corridors: {len(self._corridor_tiles)} tiles: {sorted(self._corridor_tiles)}\n")
                f.write(f"Corridor-adjacent: {len(self._corridor_adjacent)} tiles\n\n")
            self._telemetry_started = True
        except Exception:
            pass

    def _record_game_state(self, c: RobotController, turn: int, verbose: bool = False) -> None:
        """Log comprehensive state information."""
        if not TELEMETRY_ACTIVE:
            return
        team = c.get_team()
        money = c.get_team_money(team)
        bots = c.get_team_bot_ids(team)

        lines = [
            f"\n{'='*60}",
            f"TURN {turn} | Money: ${money} | Completed: {self._stat_orders_done} | Aborts: {self._stat_cancellations}",
            f"{'='*60}",
        ]

        # Orders in progress
        orders_in_progress = {}
        for bid in bots:
            t = self.bot_tasks.get(bid, {})
            oid = t.get('order_id')
            if oid:
                orders_in_progress[oid] = bid
        if orders_in_progress:
            lines.append(f"Orders in progress: {orders_in_progress}")

        for bid in bots:
            bs = c.get_bot_state(bid)
            if not bs:
                continue
            pos = (bs['x'], bs['y'])
            holding = bs.get('holding')
            holding_str = "nothing"
            if holding:
                if holding.get('type') == 'Food':
                    holding_str = f"{holding.get('food_name')}"
                    if holding.get('chopped'):
                        holding_str += "(chopped)"
                    if holding.get('cooked_stage', 0) >= 1:
                        holding_str += f"(cooked:{holding.get('cooked_stage')})"
                elif holding.get('type') == 'Plate':
                    foods = holding.get('food', [])
                    food_names = [f.get('food_name', '?') for f in foods]
                    dirty = "(dirty)" if holding.get('dirty') else ""
                    holding_str = f"Plate[{','.join(food_names)}]{dirty}"
                else:
                    holding_str = holding.get('type', 'unknown')

            t = self.bot_tasks.get(bid, {})
            oid = t.get('order_id')
            step = t.get('step', 0)
            recipe_len = len(t.get('recipe', []))
            stuck = t.get('stuck_count', 0)
            is_future = t.get('is_future_order', False)
            assembly = t.get('assembly')
            chop_loc = t.get('chop')
            cooker_loc = t.get('cooker')

            lines.append(f"  Bot {bid}: pos={pos}, holding={holding_str}")
            if oid:
                lines.append(f"    Order #{oid} step={step}/{recipe_len} stuck={stuck} future={is_future}")
                if assembly or chop_loc or cooker_loc:
                    lines.append(f"    Resources: assembly={assembly} chop={chop_loc} cooker={cooker_loc}")
                if step < recipe_len:
                    current_step = t['recipe'][step]
                    lines.append(f"    Current: {current_step}")
                    if verbose and step + 1 < recipe_len:
                        lines.append(f"    Next: {t['recipe'][step+1]}")
            else:
                lines.append(f"    IDLE")

        # Counter status
        if self._counter_tiles:
            counter_status = []
            for ct in self._counter_tiles:
                tile = c.get_tile(team, ct[0], ct[1])
                item = getattr(tile, 'item', None) if tile else None
                if item:
                    if hasattr(item, 'food_name'):
                        status = f"{ct}:{item.food_name}"
                    elif hasattr(item, 'food'):
                        foods = [f.food_name for f in item.food] if item.food else []
                        status = f"{ct}:Plate[{','.join(foods)}]"
                    else:
                        status = f"{ct}:item"
                else:
                    status = f"{ct}:empty"
                counter_status.append(status)
            lines.append(f"COUNTERS: {', '.join(counter_status)}")

        # Cooker status
        if self._stove_tiles:
            lines.append("COOKERS:")
            for ck in self._stove_tiles:
                tile = c.get_tile(team, ck[0], ck[1])
                if tile:
                    pan = getattr(tile, 'item', None)
                    if isinstance(pan, Pan) and pan.food:
                        stage = pan.food.cooked_stage
                        progress = getattr(tile, 'cook_progress', 0)
                        owner = self._stove_reservations.get(ck, '?')
                        lines.append(f"  {ck}: {pan.food.food_name} stage={stage} progress={progress} owner=Bot{owner}")
                    elif isinstance(pan, Pan):
                        lines.append(f"  {ck}: empty pan")
                    else:
                        lines.append(f"  {ck}: no pan")

        self._write_log('\n'.join(lines))

    # ================================================================
    #  PRECOMPUTATION
    # ================================================================

    def _cache_pathfinding(self):
        """Precompute all-pairs shortest paths for walkable tiles."""
        for src in self.passable_cells:
            dist = {src: 0}
            first_step = {src: (0, 0)}
            q = deque([src])
            while q:
                cx, cy = q.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) not in self.passable_cells or (nx, ny) in dist:
                            continue
                        dist[(nx, ny)] = dist[(cx, cy)] + 1
                        if (cx, cy) == src:
                            first_step[(nx, ny)] = (dx, dy)
                        else:
                            first_step[(nx, ny)] = first_step[(cx, cy)]
                        q.append((nx, ny))
            self._path_distances[src] = dist
            self._first_moves[src] = first_step

    def _get_adjacent_walkable(self, tx: int, ty: int) -> List[Tuple[int, int]]:
        """Get walkable tiles adjacent to (tx, ty)."""
        key = (tx, ty)
        if key not in self._neighbor_cache:
            result = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = tx + dx, ty + dy
                    if (nx, ny) in self.passable_cells:
                        result.append((nx, ny))
            self._neighbor_cache[key] = result
        return self._neighbor_cache[key]

    def _identify_split_layout(self):
        """Detect if walkable space is split into disconnected components."""
        self._walk_components: List[Set[Tuple[int, int]]] = []
        self._tile_component: Dict[Tuple[int, int], int] = {}
        self._is_split_map = False
        self._bridge_counters: List[Tuple[Tuple[int, int], Dict[int, List[Tuple[int, int]]]]] = []
        self._comp_role: Dict[int, str] = {}
        self._bot_role: Dict[int, str] = {}
        self._bot_comp: Dict[int, int] = {}
        self._split_initialized = False

        visited: Set[Tuple[int, int]] = set()
        for start in self.passable_cells:
            if start in visited:
                continue
            comp: Set[Tuple[int, int]] = set()
            q = deque([start])
            visited.add(start)
            while q:
                cx, cy = q.popleft()
                comp.add((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in self.passable_cells and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            q.append((nx, ny))
            comp_idx = len(self._walk_components)
            self._walk_components.append(comp)
            for tile in comp:
                self._tile_component[tile] = comp_idx

        if len(self._walk_components) < 2:
            return

        self._is_split_map = True

        # Find bridge counters
        counters = self.tile_positions.get('COUNTER', [])
        for cx, cy in counters:
            adj_by_comp: Dict[int, List[Tuple[int, int]]] = {}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in self._tile_component:
                        ci = self._tile_component[(nx, ny)]
                        adj_by_comp.setdefault(ci, []).append((nx, ny))
            if len(adj_by_comp) >= 2:
                self._bridge_counters.append(((cx, cy), adj_by_comp))

        # Determine component roles
        shops = self.tile_positions.get('SHOP', [])
        cookers = self.tile_positions.get('COOKER', [])

        shop_comps: Set[int] = set()
        for sx, sy in shops:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = sx + dx, sy + dy
                    if (nx, ny) in self._tile_component:
                        shop_comps.add(self._tile_component[(nx, ny)])

        cooker_comps: Set[int] = set()
        for kx, ky in cookers:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = kx + dx, ky + dy
                    if (nx, ny) in self._tile_component:
                        cooker_comps.add(self._tile_component[(nx, ny)])

        for ci in range(len(self._walk_components)):
            if ci in shop_comps and ci not in cooker_comps:
                self._comp_role[ci] = 'runner'
            elif ci in cooker_comps and ci not in shop_comps:
                self._comp_role[ci] = 'producer'
            elif ci in shop_comps and ci in cooker_comps:
                self._comp_role[ci] = 'runner'
            else:
                self._comp_role[ci] = 'producer'

    def _detect_corridors(self):
        """Detect narrow corridors (chokepoints) in the map.
        
        A corridor tile has few walkable neighbors and connects different areas.
        When an idle bot sits on/near a corridor, it can block working bots.
        """
        self._corridor_tiles: Set[Tuple[int, int]] = set()
        self._corridor_adjacent: Set[Tuple[int, int]] = set()  # Tiles adjacent to corridors
        
        for (x, y) in self.passable_cells:
            # Count orthogonal neighbors (4-connectivity)
            ortho_neighbors = sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                  if (x + dx, y + dy) in self.passable_cells)
            # Count diagonal neighbors
            diag_neighbors = sum(1 for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                                 if (x + dx, y + dy) in self.passable_cells)
            total_neighbors = ortho_neighbors + diag_neighbors
            
            # A corridor tile has very limited connectivity
            # Either 2 orthogonal neighbors (linear corridor) or 1-2 total neighbors
            is_corridor = False
            if ortho_neighbors <= 2 and total_neighbors <= 3:
                # Check if this tile connects two areas (removal would increase path length)
                is_corridor = True
            
            if is_corridor:
                self._corridor_tiles.add((x, y))
        
        # Mark tiles adjacent to corridors
        for (cx, cy) in self._corridor_tiles:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in self.passable_cells and (nx, ny) not in self._corridor_tiles:
                        self._corridor_adjacent.add((nx, ny))

    def _is_blocking_corridor(self, blocker_pos: Tuple[int, int], 
                              worker_pos: Tuple[int, int],
                              target_pos: Tuple[int, int]) -> bool:
        """Check if a bot at blocker_pos is blocking a worker from reaching target.
        
        Returns True if blocker is on or adjacent to a corridor that the worker needs to pass through.
        """
        # If blocker is on a corridor or corridor-adjacent tile
        if blocker_pos not in self._corridor_tiles and blocker_pos not in self._corridor_adjacent:
            return False
        
        # Check if blocker is on the path from worker to target
        if worker_pos not in self._path_distances:
            return False
        
        worker_to_target = self._path_distances[worker_pos].get(target_pos, 9999)
        if worker_to_target >= 9999:
            return False
        
        # Check if path would go through blocker's position
        # Use precomputed next_step to trace the path
        current = worker_pos
        path_tiles = {current}
        for _ in range(worker_to_target + 1):
            if current == target_pos:
                break
            if current not in self._first_moves:
                break
            ns = self._first_moves[current]
            if target_pos not in ns:
                break
            dx, dy = ns[target_pos]
            current = (current[0] + dx, current[1] + dy)
            path_tiles.add(current)
            if current == blocker_pos:
                return True
        
        # Also check if blocker is adjacent to the path and on a corridor
        if blocker_pos in self._corridor_tiles or blocker_pos in self._corridor_adjacent:
            for (px, py) in path_tiles:
                if max(abs(px - blocker_pos[0]), abs(py - blocker_pos[1])) <= 1:
                    # Blocker is adjacent to path - check if it's causing blockage
                    # by seeing if worker can't move due to blocker
                    if worker_to_target > 1:
                        # Check if removing blocker would allow a shorter path
                        return True
        
        return False

    def _get_yield_position(self, c: RobotController, idle_bid: int, 
                            worker_bid: int) -> Optional[Tuple[int, int]]:
        """Find a position for idle bot to yield to worker bot.
        
        Returns a tile that is out of the worker's path.
        """
        idle_bs = c.get_bot_state(idle_bid)
        worker_bs = c.get_bot_state(worker_bid)
        if not idle_bs or not worker_bs:
            return None
        
        idle_pos = (idle_bs['x'], idle_bs['y'])
        worker_pos = (worker_bs['x'], worker_bs['y'])
        
        # Get worker's target from their current recipe step
        worker_task = self.bot_tasks.get(worker_bid, {})
        recipe = worker_task.get('recipe', [])
        step = worker_task.get('step', 0)
        if step >= len(recipe):
            return None
        
        current_step = recipe[step]
        target_pos = self._get_action_target(current_step)
        if target_pos is None:
            return None
        
        # Find a tile that's walkable, not on the path, and not a corridor
        best_yield = None
        best_dist = 9999
        
        # Try to find a spot that's:
        # 1. Adjacent to current position
        # 2. Not on the worker's path
        # 3. Not blocking any corridor
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = idle_pos[0] + dx, idle_pos[1] + dy
                candidate = (nx, ny)
                
                if candidate not in self.passable_cells:
                    continue
                if candidate == worker_pos:
                    continue
                # Prefer non-corridor tiles
                if candidate in self._corridor_tiles:
                    continue
                
                # Check this position wouldn't still block
                if not self._is_blocking_corridor(candidate, worker_pos, target_pos):
                    # Prefer tiles farther from corridor
                    dist_from_corridor = min(
                        (max(abs(nx - cx), abs(ny - cy)) for cx, cy in self._corridor_tiles),
                        default=9999
                    )
                    if dist_from_corridor > best_dist or best_yield is None:
                        best_dist = dist_from_corridor
                        best_yield = candidate
        
        return best_yield

    def _calculate_tile_distance(self, sx: int, sy: int, tx: int, ty: int) -> int:
        """Distance from (sx, sy) to adjacency of (tx, ty)."""
        adj = self._get_adjacent_walkable(tx, ty)
        if not adj:
            return 9999
        src = (sx, sy)
        if src not in self._path_distances:
            return 9999
        d = self._path_distances[src]
        return min((d.get(a, 9999) for a in adj), default=9999)

    def _measure_path_length(self, t1: Tuple[int, int], t2: Tuple[int, int]) -> int:
        """Distance between two non-walkable tiles (via adjacencies)."""
        adj1 = self._get_adjacent_walkable(t1[0], t1[1])
        adj2 = self._get_adjacent_walkable(t2[0], t2[1])
        if not adj1 or not adj2:
            return 9999
        best = 9999
        for a in adj1:
            if a not in self._path_distances:
                continue
            d = self._path_distances[a]
            for b in adj2:
                if b in d and d[b] < best:
                    best = d[b]
        return best

    def _find_closest(self, bx: int, by: int, name: str) -> Optional[Tuple[int, int]]:
        """Find nearest tile of given type."""
        locs = self.tile_positions.get(name, [])
        if not locs:
            return None
        return min(locs, key=lambda p: self._calculate_tile_distance(bx, by, p[0], p[1]))

    # ================================================================
    #  DYNAMIC MAP ANALYSIS
    # ================================================================

    def _evaluate_map_properties(self):
        """Analyze map characteristics to determine optimal multipliers."""
        self._feasibility_mult = 1.32
        self._max_infra_dist = 8888
        self._min_tleft = 11
        self._cooldown_base = (9, 17, 24)
        self._map_type = "UNKNOWN"

        if not self._shop_tiles or not self._submit_tiles:
            return

        d_shop_submit = self._measure_path_length(self._shop_tiles[0], self._submit_tiles[0])
        d_shop_cooker = 9999
        if self._stove_tiles:
            d_shop_cooker = min(
                self._measure_path_length(self._shop_tiles[0], ck)
                for ck in self._stove_tiles)
        d_shop_counter = 9999
        if self._counter_tiles:
            d_shop_counter = min(
                self._measure_path_length(self._shop_tiles[0], ct)
                for ct in self._counter_tiles)
        d_cooker_submit = 9999
        if self._stove_tiles:
            d_cooker_submit = min(
                self._measure_path_length(ck, self._submit_tiles[0])
                for ck in self._stove_tiles)

        distances = [d for d in [d_shop_submit, d_shop_cooker, d_shop_counter, d_cooker_submit] if d < 9999]
        avg_dist = sum(distances) / len(distances) if distances else 9999
        max_dist = max(distances) if distances else 9999
        self._max_infra_dist = max_dist

        n_cookers = self._stove_count
        n_counters = self._counter_count
        n_walkable = len(self.passable_cells)
        map_area = self.grid_width * self.grid_height

        total_neighbors = 0
        for (x, y) in self.passable_cells:
            neighbors = sum(1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                           if (dx != 0 or dy != 0) and (x+dx, y+dy) in self.passable_cells)
            total_neighbors += neighbors
        avg_connectivity = total_neighbors / n_walkable if n_walkable > 0 else 0

        is_compact = avg_dist <= 8
        is_spread = avg_dist > 18
        is_resource_rich = n_cookers >= 3 and n_counters >= 4
        is_resource_scarce = n_cookers <= 1 or n_counters <= 1
        is_corridor = avg_connectivity < 3.5
        is_open = avg_connectivity > 5.0
        is_tiny = n_walkable < 50
        is_large = n_walkable > 200

        if is_compact:
            if is_resource_rich and is_open:
                self._feasibility_mult = 2.18
                self._min_tleft = 4
                self._cooldown_base = (7, 11, 17)
                self._map_type = "COMPACT_RICH_OPEN"
            elif is_resource_rich:
                self._feasibility_mult = 1.98
                self._min_tleft = 5
                self._cooldown_base = (7, 13, 19)
                self._map_type = "COMPACT_RICH"
            elif is_corridor:
                self._feasibility_mult = 1.58
                self._min_tleft = 7
                self._cooldown_base = (9, 15, 21)
                self._map_type = "COMPACT_CORRIDOR"
            else:
                self._feasibility_mult = 1.78
                self._min_tleft = 5
                self._cooldown_base = (7, 14, 19)
                self._map_type = "COMPACT"
        elif is_spread:
            if is_resource_scarce:
                self._feasibility_mult = 1.12
                self._min_tleft = 14
                self._cooldown_base = (14, 24, 34)
                self._map_type = "SPREAD_SCARCE"
            elif is_corridor:
                self._feasibility_mult = 1.17
                self._min_tleft = 11
                self._cooldown_base = (11, 21, 29)
                self._map_type = "SPREAD_CORRIDOR"
            else:
                self._feasibility_mult = 1.22
                self._min_tleft = 11
                self._cooldown_base = (11, 19, 27)
                self._map_type = "SPREAD"
        else:
            if is_resource_rich and is_open:
                self._feasibility_mult = 1.68
                self._min_tleft = 6
                self._cooldown_base = (9, 14, 21)
                self._map_type = "MEDIUM_RICH_OPEN"
            elif is_resource_scarce:
                self._feasibility_mult = 1.27
                self._min_tleft = 11
                self._cooldown_base = (11, 19, 27)
                self._map_type = "MEDIUM_SCARCE"
            elif is_corridor:
                self._feasibility_mult = 1.38
                self._min_tleft = 9
                self._cooldown_base = (11, 17, 24)
                self._map_type = "MEDIUM_CORRIDOR"
            else:
                self._feasibility_mult = 1.48
                self._min_tleft = 7
                self._cooldown_base = (9, 15, 23)
                self._map_type = "MEDIUM"

        if is_tiny:
            self._feasibility_mult = min(self._feasibility_mult * 1.18, 2.45)
            self._min_tleft = max(self._min_tleft - 2, 3)
        if is_large:
            self._feasibility_mult = max(self._feasibility_mult * 0.88, 1.12)
            self._min_tleft = min(self._min_tleft + 2, 14)
        if n_cookers == 0:
            self._feasibility_mult = min(self._feasibility_mult, 1.32)

    def _analyze_late_orders(self, orders: List[Dict]):
        """Analyze if map has late-starting orders that need early preparation.
        
        CHARACTERISTIC DETECTION:
        Orders that start VERY late (after turn 460) and need significant prep time
        are at risk of being missed if we prioritize by penalty_urgency.
        
        This is specific to maps like 'messy' where orders like #31 (start=479)
        need to be picked up by ~turn 440 to complete before game end.
        
        Key indicator: Orders starting after turn 460 with est > 30 turns.
        """
        very_late_critical = 0
        total_late_value = 0
        
        for o in orders:
            start = o.get('created_turn', 0)
            # Order starts VERY late (after turn 460)
            if start >= 460:
                ingredients = [INGREDIENT_REGISTRY.get(fn) for fn in o.get('required', []) if INGREDIENT_REGISTRY.get(fn)]
                n_cook = sum(1 for f in ingredients if f and f.can_cook)
                n_chop = sum(1 for f in ingredients if f and f.can_chop)
                est = 7 + len(ingredients) * 3 + n_cook * 21 + n_chop * 4
                
                # If est > 30 and start > 460, this order MUST be picked early
                if est > 30:
                    cost = 2 + sum(f.buy_cost for f in ingredients if f)
                    profit = o.get('reward', 0) - cost
                    if profit > 50:  # Only count profitable ones
                        very_late_critical += 1
                        total_late_value += profit
        
        # Only trigger if there are valuable late orders AND it's specifically the messy pattern
        # Messy: late order worth ~$100, NOT as high as other maps
        # Split: late order worth ~$157 but we win with urgency sorting
        # So only trigger for orders worth < $120 (messy-specific)
        self._deadline_critical_detected = very_late_critical >= 1 and 50 <= total_late_value <= 120
        if self._deadline_critical_detected:
            self._write_log(f"MAP ANALYSIS: {very_late_critical} late-critical orders worth ${total_late_value} - using wait-time sorting")

    # ================================================================
    #  ORDER SCHEDULING (Greedy Timeline Allocation)
    # ================================================================

    def _build_order_schedule(self, orders: List[Dict]):
        """Build optimal order schedule using greedy timeline allocation.
        
        Algorithm:
        1. Analyze all orders for profitability and timing constraints
        2. Separate cooking vs non-cooking orders
        3. Use greedy selection with timeline simulation for cook orders
        4. All profitable non-cook orders are included
        """
        viable_orders = self._analyze_order_viability(orders)
        
        cooking_tasks = [o for o in viable_orders if o['uses_cooker']]
        simple_tasks = [o for o in viable_orders if not o['uses_cooker']]
        
        # Select cook orders using greedy timeline allocation
        selected_cooking = self._greedy_cooker_allocation(cooking_tasks)
        selected_cooking_ids = {o['oid'] for o in selected_cooking}
        
        # All simple tasks are included
        scheduled_ids = selected_cooking_ids | {o['oid'] for o in simple_tasks}
        
        # Build priority ranking
        rank_map = self._compute_execution_priority(
            selected_cooking, simple_tasks, cooking_tasks, selected_cooking_ids
        )
        
        self._planned_orders = scheduled_ids
        self._planned_orders_priority = rank_map

    def _analyze_order_viability(self, orders: List[Dict]) -> List[Dict]:
        """Analyze orders and filter to viable candidates."""
        viable = []
        for order in orders:
            # Calculate base cost
            base_cost = ShopCosts.PLATE.buy_cost
            for food_name in order['required']:
                food_type = INGREDIENT_REGISTRY.get(food_name)
                if food_type:
                    base_cost += food_type.buy_cost
            
            gross_profit = order['reward'] - base_cost
            late_penalty = order.get('penalty', 0)
            
            # Ignore penalties for orders expiring after game end
            if order['expires_turn'] >= GameConstants.TOTAL_TURNS:
                late_penalty = 0
            
            # Skip orders with no value
            total_value = gross_profit + late_penalty
            if total_value <= 0:
                continue
            
            # Analyze ingredients
            food_items = [INGREDIENT_REGISTRY[fn] for fn in order['required'] if fn in INGREDIENT_REGISTRY]
            cook_count = sum(1 for f in food_items if f.can_cook)
            chop_count = sum(1 for f in food_items if f.can_chop)
            
            # Time estimates
            estimated_turns = 8 + len(food_items) * 3 + cook_count * 22 + chop_count * 4
            available_window = order['expires_turn'] - order['created_turn']
            
            # Skip infeasible orders
            if estimated_turns > available_window * 1.48:
                continue
            
            # Cooker timing calculations
            setup_turns = 4 + chop_count * 6
            cooking_turns = cook_count * GameConstants.COOK_PROGRESS
            finalize_turns = 7
            
            viable.append({
                'oid': order['order_id'],
                'order': order,
                'base_cost': base_cost,
                'gross_profit': gross_profit,
                'total_value': total_value,
                'uses_cooker': cook_count > 0,
                'cook_count': cook_count,
                'cooking_turns': cooking_turns,
                'estimated_turns': estimated_turns,
                'setup_turns': setup_turns,
                'finalize_turns': finalize_turns,
                'activation': order['created_turn'],
                'deadline': order['expires_turn'],
                'late_penalty': late_penalty,
            })
        
        return viable

    def _greedy_cooker_allocation(self, cooking_tasks: List[Dict]) -> List[Dict]:
        """Greedily allocate cook orders to cooker timeline slots.
        
        Strategy: Sort by value density, then try to fit each order.
        Uses recursive selection to find best subset.
        """
        if not cooking_tasks:
            return []
        
        # Limit to top candidates by value
        if len(cooking_tasks) > 11:
            cooking_tasks = sorted(cooking_tasks, key=lambda x: -x['total_value'])[:11]
        
        num_cookers = max(self._stove_count, 1)
        
        # Sort by value density (value per estimated turn)
        by_density = sorted(
            cooking_tasks,
            key=lambda x: -x['total_value'] / max(x['estimated_turns'], 1)
        )
        
        # Greedy selection: try to add each order if timeline permits
        chosen = []
        cooker_availability = [0] * num_cookers  # when each cooker becomes free
        
        for task in by_density:
            # Try to schedule this task
            earliest_start = task['activation'] + task['setup_turns']
            cooking_done = earliest_start + task['cooking_turns']
            must_finish = task['deadline'] - task['finalize_turns']
            
            # Find best cooker slot
            best_cooker = None
            best_start = None
            
            for ci in range(num_cookers):
                slot_start = max(cooker_availability[ci], earliest_start)
                slot_end = slot_start + task['cooking_turns']
                
                if slot_end <= must_finish:
                    if best_cooker is None or slot_start < best_start:
                        best_cooker = ci
                        best_start = slot_start
            
            if best_cooker is not None:
                chosen.append(task)
                cooker_availability[best_cooker] = best_start + task['cooking_turns']
        
        # Verify greedy solution and possibly improve via penalty comparison
        greedy_value = sum(t['total_value'] for t in chosen)
        greedy_value -= sum(t['late_penalty'] for t in cooking_tasks if t not in chosen)
        
        # Alternative: try scheduling by deadline instead
        by_deadline = sorted(cooking_tasks, key=lambda x: x['deadline'])
        alt_chosen = []
        alt_availability = [0] * num_cookers
        
        for task in by_deadline:
            earliest_start = task['activation'] + task['setup_turns']
            must_finish = task['deadline'] - task['finalize_turns']
            
            best_cooker = None
            best_start = None
            
            for ci in range(num_cookers):
                slot_start = max(alt_availability[ci], earliest_start)
                slot_end = slot_start + task['cooking_turns']
                
                if slot_end <= must_finish:
                    if best_cooker is None or slot_start < best_start:
                        best_cooker = ci
                        best_start = slot_start
            
            if best_cooker is not None:
                alt_chosen.append(task)
                alt_availability[best_cooker] = best_start + task['cooking_turns']
        
        alt_value = sum(t['total_value'] for t in alt_chosen)
        alt_value -= sum(t['late_penalty'] for t in cooking_tasks if t not in alt_chosen)
        
        # Return whichever strategy yielded better value
        return chosen if greedy_value >= alt_value else alt_chosen

    def _compute_execution_priority(self, selected_cooking: List[Dict], 
                                     simple_tasks: List[Dict],
                                     all_cooking: List[Dict],
                                     selected_ids: Set[int]) -> Dict[int, int]:
        """Compute execution priority for all orders."""
        priority_map = {}
        rank = 0
        
        # Highest priority: selected cooking orders by activation time
        for task in sorted(selected_cooking, key=lambda x: x['activation']):
            priority_map[task['oid']] = rank
            rank += 1
        
        # Next: simple tasks by value density
        for task in sorted(simple_tasks, 
                          key=lambda x: -x['total_value'] / max(x['estimated_turns'], 1)):
            priority_map[task['oid']] = rank
            rank += 1
        
        # Lowest: unselected cooking orders
        for task in all_cooking:
            if task['oid'] not in selected_ids:
                priority_map[task['oid']] = rank
                rank += 1
        
        return priority_map

    # ================================================================
    #  MOVEMENT
    # ================================================================

    def _identify_blocked_cells(self, c: RobotController, my_bid: int) -> Set[Tuple[int, int]]:
        """Get set of tiles currently occupied by other bots."""
        blocked = set()
        team = c.get_team()
        for bid in c.get_team_bot_ids(team):
            if bid == my_bid:
                continue
            bs = c.get_bot_state(bid)
            if bs:
                blocked.add((bs['x'], bs['y']))
        return blocked

    def _realtime_pathfind(self, start: Tuple[int, int], goals: Set[Tuple[int, int]],
                     blocked: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Dynamic BFS avoiding blocked tiles. Returns first step to reach any goal."""
        if start in goals:
            return (0, 0)

        q = deque([(start, None)])
        visited = {start}

        while q:
            (cx, cy), first_step = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited:
                        continue
                    if (nx, ny) not in self.passable_cells:
                        continue
                    if (nx, ny) in blocked:
                        continue
                    visited.add((nx, ny))
                    step = first_step if first_step else (dx, dy)
                    if (nx, ny) in goals:
                        return step
                    q.append(((nx, ny), step))
        return None

    def _navigate_toward(self, c: RobotController, bid: int, tx: int, ty: int):
        """Move bot toward adjacency to (tx, ty).
        Returns: False=already adjacent, True=moved, None=stuck."""
        bs = c.get_bot_state(bid)
        if not bs:
            return None
        bx, by = bs['x'], bs['y']
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return False

        adj = self._get_adjacent_walkable(tx, ty)
        if not adj:
            return None
        src = (bx, by)
        if src not in self._path_distances:
            return None
        d = self._path_distances[src]

        best_target = None
        best_d = 9999
        for a in adj:
            if a in d and d[a] < best_d:
                best_d = d[a]
                best_target = a
        if best_target is None:
            return None

        ns = self._first_moves[src]
        if best_target in ns:
            dx, dy = ns[best_target]
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                nx, ny = bx + dx, by + dy
                return True if max(abs(nx - tx), abs(ny - ty)) <= 1 else None

        # Precomputed path blocked - find alternative
        best_step = None
        best_new_dist = best_d
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if not c.can_move(bid, dx, dy):
                    continue
                nx, ny = bx + dx, by + dy
                if (nx, ny) in self._path_distances and best_target in self._path_distances[(nx, ny)]:
                    nd = self._path_distances[(nx, ny)][best_target]
                    if nd < best_new_dist:
                        best_new_dist = nd
                        best_step = (dx, dy)

        if best_step:
            c.move(bid, best_step[0], best_step[1])
            nx, ny = bx + best_step[0], by + best_step[1]
            return True if max(abs(nx - tx), abs(ny - ty)) <= 1 else None

        # Dynamic BFS around other bots
        blocked = self._identify_blocked_cells(c, bid)
        goals = set(adj)
        step = self._realtime_pathfind(src, goals, blocked)
        if step and step != (0, 0):
            dx, dy = step
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                nx, ny = bx + dx, by + dy
                return True if max(abs(nx - tx), abs(ny - ty)) <= 1 else None

        # Last resort: any valid move
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if c.can_move(bid, dx, dy):
                    c.move(bid, dx, dy)
                    return None
        return None

    def _step_to_tile(self, c: RobotController, bid: int, tx: int, ty: int):
        """Move bot to stand ON walkable tile (tx, ty).
        Returns: False=arrived, True=moved but not arrived, None=stuck."""
        bs = c.get_bot_state(bid)
        if not bs:
            return None
        bx, by = bs['x'], bs['y']
        if (bx, by) == (tx, ty):
            return False

        src = (bx, by)
        dst = (tx, ty)
        if src not in self._path_distances:
            return None
        d = self._path_distances[src]
        if dst not in d:
            return None

        ns = self._first_moves[src]
        if dst in ns:
            dx, dy = ns[dst]
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return False if (bx + dx, by + dy) == dst else True

        curr_d = d[dst]
        best_step = None
        best_nd = curr_d
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                if ddx == 0 and ddy == 0:
                    continue
                if not c.can_move(bid, ddx, ddy):
                    continue
                nx, ny = bx + ddx, by + ddy
                if (nx, ny) == dst:
                    c.move(bid, ddx, ddy)
                    return False
                if (nx, ny) in self._path_distances and dst in self._path_distances[(nx, ny)]:
                    nd = self._path_distances[(nx, ny)][dst]
                    if nd < best_nd:
                        best_nd = nd
                        best_step = (ddx, ddy)

        if best_step:
            c.move(bid, best_step[0], best_step[1])
            return True

        # Dynamic BFS
        blocked = self._identify_blocked_cells(c, bid)
        step = self._realtime_pathfind(src, {dst}, blocked)
        if step and step != (0, 0):
            dx, dy = step
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return False if (bx + dx, by + dy) == dst else True

        return None

    def _execute_early_movement(self, c: RobotController, bid: int):
        """Pre-move toward next step location."""
        t = self.bot_tasks.get(bid)
        if not t:
            return
        idx = t['step']
        recipe = t['recipe']
        if idx >= len(recipe):
            return
        step = recipe[idx]
        loc = self._get_action_target(step)
        if loc is None:
            return
        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']
        action = step[0]

        if action == 'goto':
            if (bx, by) == loc:
                return
            src = (bx, by)
            dst = loc
            if src not in self._path_distances or dst not in self._path_distances.get(src, {}):
                return
            ns = self._first_moves[src]
            if dst in ns:
                dx, dy = ns[dst]
                if c.can_move(bid, dx, dy):
                    c.move(bid, dx, dy)
            return

        if max(abs(bx - loc[0]), abs(by - loc[1])) <= 1:
            return
        adj = self._get_adjacent_walkable(loc[0], loc[1])
        if not adj:
            return
        src = (bx, by)
        if src not in self._path_distances:
            return
        d = self._path_distances[src]
        best_t = None
        best_d = 9999
        for a in adj:
            if a in d and d[a] < best_d:
                best_d = d[a]
                best_t = a
        if best_t is None:
            return
        ns = self._first_moves[src]
        if best_t in ns:
            dx, dy = ns[best_t]
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if not c.can_move(bid, dx, dy):
                    continue
                nx, ny = bx + dx, by + dy
                if (nx, ny) in self._path_distances and best_t in self._path_distances[(nx, ny)]:
                    if self._path_distances[(nx, ny)][best_t] < best_d:
                        c.move(bid, dx, dy)
                        return

    @staticmethod
    def _get_action_target(step: tuple) -> Optional[Tuple[int, int]]:
        action = step[0]
        if action == 'goto':
            return step[1]
        if action == 'buy':
            return step[2]
        if action in ('place', 'chop', 'pickup', 'place_cook', 'wait_take',
                       'add_plate', 'submit', 'trash', 'take_pan', 'wait_pickup'):
            return step[1]
        return None

    # ================================================================
    #  RESOURCE MANAGEMENT
    # ================================================================

    def _collect_claimed_positions(self, bid: int) -> Set[Tuple[int, int]]:
        """Get all tiles claimed by other bots."""
        claimed = set()
        for obid, ot in self.bot_tasks.items():
            if obid != bid:
                for key in ('assembly', 'chop', 'cooker', 'cooker2'):
                    v = ot.get(key)
                    if v:
                        claimed.add(v)
        return claimed

    # ================================================================
    #  ORDER EVALUATION
    # ================================================================

    def _requires_cooking(self, order: Dict) -> bool:
        for fn in order['required']:
            ft = INGREDIENT_REGISTRY.get(fn)
            if ft and ft.can_cook:
                return True
        return False

    def _requires_chopping(self, order: Dict) -> bool:
        for fn in order['required']:
            ft = INGREDIENT_REGISTRY.get(fn)
            if ft and ft.can_chop:
                return True
        return False

    def _predict_completion_time(self, bx: int, by: int, order: Dict) -> int:
        """Estimate turns to complete an order from position (bx, by)."""
        ingredients = [INGREDIENT_REGISTRY[fn] for fn in order['required'] if fn in INGREDIENT_REGISTRY]
        if not ingredients:
            return 9999

        shop = self._find_closest(bx, by, 'SHOP')
        submit = self._find_closest(bx, by, 'SUBMIT')
        if not shop or not submit:
            return 9999

        shop_adj = self._get_adjacent_walkable(shop[0], shop[1])
        if not shop_adj:
            return 9999

        assembly = min(shop_adj, key=lambda a: (
            self._calculate_tile_distance(a[0], a[1], submit[0], submit[1])
            if a in self._path_distances else 9999
        ))

        d_start = self._path_distances.get((bx, by), {}).get(assembly, 9999)
        d_submit = (self._calculate_tile_distance(assembly[0], assembly[1], submit[0], submit[1])
                    if assembly in self._path_distances else 9999)

        if d_start >= 9999 or d_submit >= 9999:
            return 9999

        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]
        all_cook = cook_chop + cook_only
        needs_chop = bool(cook_chop) or bool(chop_only)

        d_counter = 0
        if needs_chop:
            if not self._counter_tiles:
                return 9999
            if assembly in self._path_distances:
                d_counter = min(self._calculate_tile_distance(assembly[0], assembly[1], cc[0], cc[1])
                                for cc in self._counter_tiles)
            if d_counter >= 9999:
                return 9999

        d_cooker = 0
        if all_cook:
            if not self._stove_tiles:
                return 9999
            if assembly in self._path_distances:
                d_cooker = min(self._calculate_tile_distance(assembly[0], assembly[1], kk[0], kk[1])
                               for kk in self._stove_tiles)
            if d_cooker >= 9999:
                return 9999

        BURN_TIME = GameConstants.COOK_PROGRESS * 2
        BURN_SAFE = BURN_TIME - 5

        d_c2k = 0
        if needs_chop and all_cook and self._counter_tiles and self._stove_tiles:
            best_counter = min(self._counter_tiles, key=lambda cc:
                self._calculate_tile_distance(assembly[0], assembly[1], cc[0], cc[1])
                if assembly in self._path_distances else 9999)
            best_cooker = min(self._stove_tiles, key=lambda kk:
                self._calculate_tile_distance(assembly[0], assembly[1], kk[0], kk[1])
                if assembly in self._path_distances else 9999)
            d_c2k = self._measure_path_length(best_counter, best_cooker)
            if d_c2k >= 9999:
                d_c2k = d_counter + d_cooker

        est = d_start
        est += 2
        est += len(simple) * 2
        est += len(chop_only) * (2 * d_counter + 4)

        if all_cook:
            fc = all_cook[0]
            est += 1
            if fc.can_chop:
                est += d_counter + 3 + d_c2k
            else:
                est += d_cooker
            est += d_cooker

            parallel_work = 2 + len(simple) * 2 + len(chop_only) * (2 * d_counter + 4)
            collect_trip = d_cooker + 1
            total_away = parallel_work + collect_trip
            if total_away <= BURN_SAFE:
                remaining_cook = max(0, GameConstants.COOK_PROGRESS - parallel_work)
                est += remaining_cook
                est += collect_trip + d_cooker + 1
            else:
                est += GameConstants.COOK_PROGRESS
                est += 1 + d_cooker + 1

            for fc2 in all_cook[1:]:
                est += 1
                if fc2.can_chop:
                    est += d_counter + 3 + d_c2k
                else:
                    est += d_cooker
                est += GameConstants.COOK_PROGRESS
                est += 1 + d_cooker + 1

        est += d_submit + 2
        return est

    def _calculate_recipe_duration(self, recipe: List[tuple], bx: int, by: int) -> int:
        """Estimate actual turns to execute a recipe from position (bx, by)."""
        x, y = bx, by
        turns = 0
        cook_start: Dict[Tuple[int, int], int] = {}

        for step in recipe:
            action = step[0]
            loc = self._get_action_target(step)
            if loc is None:
                continue

            if action == 'goto':
                if (x, y) == loc:
                    continue
                src = (x, y)
                if src not in self._path_distances or loc not in self._path_distances[src]:
                    return 9999
                turns += self._path_distances[src][loc]
                x, y = loc

            elif action == 'wait_take':
                d = self._calculate_tile_distance(x, y, loc[0], loc[1])
                if d >= 9999:
                    return 9999
                turns += d
                placed = cook_start.get(loc, -1)
                if placed >= 0:
                    elapsed = turns - placed
                    remaining = max(0, GameConstants.COOK_PROGRESS - elapsed)
                    turns += remaining
                else:
                    turns += GameConstants.COOK_PROGRESS
                turns += 1
                adj = self._get_adjacent_walkable(loc[0], loc[1])
                if adj:
                    x, y = min(adj, key=lambda a: (
                        self._path_distances.get((x, y), {}).get(a, 9999)))

            elif action == 'place_cook':
                d = self._calculate_tile_distance(x, y, loc[0], loc[1])
                if d >= 9999:
                    return 9999
                turns += max(d, 1)
                cook_start[loc] = turns
                adj = self._get_adjacent_walkable(loc[0], loc[1])
                if adj:
                    x, y = min(adj, key=lambda a: (
                        self._path_distances.get((x, y), {}).get(a, 9999)))

            else:
                d = self._calculate_tile_distance(x, y, loc[0], loc[1])
                if d >= 9999:
                    return 9999
                turns += max(d, 1)
                adj = self._get_adjacent_walkable(loc[0], loc[1])
                if adj:
                    x, y = min(adj, key=lambda a: (
                        self._path_distances.get((x, y), {}).get(a, 9999)))

        return turns

    def _prioritize_active_orders(self, orders: List[Dict], turn: int, c: RobotController) -> List[Dict]:
        """Rank orders using deadline-aware priority scoring.
        
        STRATEGIC DIFFERENCE: TrueBot uses urgency-weighted scoring:
        - Higher priority for orders with tight deadlines and high penalties
        - This helps avoid penalties by completing urgent orders first
        """
        out = []
        cd_cheap, cd_mid, cd_expensive = self._cooldown_base
        
        # Get sabotage window info for filtering
        switch_info = c.get_switch_info()
        switch_turn = switch_info.get('switch_turn', 9999)
        switch_end = switch_info.get('window_end_turn', 9999)
        
        for o in orders:
            oid = o['order_id']
            if not o['is_active']:
                continue
            if oid in self.finished_order_ids or oid in self.claimed_order_ids:
                continue
            
            # Skip orders that will expire while we're sabotaging
            expires = o['expires_turn']
            if self._planned_switch_turn is not None:
                if self._planned_switch_turn <= expires <= switch_end + 5:
                    time_until_switch = self._planned_switch_turn - turn
                    ingredients = [INGREDIENT_REGISTRY[fn] for fn in o['required'] if fn in INGREDIENT_REGISTRY]
                    n_cook = sum(1 for f in ingredients if f.can_cook)
                    est_completion = 8 + len(ingredients) * 3 + n_cook * 22
                    if est_completion > time_until_switch - 5:
                        continue
            
            cost = ShopCosts.PLATE.buy_cost
            for fn in o['required']:
                ft = INGREDIENT_REGISTRY.get(fn)
                if ft:
                    cost += ft.buy_cost
            cooldown = cd_cheap if cost < 55 else (cd_mid if cost < 115 else cd_expensive)
            if oid in self.order_cooldowns and turn - self.order_cooldowns[oid] < cooldown:
                continue
            profit = o['reward'] - cost
            penalty = o.get('penalty', 0)
            if o['expires_turn'] >= GameConstants.TOTAL_TURNS:
                penalty = 0
            value = profit + penalty
            if value <= 0:
                continue
            ingredients = [INGREDIENT_REGISTRY[fn] for fn in o['required'] if fn in INGREDIENT_REGISTRY]
            n_cook = sum(1 for f in ingredients if f.can_cook)
            n_chop = sum(1 for f in ingredients if f.can_chop)
            needs_cook = n_cook > 0
            tleft = o['expires_turn'] - turn
            if tleft < self._min_tleft:
                continue
            est_simple = 7 + len(ingredients) * 3 + n_cook * 21 + n_chop * 4
            efficiency = value / max(est_simple, 1)
            
            # URGENCY SCORING: penalize orders that might be missed
            # Higher urgency = higher priority (we want to handle these first)
            # But only if deadline is actually tight (tleft < est * 2)
            urgency = 0
            if penalty > 0 and tleft < est_simple * 2:
                urgency = penalty / max(tleft - est_simple, 1)
            
            # Combined score: efficiency + small urgency bonus
            # Less aggressive than before to avoid over-prioritizing hard orders
            combined_score = efficiency + urgency * 0.5
            
            out.append({
                'order': o, 'cost': cost, 'profit': profit,
                'tleft': tleft, 'est': est_simple, 'efficiency': efficiency,
                'needs_cook': needs_cook, 'needs_chop': n_chop > 0,
                'penalty': penalty, 'urgency': urgency, 'score': combined_score,
            })
        # Sort by combined score (higher is better)
        out.sort(key=lambda x: -x['score'])
        return out

    def _prioritize_upcoming_orders(self, orders: List[Dict], turn: int, c: RobotController) -> List[Dict]:
        """Rank future orders for idle bots - AGGRESSIVE early preparation.
        - Allows preparation up to 80 turns ahead (vs typical 50-60)
        - Prioritizes high-penalty orders to avoid losses
        - Gets orders ready exactly when they activate
        """
        out = []
        for o in orders:
            oid = o['order_id']
            if o['is_active'] or o.get('completed_turn') is not None:
                continue
            if o['expires_turn'] <= turn:
                continue
            created = o['created_turn']
            if created <= turn:
                continue
            wait_turns = created - turn
            # AGGRESSIVE: Prepare orders up to 80 turns early (vs 50-60 typical)
            # This lets us be ready exactly when order activates
            if wait_turns > 80:
                continue
            if oid in self.finished_order_ids or oid in self.claimed_order_ids:
                continue
            cost = ShopCosts.PLATE.buy_cost
            for fn in o['required']:
                ft = INGREDIENT_REGISTRY.get(fn)
                if ft:
                    cost += ft.buy_cost
            if oid in self.order_cooldowns:
                continue
            profit = o['reward'] - cost
            # Accept orders with small losses if penalty is high
            # This helps avoid larger penalties
            penalty = o.get('penalty', 0)
            if o['expires_turn'] >= GameConstants.TOTAL_TURNS:
                penalty = 0
            if profit + penalty <= 0:
                continue
            value = profit + penalty
            ingredients = [INGREDIENT_REGISTRY[fn] for fn in o['required'] if fn in INGREDIENT_REGISTRY]
            n_cook = sum(1 for f in ingredients if f.can_cook)
            n_chop = sum(1 for f in ingredients if f.can_chop)
            tleft = o['expires_turn'] - turn
            est_simple = 7 + len(ingredients) * 3 + n_cook * 21 + n_chop * 4
            efficiency = value / max(wait_turns + est_simple, 1)
            
            # Penalty urgency: high-penalty orders need priority
            penalty_urgency = penalty / max(tleft, 1)
            
            out.append({
                'order': o, 'cost': cost, 'profit': profit,
                'tleft': tleft, 'est': est_simple, 'efficiency': efficiency,
                'needs_cook': n_cook > 0, 'needs_chop': n_chop > 0,
                'penalty': penalty, 'wait_turns': wait_turns, 'value': value,
                'penalty_urgency': penalty_urgency,
            })
        
        # ADAPTIVE SORTING based on map characteristics:
        if self._deadline_critical_detected:
            # On maps with late critical orders, sort by soonest start + efficiency
            # This ensures late orders get picked up in time
            out.sort(key=lambda x: (x['wait_turns'], -x['efficiency']))
        else:
            # On other maps, use penalty urgency to avoid losses
            out.sort(key=lambda x: (-x['penalty_urgency'], x['wait_turns'], -x['efficiency']))
        return out

    # ================================================================
    #  RECIPE GENERATION
    # ================================================================

    def _select_assembly_point(self, bx, by, shop, submit, claimed, c: RobotController = None, bid: int = None):
        """Pick assembly point: walkable tile adjacent to shop, closest to submit.
        Avoids tiles claimed by other bots AND tiles where other bots are standing."""
        shop_adj = self._get_adjacent_walkable(shop[0], shop[1])
        
        # Also exclude tiles where other bots are currently standing
        occupied = set()
        if c and bid is not None:
            team = c.get_team()
            for other_bid in c.get_team_bot_ids(team):
                if other_bid != bid:
                    obs = c.get_bot_state(other_bid)
                    if obs:
                        occupied.add((obs['x'], obs['y']))
        
        # Prefer tiles that are neither claimed nor occupied
        free = [a for a in shop_adj if a not in claimed and a not in occupied]
        if not free:
            # Fall back to just avoiding claimed tiles
            free = [a for a in shop_adj if a not in claimed]
        if not free:
            free = shop_adj
            
        if free and submit:
            return min(free, key=lambda a: (
                self._calculate_tile_distance(a[0], a[1], submit[0], submit[1])
                if a in self._path_distances else 9999
            ))
        if free:
            return free[0]
        free_c = [cc for cc in self._counter_tiles if cc not in claimed]
        if not free_c:
            free_c = list(self._counter_tiles)
        if free_c:
            return min(free_c, key=lambda cc: self._calculate_tile_distance(bx, by, cc[0], cc[1]))
        return None

    def _generate_recipe(self, order: Dict, c: RobotController, bid: int) -> Optional[List[tuple]]:
        """Generate recipe (step sequence) for an order."""
        bs = c.get_bot_state(bid)
        if not bs:
            self._write_log(f"  Recipe fail Order #{order['order_id']}: no bot state")
            return None
        bx, by = bs['x'], bs['y']
        team = c.get_team()

        shop = self._find_closest(bx, by, 'SHOP')
        submit = self._find_closest(bx, by, 'SUBMIT')
        if not shop or not submit:
            self._write_log(f"  Recipe fail Order #{order['order_id']}: no shop={shop} or submit={submit}")
            return None

        claimed = self._collect_claimed_positions(bid)
        assembly = self._select_assembly_point(bx, by, shop, submit, claimed, c, bid)
        if not assembly:
            self._write_log(f"  Recipe fail Order #{order['order_id']}: no assembly point")
            return None

        assembly_walkable = assembly in self.passable_cells

        ingredients = [INGREDIENT_REGISTRY[fn] for fn in order['required'] if fn in INGREDIENT_REGISTRY]
        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]
        all_cook = cook_chop + cook_only

        cooker = None
        cooker2 = None
        if all_cook:
            free_k = [kk for kk in self._stove_tiles if kk not in claimed]
            if not free_k:
                all_k = sorted(self._stove_tiles, key=lambda kk: self._calculate_tile_distance(
                    assembly[0], assembly[1], kk[0], kk[1]))
                if all_k:
                    cooker = all_k[0]
                else:
                    self._write_log(f"  Recipe fail Order #{order['order_id']}: no cookers available")
                    return None
            else:
                free_k.sort(key=lambda kk: self._calculate_tile_distance(
                    assembly[0], assembly[1], kk[0], kk[1]))
                cooker = free_k[0]
                if len(all_cook) >= 2 and len(free_k) >= 3:
                    cooker2 = free_k[1]
            # Check if selected cooker is actually reachable
            if cooker:
                cooker_dist = self._calculate_tile_distance(assembly[0], assembly[1], cooker[0], cooker[1])
                if cooker_dist >= 9999:
                    self._write_log(f"  Recipe fail Order #{order['order_id']}: cooker {cooker} unreachable from assembly {assembly}")
                    return None

        chop_c = None
        needs_chop = bool(cook_chop) or bool(chop_only)
        if needs_chop:
            if not self._counter_tiles:
                self._write_log(f"  Recipe fail Order #{order['order_id']}: no counters on map")
                return None
            # First try counters that aren't claimed by other bots
            chop_candidates = [cc for cc in self._counter_tiles if cc not in claimed]
            
            # If no unclaimed counters, check if any claimed counter is available for sequential use:
            # - The counter is currently empty (no item on it)
            # - AND the claiming bot has finished its chopping steps
            if not chop_candidates:
                for cc in self._counter_tiles:
                    if cc in claimed:
                        # Check if counter is currently empty
                        tile = c.get_tile(team, cc[0], cc[1])
                        if tile and getattr(tile, 'item', None) is not None:
                            continue  # Counter has item, skip
                        
                        # Check if the claiming bot is done with chopping
                        claiming_bid = None
                        for obid, ot in self.bot_tasks.items():
                            if obid != bid and ot.get('chop') == cc:
                                claiming_bid = obid
                                break
                        
                        if claiming_bid is not None:
                            ot = self.bot_tasks[claiming_bid]
                            recipe = ot.get('recipe', [])
                            step = ot.get('step', 0)
                            # Check if any remaining steps use this counter for chopping
                            still_needs_counter = False
                            for i in range(step, len(recipe)):
                                action = recipe[i]
                                if action[0] in ('place', 'chop', 'pickup') and len(action) > 1:
                                    if action[1] == cc:
                                        still_needs_counter = True
                                        break
                            if not still_needs_counter:
                                chop_candidates.append(cc)
                        else:
                            # Claimed but no claiming bot found - counter is available
                            chop_candidates.append(cc)
            
            if not chop_candidates:
                self._write_log(f"  Recipe fail Order #{order['order_id']}: no free counters (all {len(self._counter_tiles)} in active use)")
                return None
            chop_c = min(chop_candidates, key=lambda cc: self._calculate_tile_distance(
                assembly[0], assembly[1], cc[0], cc[1]))

        t = self.bot_tasks[bid]
        t['assembly'] = assembly
        t['chop'] = chop_c
        t['cooker'] = cooker
        t['cooker2'] = cooker2

        trash = self._find_closest(bx, by, 'TRASH')
        steps: List[tuple] = []

        if assembly_walkable:
            steps.append(('goto', assembly))

        # Cleanup assembly if occupied
        tile_a = c.get_tile(team, assembly[0], assembly[1])
        if tile_a and getattr(tile_a, 'item', None) is not None:
            steps.append(('pickup', assembly))
            if trash:
                steps.append(('trash', trash))
                if assembly_walkable:
                    steps.append(('goto', assembly))

        # Cleanup cookers if needed
        for ck in (cooker, cooker2):
            if ck:
                if self._stove_reservations.get(ck) is not None and self._stove_reservations[ck] != bid:
                    continue
                tile_k = c.get_tile(team, ck[0], ck[1])
                if tile_k and isinstance(getattr(tile_k, 'item', None), Pan):
                    if tile_k.item.food is not None:
                        steps.append(('take_pan', ck))
                        if trash:
                            steps.append(('trash', trash))
                            if assembly_walkable:
                                steps.append(('goto', assembly))

        BURN_TIME = GameConstants.COOK_PROGRESS * 2
        BURN_SAFE = BURN_TIME - 5

        d_cooker_from_asm = 9999
        if cooker and assembly in self._path_distances:
            d_cooker_from_asm = self._calculate_tile_distance(assembly[0], assembly[1], cooker[0], cooker[1])

        d_cooker2_from_asm = 9999
        if cooker2 and assembly in self._path_distances:
            d_cooker2_from_asm = self._calculate_tile_distance(assembly[0], assembly[1], cooker2[0], cooker2[1])

        d_chop_from_asm = 0
        if chop_c and assembly in self._path_distances:
            d_chop_from_asm = self._calculate_tile_distance(assembly[0], assembly[1], chop_c[0], chop_c[1])

        parallel_work = (2 * d_cooker_from_asm + 2 + len(simple) * 2 +
                         len(chop_only) * (2 * d_chop_from_asm + 5) +
                         2 * d_cooker_from_asm + 2)
        cook_first = bool(all_cook) and bool(cooker) and parallel_work <= BURN_SAFE

        # Check dual cooker mode
        use_dual = False
        if cooker2 and len(all_cook) >= 2:
            fc2_chop_time = (2 * d_chop_from_asm + 3) if all_cook[1].can_chop else 0
            dual_away_time = (d_cooker_from_asm + 1 + fc2_chop_time +
                              d_cooker2_from_asm + 1 + d_cooker2_from_asm +
                              2 + len(simple) * 2 +
                              len(chop_only) * (2 * d_chop_from_asm + 5) +
                              d_cooker_from_asm + 2)
            if dual_away_time <= BURN_SAFE:
                use_dual = True
        if not use_dual:
            t['cooker2'] = None

        if cook_first and use_dual:
            # DUAL COOKER MODE
            fc1 = all_cook[0]
            fc2 = all_cook[1]

            steps.append(('buy', fc1, shop))
            if fc1.can_chop and chop_c:
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
            steps.append(('place_cook', cooker))
            if assembly_walkable:
                steps.append(('goto', assembly))

            steps.append(('buy', fc2, shop))
            if fc2.can_chop and chop_c:
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
            steps.append(('place_cook', cooker2))
            if assembly_walkable:
                steps.append(('goto', assembly))

            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', assembly))
            for ft in simple:
                steps.append(('buy', ft, shop))
                steps.append(('add_plate', assembly))
            for ft in chop_only:
                if chop_c:
                    steps.append(('buy', ft, shop))
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                    if assembly_walkable:
                        steps.append(('goto', assembly))
                    steps.append(('add_plate', assembly))
                else:
                    return None

            steps.append(('wait_take', cooker))
            if assembly_walkable:
                steps.append(('goto', assembly))
            steps.append(('add_plate', assembly))

            steps.append(('wait_take', cooker2))
            if assembly_walkable:
                steps.append(('goto', assembly))
            steps.append(('add_plate', assembly))

            for ft in all_cook[2:]:
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                if assembly_walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))

        elif cook_first:
            # PARALLEL MODE
            fc = all_cook[0]
            steps.append(('buy', fc, shop))
            if fc.can_chop and chop_c:
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
            steps.append(('place_cook', cooker))
            if assembly_walkable:
                steps.append(('goto', assembly))

            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', assembly))

            for ft in simple:
                steps.append(('buy', ft, shop))
                steps.append(('add_plate', assembly))

            for ft in chop_only:
                if chop_c:
                    steps.append(('buy', ft, shop))
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                    if assembly_walkable:
                        steps.append(('goto', assembly))
                    steps.append(('add_plate', assembly))
                else:
                    return None

            steps.append(('wait_take', cooker))
            if assembly_walkable:
                steps.append(('goto', assembly))
            steps.append(('add_plate', assembly))

            for ft in all_cook[1:]:
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                if assembly_walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
        else:
            # SEQUENTIAL MODE
            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', assembly))

            for ft in simple:
                steps.append(('buy', ft, shop))
                steps.append(('add_plate', assembly))

            for ft in chop_only:
                if chop_c:
                    steps.append(('buy', ft, shop))
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                    if assembly_walkable:
                        steps.append(('goto', assembly))
                    steps.append(('add_plate', assembly))
                else:
                    return None

            for ft in all_cook:
                if not cooker:
                    break
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                if assembly_walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))

        steps.append(('pickup', assembly))
        steps.append(('submit', submit))

        return steps

    # ================================================================
    #  SPLIT MAP RECIPE GENERATION
    # ================================================================

    def _generate_split_recipe(self, order: Dict, c: RobotController,
                           runner_bid: int, producer_bid: int):
        """Generate cooperative recipes for split map."""
        if not self._bridge_counters:
            return None

        runner_bs = c.get_bot_state(runner_bid)
        producer_bs = c.get_bot_state(producer_bid)
        if not runner_bs or not producer_bs:
            return None

        runner_comp = self._bot_comp.get(runner_bid)
        producer_comp = self._bot_comp.get(producer_bid)
        if runner_comp is None or producer_comp is None:
            return None

        # Find bridge counters
        bridge = None
        chop_bridge = None
        for bc_pos, adj_by_comp in self._bridge_counters:
            r_adj = adj_by_comp.get(runner_comp, [])
            p_adj = adj_by_comp.get(producer_comp, [])
            if r_adj and p_adj:
                if bridge is None:
                    bridge = bc_pos
                elif chop_bridge is None:
                    chop_bridge = bc_pos
        if chop_bridge is None:
            chop_bridge = bridge
        if bridge is None:
            return None

        # Find resources in runner's component
        shop = None
        for sx, sy in self.tile_positions.get('SHOP', []):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = sx + dx, sy + dy
                    if (nx, ny) in self._tile_component and self._tile_component[(nx, ny)] == runner_comp:
                        shop = (sx, sy)
                        break
                if shop:
                    break
            if shop:
                break

        submit = None
        for sx, sy in self.tile_positions.get('SUBMIT', []):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = sx + dx, sy + dy
                    if (nx, ny) in self._tile_component and self._tile_component[(nx, ny)] == runner_comp:
                        submit = (sx, sy)
                        break
                if submit:
                    break
            if submit:
                break

        if not shop or not submit:
            return None

        # Find cooker in producer's component
        cooker = None
        for kx, ky in self.tile_positions.get('COOKER', []):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = kx + dx, ky + dy
                    if (nx, ny) in self._tile_component and self._tile_component[(nx, ny)] == producer_comp:
                        cooker = (kx, ky)
                        break
                if cooker:
                    break
            if cooker:
                break

        if not cooker:
            return None

        # Find assembly point
        shop_adj = self._get_adjacent_walkable(shop[0], shop[1])
        runner_shop_adj = [a for a in shop_adj if self._tile_component.get(a) == runner_comp]
        if not runner_shop_adj:
            return None
        assembly = min(runner_shop_adj, key=lambda a: (
            self._calculate_tile_distance(a[0], a[1], submit[0], submit[1])
            if a in self._path_distances else 9999
        ))

        # Classify ingredients
        ingredients = [INGREDIENT_REGISTRY[fn] for fn in order['required'] if fn in INGREDIENT_REGISTRY]
        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]
        all_cook = cook_chop + cook_only

        runner_steps: List[tuple] = []
        producer_steps: List[tuple] = []

        runner_steps.append(('goto', assembly))
        runner_steps.append(('buy', ShopCosts.PLATE, shop))
        runner_steps.append(('place', assembly))

        for ft in simple:
            runner_steps.append(('buy', ft, shop))
            runner_steps.append(('add_plate', assembly))

        for ft in chop_only:
            runner_steps.append(('buy', ft, shop))
            runner_steps.append(('place', chop_bridge))
            runner_steps.append(('chop', chop_bridge))
            runner_steps.append(('pickup', chop_bridge))
            runner_steps.append(('goto', assembly))
            runner_steps.append(('add_plate', assembly))

        for ft in all_cook:
            runner_steps.append(('buy', ft, shop))
            if ft.can_chop:
                runner_steps.append(('place', chop_bridge))
                runner_steps.append(('chop', chop_bridge))
                runner_steps.append(('pickup', chop_bridge))
            runner_steps.append(('place', bridge))
            runner_steps.append(('wait_pickup', bridge))
            runner_steps.append(('goto', assembly))
            runner_steps.append(('add_plate', assembly))

            producer_steps.append(('wait_pickup', bridge))
            producer_steps.append(('place_cook', cooker))
            producer_steps.append(('wait_take', cooker))
            producer_steps.append(('place', bridge))

        runner_steps.append(('pickup', assembly))
        runner_steps.append(('submit', submit))

        return (runner_steps, producer_steps)

    # ================================================================
    #  ABORT
    # ================================================================

    def _cancel_task(self, c: RobotController, bid: int, reason: str = "unknown"):
        """Abort current task and clean up any held/placed items."""
        self._stat_cancellations += 1
        t = self.bot_tasks.get(bid, {})
        oid = t.get('order_id')
        turn = c.get_turn()

        self._write_log(f"ABORT Bot {bid}: reason='{reason}' order_id={oid}")

        if oid:
            self.claimed_order_ids.discard(oid)
            self._total_task_cancellations += 1
            give_up_threshold = 8 if self._is_split_map else 4
            if self._total_task_cancellations >= give_up_threshold and self._stat_orders_done == 0:
                self._surrender_mode = True

        # Release cooker claims
        for ck_pos, owner in list(self._stove_reservations.items()):
            if owner == bid:
                del self._stove_reservations[ck_pos]

        # Clean up: if bot is holding something, create a trash recipe
        bs = c.get_bot_state(bid)
        if bs and bs.get('holding'):
            trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
            if trash:
                self._write_log(f"  Cleanup: Bot {bid} trashing held item")
                t.update({'recipe': [('trash', trash)], 'step': 0, 'order_id': None,
                          'assembly': None, 'chop': None, 'cooker': None,
                          'cooker2': None, 'is_future_order': False,
                          'is_split_order': False, 'partner_bid': None,
                          'stuck_count': 0, 'last_progress': turn})
            else:
                t.update({'recipe': [], 'step': 0, 'order_id': None,
                          'assembly': None, 'chop': None, 'cooker': None,
                          'cooker2': None, 'is_future_order': False,
                          'is_split_order': False, 'partner_bid': None,
                          'stuck_count': 0, 'last_progress': turn})
        else:
            partner_bid = t.get('partner_bid')
            t.update({'recipe': [], 'step': 0, 'order_id': None,
                      'assembly': None, 'chop': None, 'cooker': None,
                      'cooker2': None, 'is_future_order': False,
                      'is_split_order': False, 'partner_bid': None,
                      'stuck_count': 0, 'last_progress': turn})

            if partner_bid is not None and partner_bid in self.bot_tasks:
                pt = self.bot_tasks[partner_bid]
                if pt.get('partner_bid') == bid:
                    pt['partner_bid'] = None
                    self._cancel_task(c, partner_bid, reason=f"partner {bid} aborted")

    def _manage_burnt_food(self, c: RobotController, bid: int):
        """Handle burnt food."""
        t = self.bot_tasks[bid]
        oid = t.get('order_id')
        if oid:
            self.claimed_order_ids.discard(oid)
            self.order_cooldowns[oid] = c.get_turn()

        for ck_pos, owner in list(self._stove_reservations.items()):
            if owner == bid:
                del self._stove_reservations[ck_pos]

        bs = c.get_bot_state(bid)
        if bs:
            trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
            if trash:
                t['recipe'] = [('trash', trash)]
                t['step'] = 0
                t['order_id'] = None
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
                return
        t.update({'recipe': [], 'step': 0, 'order_id': None,
                  'assembly': None, 'chop': None, 'cooker': None,
                  'cooker2': None, 'is_future_order': False,
                  'is_split_order': False, 'partner_bid': None,
                  'stuck_count': 0, 'last_progress': c.get_turn()})

    # ================================================================
    #  STEP EXECUTOR
    # ================================================================

    def _run_task_step(self, c: RobotController, bid: int):
        """Execute current recipe step."""
        t = self.bot_tasks[bid]
        recipe = t['recipe']
        idx = t['step']
        turn = c.get_turn()

        if idx >= len(recipe):
            self._stat_idle_turns += 1
            return

        step = recipe[idx]
        action = step[0]
        if action == 'wait_take':
            self._stat_cooking_turns += 1
        used_move = False
        prev_step = idx
        team = c.get_team()

        # GOTO
        if action == 'goto':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if not bs:
                return
            bx, by = bs['x'], bs['y']
            if (bx, by) == (loc[0], loc[1]):
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = turn
                self._run_task_step(c, bid)
                return
            mv = self._step_to_tile(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
            elif mv is True:
                t['stuck_count'] = 0
                t['last_progress'] = turn
            elif mv is False:
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = turn
            return

        # BUY
        elif action == 'buy':
            item, loc = step[1], step[2]
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
                if trash:
                    t['recipe'] = [('trash', trash)] + recipe[idx:]
                    t['step'] = 0
                    t['last_progress'] = turn
                else:
                    self._cancel_task(c, bid, reason="holding during buy, no trash")
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            cost = getattr(item, 'buy_cost', 0)
            current_money = c.get_team_money(team)
            if current_money < cost:
                # Check if we're hopelessly stuck - if money is negative or very low, abort immediately
                # Don't wait 16 turns - this wastes time and leaves items stranded
                if current_money < 0 or t.get('stuck_count', 0) >= 3:
                    self._cancel_task(c, bid, reason=f"insufficient money for buy (have ${current_money}, need ${cost})")
                    return
                t['stuck_count'] += 1
                return
            if c.buy(bid, item, loc[0], loc[1]):
                t['step'] += 1

        # PLACE
        elif action == 'place':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if not (bs and bs.get('holding')):
                t['stuck_count'] += 1
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            if c.place(bid, loc[0], loc[1]):
                t['step'] += 1
            else:
                bs2 = c.get_bot_state(bid)
                if bs2:
                    alt = self._locate_free_counter(c, bs2['x'], bs2['y'], exclude={loc})
                    if alt:
                        self._reassign_task(t, idx, loc, alt)
                    else:
                        t['stuck_count'] += 1

        # PLACE_COOK
        elif action == 'place_cook':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if not (bs and bs.get('holding')):
                t['stuck_count'] += 1
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            tile = c.get_tile(team, loc[0], loc[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food is not None:
                    if pan.food.cooked_stage >= 2:
                        bs2 = c.get_bot_state(bid)
                        if bs2 and bs2.get('holding'):
                            temp = self._locate_free_counter(c, bs2['x'], bs2['y'])
                            trash = self._find_closest(bs2['x'], bs2['y'], 'TRASH')
                            if temp and trash:
                                cleanup = [('place', temp), ('take_pan', loc),
                                            ('trash', trash), ('pickup', temp)]
                                t['recipe'] = cleanup + recipe[idx:]
                                t['step'] = 0
                                t['last_progress'] = turn
                            else:
                                t['stuck_count'] += 1
                        else:
                            t['stuck_count'] += 1
                    else:
                        t['last_progress'] = turn
                    return
            if c.place(bid, loc[0], loc[1]):
                self._stove_reservations[loc] = bid
                t['step'] += 1
            else:
                t['stuck_count'] += 1

        # CHOP
        elif action == 'chop':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                t['stuck_count'] += 1
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            if c.chop(bid, loc[0], loc[1]):
                t['step'] += 1
            else:
                t['stuck_count'] += 1

        # PICKUP
        elif action == 'pickup':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
                if trash:
                    t['recipe'] = [('trash', trash)] + recipe[idx:]
                    t['step'] = 0
                    t['last_progress'] = turn
                else:
                    self._cancel_task(c, bid, reason="holding during pickup, no trash")
                return
            
            # Check if target has an item to pickup
            # If this is a cleanup task and the item is gone, clear the task
            tile = c.get_tile(team, loc[0], loc[1])
            has_item = tile and getattr(tile, 'item', None) is not None
            if not has_item and t.get('order_id') is None:
                # Cleanup task but item is gone - clear the task
                t['recipe'] = []
                t['step'] = 0
                return
            
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            if c.pickup(bid, loc[0], loc[1]):
                t['step'] += 1
            else:
                t['stuck_count'] += 1

        # WAIT_TAKE
        elif action == 'wait_take':
            loc = step[1]
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                self._cancel_task(c, bid, reason="holding during wait_take")
                return
            tile = c.get_tile(team, loc[0], loc[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food:
                    claim_owner = self._stove_reservations.get(loc)
                    if claim_owner is not None and claim_owner != bid:
                        t['last_progress'] = turn
                        return
                    if pan.food.cooked_stage == 1:
                        if c.take_from_pan(bid, loc[0], loc[1]):
                            self._stove_reservations.pop(loc, None)
                            t['step'] += 1
                    elif pan.food.cooked_stage >= 2:
                        if c.take_from_pan(bid, loc[0], loc[1]):
                            self._stove_reservations.pop(loc, None)
                            self._manage_burnt_food(c, bid)
                            return
                else:
                    t['stuck_count'] += 1
            else:
                t['stuck_count'] += 1

        # WAIT_PICKUP
        elif action == 'wait_pickup':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                t['stuck_count'] += 1
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            tile = c.get_tile(team, loc[0], loc[1])
            if tile and getattr(tile, 'item', None) is not None:
                if c.pickup(bid, loc[0], loc[1]):
                    t['step'] += 1
                else:
                    t['stuck_count'] += 1
            else:
                t['last_progress'] = turn

        # TAKE_PAN
        elif action == 'take_pan':
            loc = step[1]
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                self._cancel_task(c, bid, reason="holding during take_pan")
                return
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            tile = c.get_tile(team, loc[0], loc[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                if tile.item.food is None:
                    t['step'] += 1
                    t['last_progress'] = turn
                    return
            if c.take_from_pan(bid, loc[0], loc[1]):
                self._stove_reservations.pop(loc, None)
                t['step'] += 1
            else:
                t['stuck_count'] += 1

        # ADD_PLATE
        elif action == 'add_plate':
            loc = step[1]
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            if c.add_food_to_plate(bid, loc[0], loc[1]):
                t['step'] += 1
            else:
                t['stuck_count'] += 1

        # SUBMIT
        elif action == 'submit':
            loc = step[1]
            oid = t.get('order_id')
            if t.get('is_future_order'):
                order_active = False
                for o in c.get_orders(team):
                    if o['order_id'] == oid and o['is_active']:
                        order_active = True
                        break
                if not order_active:
                    return
                t['is_future_order'] = False
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            if c.submit(bid, loc[0], loc[1]):
                self._stat_orders_done += 1
                self._write_log(f"COMPLETED Order #{oid}")
                if oid:
                    self.finished_order_ids.add(oid)
                    self.claimed_order_ids.discard(oid)
                self._counter_failures = 0
                self._stove_failures = 0
                self._chopping_disabled = False
                self._cooking_disabled = False
                self._total_task_cancellations = 0
                t.update({'recipe': [], 'step': 0, 'order_id': None,
                          'assembly': None, 'chop': None, 'cooker': None,
                          'cooker2': None, 'is_future_order': False,
                          'is_split_order': False, 'partner_bid': None,
                          'stuck_count': 0, 'last_progress': turn})
                return
            else:
                t['stuck_count'] += 1

        # TRASH
        elif action == 'trash':
            loc = step[1]
            mv = self._navigate_toward(c, bid, loc[0], loc[1])
            if mv is None:
                t['stuck_count'] += 1
                return
            used_move = (mv is True)
            bs = c.get_bot_state(bid)
            if not (bs and bs.get('holding')):
                t['step'] += 1
                return
            # Check if we're holding an empty plate (can't be trashed)
            holding = bs.get('holding')
            if hasattr(holding, 'contents') and len(holding.contents) == 0:
                # Empty plate - place it on a counter instead of trashing
                alt = self._locate_free_counter(c, bs['x'], bs['y'])
                if not alt:
                    alt = self._locate_empty_tile(c, bs['x'], bs['y'])
                if alt:
                    t['recipe'] = [('place', alt)] + recipe[idx + 1:]
                    t['step'] = 0
                    t['last_progress'] = turn
                    return
            if c.trash(bid, loc[0], loc[1]):
                t['step'] += 1
                bs2 = c.get_bot_state(bid)
                if bs2 and bs2.get('holding'):
                    trash_count = t.get('_trash_count', 0) + 1
                    if trash_count <= 2:
                        t['_trash_count'] = trash_count
                        t['recipe'] = [('trash', loc)] + recipe[idx + 1:]
                        t['step'] = 0
                        t['last_progress'] = turn
                        return
                    else:
                        alt = self._locate_free_counter(c, bs2['x'], bs2['y'])
                        if not alt:
                            alt = self._locate_empty_tile(c, bs2['x'], bs2['y'])
                        if alt:
                            t['recipe'] = [('place', alt)] + recipe[idx + 1:]
                            t['step'] = 0
                            t['_trash_count'] = 0
                            t['last_progress'] = turn
                            return
                        t['_trash_count'] = 0
                        self._cancel_task(c, bid, reason="trash loop")
                        return
                t['_trash_count'] = 0
                if t['step'] >= len(recipe):
                    t.update({'recipe': [], 'step': 0, 'order_id': None,
                              'assembly': None, 'chop': None, 'cooker': None,
                              'cooker2': None, 'is_future_order': False,
                              'is_split_order': False, 'partner_bid': None,
                              'stuck_count': 0, 'last_progress': turn})
                    return
            else:
                t['stuck_count'] += 1

        # Reset stuck counter on progress
        if t['step'] != prev_step:
            t['stuck_count'] = 0
            t['last_progress'] = turn
            if t['step'] < len(t['recipe']):
                self._attempt_chained_action(c, bid, used_move)
                return

        if not used_move and t['step'] < len(t['recipe']):
            self._execute_early_movement(c, bid)

    def _attempt_chained_action(self, c: RobotController, bid: int, prev_used_move: bool):
        """After completing a step, try to execute next step in same turn."""
        t = self.bot_tasks[bid]
        recipe = t['recipe']
        idx = t['step']
        if idx >= len(recipe):
            return

        step = recipe[idx]
        action = step[0]
        loc = self._get_action_target(step)
        if loc is None:
            if not prev_used_move:
                self._execute_early_movement(c, bid)
            return

        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']

        if action == 'goto':
            if (bx, by) == loc or max(abs(bx - loc[0]), abs(by - loc[1])) <= 1:
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
                if t['step'] < len(t['recipe']):
                    self._attempt_chained_action(c, bid, prev_used_move)
                return
            if not prev_used_move:
                self._execute_early_movement(c, bid)
            return

        if max(abs(bx - loc[0]), abs(by - loc[1])) > 1:
            if not prev_used_move:
                self._execute_early_movement(c, bid)
            return

        team = c.get_team()
        if action == 'buy':
            item = step[1]
            if bs.get('holding'):
                return
            cost = getattr(item, 'buy_cost', 0)
            if c.get_team_money(team) < cost:
                return
            if c.buy(bid, item, loc[0], loc[1]):
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
        elif action == 'place':
            if not bs.get('holding'):
                return
            if c.place(bid, loc[0], loc[1]):
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
        elif action == 'add_plate':
            if c.add_food_to_plate(bid, loc[0], loc[1]):
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
        elif action == 'pickup':
            if c.pickup(bid, loc[0], loc[1]):
                t['step'] += 1
                t['stuck_count'] = 0
                t['last_progress'] = c.get_turn()
        elif action == 'place_cook':
            if bs.get('holding'):
                if c.place(bid, loc[0], loc[1]):
                    self._stove_reservations[loc] = bid
                    t['step'] += 1
                    t['stuck_count'] = 0
                    t['last_progress'] = c.get_turn()
        elif action == 'chop':
            if not bs.get('holding'):
                if c.chop(bid, loc[0], loc[1]):
                    t['step'] += 1
                    t['stuck_count'] = 0
                    t['last_progress'] = c.get_turn()
        elif action == 'trash':
            if bs.get('holding'):
                if c.trash(bid, loc[0], loc[1]):
                    t['step'] += 1
                    t['stuck_count'] = 0
                    t['last_progress'] = c.get_turn()
        elif action == 'take_pan':
            if not bs.get('holding'):
                if c.take_from_pan(bid, loc[0], loc[1]):
                    self._stove_reservations.pop(loc, None)
                    t['step'] += 1
                    t['stuck_count'] = 0
                    t['last_progress'] = c.get_turn()
        elif action == 'wait_pickup':
            if not bs.get('holding'):
                tile = c.get_tile(team, loc[0], loc[1])
                if tile and getattr(tile, 'item', None) is not None:
                    if c.pickup(bid, loc[0], loc[1]):
                        t['step'] += 1
                        t['stuck_count'] = 0
                        t['last_progress'] = c.get_turn()
        elif action == 'submit':
            oid = t.get('order_id')
            if t.get('is_future_order'):
                order_active = False
                for o in c.get_orders(team):
                    if o['order_id'] == oid and o['is_active']:
                        order_active = True
                        break
                if not order_active:
                    return
                t['is_future_order'] = False
            if c.submit(bid, loc[0], loc[1]):
                self._stat_orders_done += 1
                if oid:
                    self.finished_order_ids.add(oid)
                    self.claimed_order_ids.discard(oid)
                self._counter_failures = 0
                self._stove_failures = 0
                self._chopping_disabled = False
                self._cooking_disabled = False
                self._total_task_cancellations = 0
                t.update({'recipe': [], 'step': 0, 'order_id': None,
                          'assembly': None, 'chop': None, 'cooker': None,
                          'cooker2': None, 'is_future_order': False,
                          'is_split_order': False, 'partner_bid': None,
                          'stuck_count': 0, 'last_progress': c.get_turn()})

    # ================================================================
    #  HELPERS
    # ================================================================

    def _locate_free_counter(self, c: RobotController, bx: int, by: int,
                            exclude: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        if exclude is None:
            exclude = set()
        best_d = 9999
        best = None
        team = c.get_team()
        for cx, cy in self._counter_tiles:
            if (cx, cy) in exclude:
                continue
            tile = c.get_tile(team, cx, cy)
            if tile and getattr(tile, 'item', None) is None:
                d = self._calculate_tile_distance(bx, by, cx, cy)
                if d < best_d:
                    best_d = d
                    best = (cx, cy)
        return best

    def _locate_empty_tile(self, c: RobotController, bx: int, by: int) -> Optional[Tuple[int, int]]:
        """Find any empty walkable tile to drop an item on (last resort)."""
        team = c.get_team()
        src = (bx, by)
        if src not in self._path_distances:
            return None
        d = self._path_distances[src]
        candidates = sorted(self.passable_cells, key=lambda p: d.get(p, 9999))
        for wx, wy in candidates[:50]:
            if d.get((wx, wy), 9999) >= 9999:
                break
            tile = c.get_tile(team, wx, wy)
            if tile and hasattr(tile, 'item') and tile.item is None:
                return (wx, wy)
        return None

    def _reassign_task(self, task: Dict, from_idx: int,
               old: Tuple[int, int], new: Tuple[int, int]):
        for i in range(from_idx, len(task['recipe'])):
            s = task['recipe'][i]
            if len(s) >= 2 and s[1] == old:
                task['recipe'][i] = (s[0], new) + s[2:]

    def _navigate_to_disposal(self, c: RobotController, bid: int):
        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']
        targets = []
        for tx, ty in self._disposal_tiles:
            targets.extend(self._get_adjacent_walkable(tx, ty))
        team = c.get_team()
        for tx, ty in self._counter_tiles:
            tile = c.get_tile(team, tx, ty)
            if tile and getattr(tile, 'item', None) is None:
                targets.extend(self._get_adjacent_walkable(tx, ty))
        if not targets:
            return
        src = (bx, by)
        if src not in self._path_distances:
            return
        d = self._path_distances[src]
        best_t = min(targets, key=lambda a: d.get(a, 9999))
        if best_t not in d:
            return
        ns = self._first_moves[src]
        if best_t in ns:
            dx, dy = ns[best_t]
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if c.can_move(bid, dx, dy):
                    c.move(bid, dx, dy)
                    return

    # ================================================================
    #  SABOTAGE PLANNING AND EXECUTION
    # ================================================================

    def _plan_sabotage(self, c: RobotController, orders: List[Dict]) -> None:
        """Plan sabotage timing.
        
        IMPORTANT: Sabotage is risky because when WE switch:
        - Our bots go to enemy map (can't complete our orders)
        - Enemy bots stay on THEIR map (continue completing their orders)
        
        So we DON'T both miss orders - only we do. Sabotage only helps if:
        - Items we trash cost more than orders we miss (rare)
        - Or there are very few/no orders during the window
        """
        if self._switch_strategy_set:
            return
            
        switch_info = c.get_switch_info()
        switch_turn = switch_info['switch_turn']
        switch_end = switch_info['window_end_turn']
        turn = c.get_turn()
        
        if turn >= switch_turn:
            self._switch_strategy_set = True
            return
        
        # Count orders active during sabotage window
        orders_during_switch = 0
        total_order_value = 0
        for o in orders:
            if o.get('completed_turn') is not None:
                continue
            starts = o.get('created_turn', 0)
            expires = o.get('expires_turn', 0)
            # Orders that are active during the switch window
            if starts <= switch_end and expires >= switch_turn:
                orders_during_switch += 1
                total_order_value += o.get('reward', 0)
        
        # Disable sabotage if there are orders we could complete
        # Each order we could complete during sabotage is worth ~$50-200
        # Each item we trash is worth ~$5-40
        # So sabotage is rarely worth it
        if orders_during_switch >= 1:
            self._planned_switch_turn = None
            self._switch_strategy_set = True
            self._write_log(f"SABOTAGE DISABLED: {orders_during_switch} orders during window worth ~${total_order_value}")
            return
        
        # Only sabotage if there are NO orders during the window
        # (then we're not losing anything by switching)
        best_switch_turn = switch_turn + 15
        if best_switch_turn >= switch_end:
            best_switch_turn = switch_turn
        
        self._planned_switch_turn = best_switch_turn
        self._order_deadline_cutoff = switch_turn - 10
        
        # Identify a good order to prep for when we return
        for o in orders:
            if o.get('completed_turn') is not None:
                continue
            starts = o.get('created_turn', 0)
            expires = o.get('expires_turn', 0)
            # Order that starts after switch window ends
            if starts >= switch_end - 10:
                self._post_switch_target = o.get('order_id')
                # Track what items we need for this order
                for item in o.get('required', []):
                    self._items_to_preserve.add(item)
                break
        
        self._switch_strategy_set = True
        self._write_log(f"SABOTAGE ENABLED: Switch at turn {best_switch_turn}, window ends {switch_end}")
        self._write_log(f"  Prep order: #{self._post_switch_target}, useful items: {self._items_to_preserve}")

    def _should_switch_now(self, c: RobotController) -> bool:
        """Determine if we should switch maps now."""
        switch_info = c.get_switch_info()
        
        if not switch_info['window_active']:
            return False
        if switch_info['my_team_switched']:
            return False
        
        # If sabotage is disabled, don't switch
        if self._planned_switch_turn is None:
            return False
            
        turn = c.get_turn()
        
        # If we have a planned switch turn, use that
        return turn >= self._planned_switch_turn

    def _execute_defense_protocol(self, c: RobotController, bots: List[int], 
                                   team: Team, turn: int) -> None:
        """Protect valuable items when enemy has switched to our map.
        
        Strategy: Find idle bots and have them pick up any valuable items
        on counters/cookers that enemy bots might try to trash.
        """
        # Find items on our map that need protection
        vulnerable_items = []
        
        # Check counters for items
        for counter_pos in self._counter_tiles:
            tile = c.get_tile(team, counter_pos[0], counter_pos[1])
            if tile and hasattr(tile, 'item') and tile.item is not None:
                item = tile.item
                value = 0
                if isinstance(item, Plate):
                    value = 2 + len(getattr(item, 'food', []) or []) * 10
                elif isinstance(item, Food):
                    value = getattr(item, 'buy_cost', 5)
                if value > 0:
                    vulnerable_items.append((counter_pos, value, 'counter'))
        
        # Check cookers for cooked food
        for stove_pos in self._stove_tiles:
            tile = c.get_tile(team, stove_pos[0], stove_pos[1])
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                pan = tile.item
                if pan.food is not None and pan.progress >= GameConstants.COOK_PROGRESS:
                    value = pan.food.buy_cost + 20  # Cooked food is valuable
                    vulnerable_items.append((stove_pos, value, 'cooker'))
        
        if not vulnerable_items:
            return
        
        # Sort by value (protect most valuable first)
        vulnerable_items.sort(key=lambda x: -x[1])
        
        # Find idle bots that can help protect
        for bid in bots:
            t = self.bot_tasks[bid]
            bs = c.get_bot_state(bid)
            if not bs:
                continue
            
            # Skip bots that are busy or holding something
            if bs.get('holding'):
                continue
            if t.get('order_id') is not None and t.get('recipe') and t['step'] < len(t['recipe']):
                # Bot is working on an order - only interrupt if item is very valuable
                pass  # For now, don't interrupt
            
            # If bot has no task, assign protection duty
            if not t.get('recipe') or t['step'] >= len(t.get('recipe', [])):
                for item_pos, value, item_type in vulnerable_items:
                    if item_pos in self._protected_items:
                        continue
                    
                    # Create a recipe to pick up and hold the item
                    if item_type == 'cooker':
                        recipe = [('take_pan', item_pos)]
                    else:
                        recipe = [('pickup', item_pos)]
                    
                    t['recipe'] = recipe
                    t['step'] = 0
                    t['order_id'] = None  # No order, just protection
                    t['stuck_count'] = 0
                    t['last_progress'] = turn
                    self._protected_items.add(item_pos)
                    self._write_log(f"DEFENSE: Bot {bid} protecting item at {item_pos} (value={value})")
                    break

    def _execute_sabotage_actions(self, c: RobotController, bots: List[int]) -> bool:
        """Execute sabotage actions while on enemy map. Returns True if we're in sabotage mode."""
        switch_info = c.get_switch_info()
        turn = c.get_turn()
        team = c.get_team()
        
        # Check if we're on enemy map
        if not switch_info['my_team_switched']:
            self._switch_in_progress = False
            return False
            
        self._switch_in_progress = True
        self._write_log(f"SABOTAGE MODE: Turn {turn}")
        
        enemy_team = c.get_enemy_team()
        enemy_map = c.get_map(enemy_team)
        map_w = enemy_map.width
        map_h = enemy_map.height
        
        # Find all trash locations on enemy map
        enemy_trash_locs = []
        for x in range(map_w):
            for y in range(map_h):
                tile = c.get_tile(enemy_team, x, y)
                if tile and tile.tile_name == 'TRASH':
                    enemy_trash_locs.append((x, y))
        
        # Find all items on enemy map (counters and cookers with items)
        enemy_items = []
        for x in range(map_w):
            for y in range(map_h):
                tile = c.get_tile(enemy_team, x, y)
                if tile and hasattr(tile, 'item') and tile.item is not None:
                    enemy_items.append((x, y, tile.item))
        
        self._write_log(f"  Enemy items found: {len(enemy_items)} at {[(x,y) for x,y,_ in enemy_items[:5]]}")
        
        for bid in bots:
            bs = c.get_bot_state(bid)
            if not bs:
                continue
                
            bx, by = bs['x'], bs['y']
            holding = bs.get('holding')
            
            # Find nearest trash on enemy map
            nearest_trash = None
            min_dist = float('inf')
            for tx, ty in enemy_trash_locs:
                d = abs(bx - tx) + abs(by - ty)
                if d < min_dist:
                    min_dist = d
                    nearest_trash = (tx, ty)
            
            if holding:
                # We're holding something - go trash it
                if nearest_trash:
                    tx, ty = nearest_trash
                    # Check if adjacent to trash
                    if abs(bx - tx) <= 1 and abs(by - ty) <= 1:
                        if c.trash(bid, tx, ty):
                            self._write_log(f"  Bot {bid}: TRASHED item at ({tx},{ty})")
                        else:
                            self._write_log(f"  Bot {bid}: trash() failed at ({tx},{ty})")
                    else:
                        # Move toward trash
                        mv = self._navigate_toward(c, bid, tx, ty)
                        if mv is None:
                            self._random_move(c, bid)
                        self._write_log(f"  Bot {bid}: Moving to trash at ({tx},{ty}), holding item")
                continue
            
            # Not holding anything - find items to pick up
            # Find nearest enemy item
            nearest_item = None
            min_dist = float('inf')
            for ix, iy, item in enemy_items:
                d = abs(bx - ix) + abs(by - iy)
                if d < min_dist:
                    min_dist = d
                    nearest_item = (ix, iy)
            
            if nearest_item:
                ix, iy = nearest_item
                # Check if adjacent to item
                if abs(bx - ix) <= 1 and abs(by - iy) <= 1:
                    if c.pickup(bid, ix, iy):
                        self._write_log(f"  Bot {bid}: PICKED UP enemy item at ({ix},{iy})")
                    else:
                        # Might be a cooker - try take_from_pan
                        if c.take_from_pan(bid, ix, iy):
                            self._write_log(f"  Bot {bid}: TOOK FROM PAN at ({ix},{iy})")
                        else:
                            self._write_log(f"  Bot {bid}: pickup/take failed at ({ix},{iy})")
                            self._random_move(c, bid)
                else:
                    # Move toward item
                    mv = self._navigate_toward(c, bid, ix, iy)
                    if mv is None:
                        self._random_move(c, bid)
                    self._write_log(f"  Bot {bid}: Moving to item at ({ix},{iy})")
            else:
                # No items found, explore randomly
                self._random_move(c, bid)
                self._write_log(f"  Bot {bid}: No items found, exploring")
                
        return True

    def _random_move(self, c: RobotController, bid: int) -> None:
        """Make a random valid move."""
        import random
        moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
        random.shuffle(moves)
        for dx, dy in moves:
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return

    # ================================================================
    #  MAIN TURN EXECUTION
    # ================================================================

    def play_turn(self, c: RobotController):
        """Main entry point."""
        turn = c.get_turn()
        team = c.get_team()
        if self.my_team is None:
            self.my_team = team

        # Initialize log
        if turn == 1:
            self._start_telemetry()

        bots = c.get_team_bot_ids(team)
        orders = c.get_orders(team)
        money = c.get_team_money(team)

        # Analyze order timing characteristics (once)
        if not self._order_analysis_complete:
            self._analyze_late_orders(orders)
            self._order_analysis_complete = True

        # Sabotage prevention: detect if enemy has switched to our map
        switch_info = c.get_switch_info()
        if switch_info['enemy_team_switched'] and not self._enemy_switched:
            self._enemy_switched = True
            self._defense_mode = True
            self._write_log(f"DEFENSE MODE ACTIVATED: Enemy switched to our map at turn {turn}")
        
        # Defense mode ends when switch window ends
        if self._defense_mode and not switch_info['window_active']:
            self._defense_mode = False
            self._enemy_switched = False
            self._write_log(f"DEFENSE MODE DEACTIVATED: Switch window ended at turn {turn}")

        for bid in bots:
            if bid not in self.bot_tasks:
                self.bot_tasks[bid] = {
                    'recipe': [], 'step': 0, 'order_id': None,
                    'assembly': None, 'chop': None, 'cooker': None, 'cooker2': None,
                    'stuck_count': 0, 'last_progress': turn,
                    'is_future_order': False,
                    'is_split_order': False, 'partner_bid': None,
                }

        # Split map initialization
        if self._is_split_map and not self._split_initialized:
            self._split_initialized = True
            for bid in bots:
                bs = c.get_bot_state(bid)
                if bs:
                    pos = (bs['x'], bs['y'])
                    comp = self._tile_component.get(pos)
                    if comp is not None:
                        self._bot_comp[bid] = comp
                        self._bot_role[bid] = self._comp_role.get(comp, 'runner')

            # Check if all bots are in the same component - if so, disable split map mode
            # This handles maps where the walkable area is technically disconnected but
            # all bots spawn in one region
            bot_comps = set(self._bot_comp.values())
            if len(bot_comps) <= 1:
                self._is_split_map = False
                self._write_log(f"Disabled split map mode: all bots in component {bot_comps}")

            # Also check if we have both runners and producers
            roles = set(self._bot_role.values())
            if 'runner' not in roles or 'producer' not in roles:
                self._is_split_map = False
                self._write_log(f"Disabled split map mode: missing roles (have {roles})")

        # Log state every turn for detailed debugging
        self._record_game_state(c, turn, verbose=(turn <= 10 or turn % 10 == 0))

        # ============================================
        # SABOTAGE LOGIC
        # ============================================
        
        # Plan sabotage timing
        self._plan_sabotage(c, orders)
        
        # Check if we should switch to enemy map
        if self._should_switch_now(c):
            if c.switch_maps():
                self._write_log(f"SWITCHED TO ENEMY MAP at turn {turn}!")
                self._switch_in_progress = True
        
        # If we're in sabotage mode, execute sabotage actions and skip normal processing
        if self._execute_sabotage_actions(c, bots):
            return  # Don't do normal order processing while sabotaging
        
        # ============================================
        # NORMAL ORDER PROCESSING
        # ============================================

        # Expire stale assignments
        active_ids = {o['order_id'] for o in orders if o['is_active']}
        completed_ids = {o['order_id'] for o in orders if o.get('completed_turn') is not None}
        future_ids = {o['order_id'] for o in orders if not o['is_active'] and o['created_turn'] > turn}
        for bid in bots:
            t = self.bot_tasks[bid]
            oid = t.get('order_id')
            if oid and oid not in active_ids and oid not in completed_ids:
                if t.get('is_future_order') and oid in future_ids:
                    continue
                self._cancel_task(c, bid, reason=f"order {oid} no longer active")

        # Update position tracking
        for bid in bots:
            bs = c.get_bot_state(bid)
            if bs:
                pos = (bs['x'], bs['y'])
                old_pos = self._previous_positions.get(bid)
                self._previous_positions[bid] = pos
                t = self.bot_tasks[bid]
                if old_pos and old_pos != pos and t.get('recipe'):
                    t['last_progress'] = turn
                    t['stuck_count'] = 0

        # Detect stuck bots
        for bid in bots:
            t = self.bot_tasks[bid]
            if not t['recipe'] or t['step'] >= len(t['recipe']):
                continue
            if t.get('is_future_order'):
                step_idx = t.get('step', 0)
                recipe = t.get('recipe', [])
                if step_idx < len(recipe) and recipe[step_idx][0] == 'submit':
                    continue
            sc = t.get('stuck_count', 0)
            no_progress = turn - t.get('last_progress', turn)
            if t.get('is_split_order'):
                step_idx_s = t.get('step', 0)
                recipe_s = t.get('recipe', [])
                if step_idx_s < len(recipe_s) and recipe_s[step_idx_s][0] == 'wait_pickup':
                    if sc > 60 or no_progress > 100:
                        pass
                    else:
                        continue
            if sc > 14 or no_progress > 33:
                oid = t.get('order_id')
                if oid:
                    self.order_cooldowns[oid] = turn
                    step_idx = t.get('step', 0)
                    recipe = t.get('recipe', [])
                    if step_idx < len(recipe):
                        stuck_action = recipe[step_idx][0]
                        if stuck_action in ('place', 'chop') and t.get('chop'):
                            self._counter_failures += 1
                            if self._counter_failures >= 3:
                                self._chopping_disabled = True
                        elif stuck_action in ('place_cook', 'wait_take') and t.get('cooker'):
                            self._stove_failures += 1
                            if self._stove_failures >= 3:
                                self._cooking_disabled = True
                self._cancel_task(c, bid, reason=f"stuck sc={sc} no_progress={no_progress}")

        # Handle idle bots holding items
        idle_bots = []
        for bid in bots:
            t = self.bot_tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
                if trash:
                    t['recipe'] = [('trash', trash)]
                    t['step'] = 0
                    t['order_id'] = None
                    t['stuck_count'] = 0
                    t['last_progress'] = turn
                    continue
                alt = self._locate_free_counter(c, bs['x'], bs['y'])
                if not alt:
                    alt = self._locate_empty_tile(c, bs['x'], bs['y'])
                if alt:
                    t['recipe'] = [('place', alt)]
                    t['step'] = 0
                    t['order_id'] = None
                    t['stuck_count'] = 0
                    t['last_progress'] = turn
                    continue
                self._navigate_to_disposal(c, bid)
                continue
            idle_bots.append(bid)
        
        # Clean up abandoned items on counters/cookers that are blocking resources
        # Only do this if we have idle bots and resources are blocked
        if idle_bots:
            # Check if counter has an item that's blocking (not claimed by any active order)
            for cc in self._counter_tiles:
                tile = c.get_tile(team, cc[0], cc[1])
                if tile and getattr(tile, 'item', None) is not None:
                    # Check if any bot is using this counter
                    counter_in_use = False
                    for obid, ot in self.bot_tasks.items():
                        if ot.get('chop') == cc and ot.get('order_id') is not None:
                            counter_in_use = True
                            break
                    if not counter_in_use:
                        # This counter has an abandoned item - assign cleanup to an idle bot
                        for bid in idle_bots:
                            bs = c.get_bot_state(bid)
                            if not bs or bs.get('holding'):
                                continue
                            trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
                            if trash:
                                self._write_log(f"  Cleanup: Bot {bid} clearing abandoned item from counter {cc}")
                                t = self.bot_tasks[bid]
                                t['recipe'] = [('pickup', cc), ('trash', trash)]
                                t['step'] = 0
                                t['order_id'] = None
                                t['stuck_count'] = 0
                                t['last_progress'] = turn
                                idle_bots.remove(bid)
                                break
                        break  # Only clean one item per turn
            
            # Check if cooker has food that's abandoned (cooked but not claimed, or burnt)
            for kk in self._stove_tiles:
                tile = c.get_tile(team, kk[0], kk[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food is not None:
                        # Check if any bot is using this cooker
                        cooker_in_use = self._stove_reservations.get(kk) is not None
                        if not cooker_in_use:
                            for obid, ot in self.bot_tasks.items():
                                if ot.get('cooker') == kk and ot.get('order_id') is not None:
                                    cooker_in_use = True
                                    break
                        if not cooker_in_use:
                            # Abandoned food in cooker - clean it up
                            for bid in idle_bots:
                                bs = c.get_bot_state(bid)
                                if not bs or bs.get('holding'):
                                    continue
                                trash = self._find_closest(bs['x'], bs['y'], 'TRASH')
                                if trash:
                                    self._write_log(f"  Cleanup: Bot {bid} clearing abandoned food from cooker {kk}")
                                    t = self.bot_tasks[bid]
                                    t['recipe'] = [('take_pan', kk), ('trash', trash)]
                                    t['step'] = 0
                                    t['order_id'] = None
                                    t['stuck_count'] = 0
                                    t['last_progress'] = turn
                                    idle_bots.remove(bid)
                                    break
                            break  # Only clean one item per turn

        if self._surrender_mode:
            for bid in bots:
                self._run_task_step(c, bid)
            return

        # Count resource usage
        n_cooking_bots = sum(1 for bid in bots
                             if self.bot_tasks[bid].get('cooker') is not None
                             and self.bot_tasks[bid].get('order_id') is not None)
        n_chopping_bots = sum(1 for bid in bots
                              if self.bot_tasks[bid].get('chop') is not None
                              and self.bot_tasks[bid].get('order_id') is not None)

        # Calculate available money
        committed_spending = 0
        for bid in bots:
            t = self.bot_tasks[bid]
            if t.get('recipe') and t.get('order_id') is not None:
                recipe = t.get('recipe', [])
                step = t.get('step', 0)
                for i in range(step, len(recipe)):
                    action = recipe[i]
                    if action[0] == 'buy':
                        item = action[1]
                        cost = getattr(item, 'buy_cost', 0)
                        committed_spending += cost
        available_money = money - committed_spending

        # Split map handling
        if self._is_split_map:
            self._allocate_split_tasks(c, bots, idle_bots, orders, turn, money)
            for bid in bots:
                self._run_task_step(c, bid)
            # Idle movement for split map
            for bid in bots:
                t = self.bot_tasks[bid]
                if t['recipe'] and t['step'] < len(t['recipe']):
                    continue
                bs = c.get_bot_state(bid)
                if not bs or bs.get('holding'):
                    continue
                role = self._bot_role.get(bid, 'runner')
                bx, by = bs['x'], bs['y']
                if role == 'runner':
                    shop = self._find_closest(bx, by, 'SHOP')
                    if shop:
                        shop_adj = self._get_adjacent_walkable(shop[0], shop[1])
                        comp = self._bot_comp.get(bid)
                        shop_adj = [a for a in shop_adj if self._tile_component.get(a) == comp]
                        if shop_adj and (bx, by) not in shop_adj:
                            target = min(shop_adj, key=lambda a: self._path_distances.get((bx, by), {}).get(a, 9999))
                            if target in self._path_distances.get((bx, by), {}):
                                self._step_to_tile(c, bid, target[0], target[1])
                elif role == 'producer' and self._bridge_counters:
                    bc_pos = self._bridge_counters[0][0]
                    bc_adj = self._get_adjacent_walkable(bc_pos[0], bc_pos[1])
                    comp = self._bot_comp.get(bid)
                    bc_adj = [a for a in bc_adj if self._tile_component.get(a) == comp]
                    if bc_adj and (bx, by) not in bc_adj:
                        target = min(bc_adj, key=lambda a: self._path_distances.get((bx, by), {}).get(a, 9999))
                        if target in self._path_distances.get((bx, by), {}):
                            self._step_to_tile(c, bid, target[0], target[1])
            return

        # Normal map order assignment
        candidates = self._prioritize_active_orders(orders, turn, c)
        assignments = []
        newly_committed = 0

        # Log available orders on first turn or when idle bots exist
        if turn == 1 or idle_bots:
            self._write_log(f"Available active orders: {len(candidates)}")
            for ci, cand in enumerate(candidates[:5]):
                o = cand['order']
                self._write_log(f"  Order #{o['order_id']}: {o['required']} profit={cand['profit']} tleft={cand['tleft']} needs_cook={cand['needs_cook']} needs_chop={cand['needs_chop']}")

        for bid in idle_bots:
            bot_state = c.get_bot_state(bid)
            if not bot_state:
                continue
            bx, by = bot_state['x'], bot_state['y']
            self._write_log(f"  Bot {bid} at ({bx},{by}) evaluating orders...")
            for ci, cand in enumerate(candidates):
                oid = cand['order']['order_id']
                if oid in self.claimed_order_ids:
                    continue
                # Check if enough money remains after committed spending
                if available_money - newly_committed < cand['cost'] + 2:
                    self._write_log(f"    Skip #{oid}: insufficient money (need {cand['cost']+2}, have {available_money - newly_committed})")
                    continue
                if cand['needs_cook'] and n_cooking_bots >= max(self._stove_count * 2, 2):
                    self._write_log(f"    Skip #{oid}: too many cooking bots ({n_cooking_bots} >= {max(self._stove_count * 2, 2)})")
                    continue
                if cand['needs_chop'] and n_chopping_bots >= max(self._counter_count * 2, 2):
                    self._write_log(f"    Skip #{oid}: too many chopping bots ({n_chopping_bots} >= {max(self._counter_count * 2, 2)})")
                    continue
                if self._chopping_disabled and cand['needs_chop']:
                    self._write_log(f"    Skip #{oid}: skip_chop flag set")
                    continue
                if self._cooking_disabled and cand['needs_cook']:
                    self._write_log(f"    Skip #{oid}: skip_cook flag set")
                    continue
                est = self._predict_completion_time(bx, by, cand['order'])
                if est > cand['tleft'] * self._feasibility_mult:
                    self._write_log(f"    Skip #{oid}: est={est} > tleft*mult={cand['tleft']}*{self._feasibility_mult:.1f}={cand['tleft']*self._feasibility_mult:.0f}")
                    continue
                if turn + est > GameConstants.TOTAL_TURNS:
                    self._write_log(f"    Skip #{oid}: would end after game (turn {turn} + est {est} > {GameConstants.TOTAL_TURNS})")
                    continue
                eff = cand['profit'] / max(est, 1)
                self._write_log(f"    Consider #{oid}: est={est} eff={eff:.1f}")
                assignments.append((eff, bid, ci))

        assignments.sort(key=lambda x: -x[0])
        assigned_bots = set()
        assigned_oids = set()

        for eff, bid, ci in assignments:
            if bid in assigned_bots:
                continue
            cand = candidates[ci]
            oid = cand['order']['order_id']
            if oid in self.claimed_order_ids or oid in assigned_oids:
                continue
            if available_money - newly_committed < cand['cost'] + 2:
                continue
            bot_state2 = c.get_bot_state(bid)
            if not bot_state2:
                continue
            bx2, by2 = bot_state2['x'], bot_state2['y']
            saved_task = {k: v for k, v in self.bot_tasks[bid].items()}
            recipe = self._generate_recipe(cand['order'], c, bid)
            if not recipe:
                self.bot_tasks[bid].update(saved_task)
                continue
            recipe_est = self._calculate_recipe_duration(recipe, bx2, by2)
            if recipe_est > cand['tleft'] * 1.1:
                self.bot_tasks[bid].update(saved_task)
                continue
            if turn + recipe_est > GameConstants.TOTAL_TURNS:
                self.bot_tasks[bid].update(saved_task)
                continue
            t = self.bot_tasks[bid]
            t['recipe'] = recipe
            t['step'] = 0
            t['order_id'] = oid
            t['stuck_count'] = 0
            t['last_progress'] = turn
            self.claimed_order_ids.add(oid)
            self._stat_orders_started += 1
            self._write_log(f"ASSIGNED Order #{oid} to Bot {bid}")
            assigned_bots.add(bid)
            assigned_oids.add(oid)
            newly_committed += cand['cost']
            if cand['needs_cook']:
                n_cooking_bots += 1
            if cand['needs_chop']:
                n_chopping_bots += 1

        # Assign future orders to still-idle bots
        still_idle = [bid for bid in idle_bots if bid not in assigned_bots]
        if still_idle:
            future_candidates = self._prioritize_upcoming_orders(orders, turn, c)
            self._write_log(f"  Future order candidates: {len(future_candidates)}")
            for bid in still_idle:
                if not future_candidates:
                    self._write_log(f"  Bot {bid} has no future orders to consider")
                    break
                bot_state = c.get_bot_state(bid)
                if not bot_state:
                    continue
                bx, by = bot_state['x'], bot_state['y']
                self._write_log(f"  Bot {bid} at ({bx},{by}) evaluating future orders...")
                for fc in future_candidates:
                    oid = fc['order']['order_id']
                    if oid in self.claimed_order_ids or oid in assigned_oids:
                        continue
                    # Check if enough money remains after committed spending
                    if available_money - newly_committed < fc['cost'] + 2:
                        self._write_log(f"    Skip future #{oid}: insufficient money")
                        continue
                    if fc['needs_cook'] and n_cooking_bots >= max(self._stove_count * 2, 2):
                        self._write_log(f"    Skip future #{oid}: too many cooking bots")
                        continue
                    if fc['needs_chop'] and n_chopping_bots >= max(self._counter_count * 2, 2):
                        self._write_log(f"    Skip future #{oid}: too many chopping bots")
                        continue
                    est = self._predict_completion_time(bx, by, fc['order'])
                    if turn + est > GameConstants.TOTAL_TURNS:
                        self._write_log(f"    Skip future #{oid}: would end after game")
                        continue
                    # For future orders, check against ACTUAL tleft (time until expiry)
                    # NOT effective_tleft which incorrectly adds wait_turns
                    actual_tleft = fc['tleft']  # This is expires_turn - current_turn
                    if est > actual_tleft * self._feasibility_mult:
                        self._write_log(f"    Skip future #{oid}: est={est} > tleft*mult={actual_tleft}*{self._feasibility_mult:.1f}={actual_tleft*self._feasibility_mult:.0f}")
                        continue
                    self._write_log(f"    Consider future #{oid}: wait={fc.get('wait_turns', 0)} est={est}")
                    saved_task = {k: v for k, v in self.bot_tasks[bid].items()}
                    recipe = self._generate_recipe(fc['order'], c, bid)
                    if recipe:
                        recipe_est = self._calculate_recipe_duration(recipe, bx, by)
                        if turn + recipe_est > GameConstants.TOTAL_TURNS:
                            self.bot_tasks[bid].update(saved_task)
                            self._write_log(f"    Reject future #{oid}: recipe would end after game")
                            continue
                        # Also check against actual time left, not inflated effective_tleft
                        if recipe_est > actual_tleft * 1.1:
                            self.bot_tasks[bid].update(saved_task)
                            self._write_log(f"    Reject future #{oid}: recipe_est={recipe_est} > tleft*1.1={actual_tleft*1.1:.0f}")
                            continue
                        t = self.bot_tasks[bid]
                        t['recipe'] = recipe
                        t['step'] = 0
                        t['order_id'] = oid
                        t['stuck_count'] = 0
                        t['last_progress'] = turn
                        t['is_future_order'] = True
                        self.claimed_order_ids.add(oid)
                        self._stat_orders_started += 1
                        self._write_log(f"ASSIGNED Future Order #{oid} to Bot {bid}")
                        assigned_bots.add(bid)
                        assigned_oids.add(oid)
                        newly_committed += fc['cost']
                        if fc['needs_cook']:
                            n_cooking_bots += 1
                        if fc['needs_chop']:
                            n_chopping_bots += 1
                        break

        # DEFENSE MODE: When enemy has switched to our map, protect valuable items
        if self._defense_mode:
            self._execute_defense_protocol(c, bots, team, turn)

        # Execute all bots
        for bid in bots:
            self._run_task_step(c, bid)

        # Post-exec: assign orders to newly idle bots
        newly_idle = []
        for bid in bots:
            t = self.bot_tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                continue
            if bid not in idle_bots:
                newly_idle.append(bid)

        if newly_idle:
            money2 = c.get_team_money(team)
            committed_spending2 = 0
            for bid in bots:
                t = self.bot_tasks[bid]
                if t.get('recipe') and t.get('order_id') is not None:
                    recipe = t.get('recipe', [])
                    step = t.get('step', 0)
                    for i in range(step, len(recipe)):
                        action = recipe[i]
                        if action[0] == 'buy':
                            item = action[1]
                            cost = getattr(item, 'buy_cost', 0)
                            committed_spending2 += cost
            available_money2 = money2 - committed_spending2
            candidates2 = self._prioritize_active_orders(orders, turn, c)
            n_ck2 = sum(1 for b in bots
                        if self.bot_tasks[b].get('cooker') is not None
                        and self.bot_tasks[b].get('order_id') is not None)
            n_ch2 = sum(1 for b in bots
                        if self.bot_tasks[b].get('chop') is not None
                        and self.bot_tasks[b].get('order_id') is not None)
            assignments2 = []
            newly_committed2 = 0
            for bid in newly_idle:
                bot_state = c.get_bot_state(bid)
                if not bot_state:
                    continue
                bx, by = bot_state['x'], bot_state['y']
                for ci, cand in enumerate(candidates2):
                    oid = cand['order']['order_id']
                    if oid in self.claimed_order_ids:
                        continue
                    if available_money2 - newly_committed2 < cand['cost'] + 2:
                        continue
                    if cand['needs_cook'] and n_ck2 >= max(self._stove_count * 2, 2):
                        continue
                    if cand['needs_chop'] and n_ch2 >= max(self._counter_count * 2, 2):
                        continue
                    if self._chopping_disabled and cand['needs_chop']:
                        continue
                    if self._cooking_disabled and cand['needs_cook']:
                        continue
                    est = self._predict_completion_time(bx, by, cand['order'])
                    if est > cand['tleft'] * self._feasibility_mult:
                        continue
                    if turn + est > GameConstants.TOTAL_TURNS:
                        continue
                    eff = cand['profit'] / max(est, 1)
                    assignments2.append((eff, bid, ci))
            assignments2.sort(key=lambda x: -x[0])
            assigned_bots2 = set()
            assigned_oids2 = set()
            for eff, bid, ci in assignments2:
                if bid in assigned_bots2:
                    continue
                cand = candidates2[ci]
                oid = cand['order']['order_id']
                if oid in self.claimed_order_ids or oid in assigned_oids2:
                    continue
                if available_money2 - newly_committed2 < cand['cost'] + 2:
                    continue
                bot_state3 = c.get_bot_state(bid)
                if not bot_state3:
                    continue
                bx3, by3 = bot_state3['x'], bot_state3['y']
                saved_task2 = {k: v for k, v in self.bot_tasks[bid].items()}
                recipe = self._generate_recipe(cand['order'], c, bid)
                if recipe:
                    recipe_est = self._calculate_recipe_duration(recipe, bx3, by3)
                    if recipe_est > cand['tleft'] * 1.1:
                        self.bot_tasks[bid].update(saved_task2)
                        continue
                    if turn + recipe_est > GameConstants.TOTAL_TURNS:
                        self.bot_tasks[bid].update(saved_task2)
                        continue
                    t = self.bot_tasks[bid]
                    t['recipe'] = recipe
                    t['step'] = 0
                    t['order_id'] = oid
                    t['stuck_count'] = 0
                    t['last_progress'] = turn
                    self.claimed_order_ids.add(oid)
                    self._stat_orders_started += 1
                    assigned_bots2.add(bid)
                    assigned_oids2.add(oid)
                    newly_committed2 += cand['cost']

        # Check if idle bots are blocking working bots at corridors - yield if needed
        idle_bot_list = []
        working_bot_list = []
        for bid in bots:
            t = self.bot_tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                working_bot_list.append(bid)
            else:
                bs = c.get_bot_state(bid)
                if bs and not bs.get('holding'):
                    idle_bot_list.append(bid)
        
        # Check each idle bot to see if they're blocking a working bot
        for idle_bid in idle_bot_list:
            idle_bs = c.get_bot_state(idle_bid)
            if not idle_bs:
                continue
            idle_pos = (idle_bs['x'], idle_bs['y'])
            
            # Check if this idle bot is blocking any working bot
            for worker_bid in working_bot_list:
                worker_bs = c.get_bot_state(worker_bid)
                if not worker_bs:
                    continue
                worker_pos = (worker_bs['x'], worker_bs['y'])
                
                # Get worker's target
                worker_task = self.bot_tasks.get(worker_bid, {})
                recipe = worker_task.get('recipe', [])
                step = worker_task.get('step', 0)
                if step >= len(recipe):
                    continue
                
                current_step = recipe[step]
                target_pos = self._get_action_target(current_step)
                if target_pos is None:
                    continue
                
                # Get the adjacency target for the worker
                if current_step[0] != 'goto':
                    adj = self._get_adjacent_walkable(target_pos[0], target_pos[1])
                    if adj:
                        target_pos = min(adj, key=lambda a: self._path_distances.get(worker_pos, {}).get(a, 9999))
                
                # Check if idle bot is blocking
                if self._is_blocking_corridor(idle_pos, worker_pos, target_pos):
                    yield_pos = self._get_yield_position(c, idle_bid, worker_bid)
                    if yield_pos:
                        self._write_log(f"Bot {idle_bid} yielding from {idle_pos} to {yield_pos} for worker Bot {worker_bid}")
                        self._step_to_tile(c, idle_bid, yield_pos[0], yield_pos[1])
                        break  # Only need to yield once
        
        # Move idle bots toward shop (but avoid corridors if possible)
        # The yielding logic above already handles cases where idle bots block working bots
        for bid in bots:
            t = self.bot_tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            bs = c.get_bot_state(bid)
            if not bs or bs.get('holding'):
                continue
            bx, by = bs['x'], bs['y']
            
            shop = self._find_closest(bx, by, 'SHOP')
            if shop:
                shop_adj = self._get_adjacent_walkable(shop[0], shop[1])
                # Prefer non-corridor shop-adjacent tiles for idle positioning
                shop_adj_safe = [a for a in shop_adj if a not in self._corridor_tiles and a not in self._corridor_adjacent]
                if not shop_adj_safe:
                    shop_adj_safe = [a for a in shop_adj if a not in self._corridor_tiles]
                if not shop_adj_safe:
                    shop_adj_safe = shop_adj
                
                if shop_adj_safe and (bx, by) not in shop_adj_safe:
                    target = min(shop_adj_safe, key=lambda a: self._path_distances.get((bx, by), {}).get(a, 9999))
                    if target and target in self._path_distances.get((bx, by), {}):
                        self._navigate_toward(c, bid, target[0], target[1])

    def _prioritize_split_orders(self, orders: List[Dict], turn: int,
                              c: RobotController) -> List[Dict]:
        """Like _prioritize_active_orders but with relaxed filters for split maps."""
        out = []
        for o in orders:
            oid = o['order_id']
            if not o['is_active']:
                continue
            if oid in self.finished_order_ids or oid in self.claimed_order_ids:
                continue
            cost = ShopCosts.PLATE.buy_cost
            for fn in o['required']:
                ft = INGREDIENT_REGISTRY.get(fn)
                if ft:
                    cost += ft.buy_cost
            if oid in self.order_cooldowns:
                continue
            profit = o['reward'] - cost
            penalty = o.get('penalty', 0)
            if o['expires_turn'] >= GameConstants.TOTAL_TURNS:
                penalty = 0
            # Accept if profit is positive OR if completing avoids penalty
            if profit + penalty <= 0 and penalty <= 0:
                continue
            ingredients = [INGREDIENT_REGISTRY[fn] for fn in o['required'] if fn in INGREDIENT_REGISTRY]
            n_cook = sum(1 for f in ingredients if f.can_cook)
            n_chop = sum(1 for f in ingredients if f.can_chop)
            needs_cook = n_cook > 0
            tleft = o['expires_turn'] - turn
            if tleft < 10:
                continue
            value = max(profit + penalty, penalty)
            est_simple = 7 + len(ingredients) * 3 + n_cook * 21 + n_chop * 4
            efficiency = value / max(est_simple, 1)
            out.append({
                'order': o, 'cost': cost, 'profit': profit,
                'tleft': tleft, 'est': est_simple, 'efficiency': efficiency,
                'needs_cook': needs_cook, 'needs_chop': n_chop > 0,
                'penalty': penalty,
            })
        out.sort(key=lambda x: -x['efficiency'])
        return out

    def _predict_split_completion(self, order: Dict) -> int:
        """Rough turn estimate for a split map order."""
        ingredients = [INGREDIENT_REGISTRY[fn] for fn in order['required'] if fn in INGREDIENT_REGISTRY]
        if not ingredients:
            return 9999
        n_cook = sum(1 for f in ingredients if f.can_cook)
        n_chop = sum(1 for f in ingredients if f.can_chop)
        # Base: travel + plate + simple items
        est = 10 + len(ingredients) * 3
        # Chop items: buy + place + chop + pickup + return (~8 each)
        est += n_chop * 8
        # Cook items: bridge handoff + cook time + return (~30 each)
        est += n_cook * (GameConstants.COOK_PROGRESS + 10)
        # Submit
        est += 5
        return est

    def _allocate_split_tasks(self, c: RobotController, bots, idle_bots,
                          orders, turn, money):
        """Assign cooperative orders on split maps."""
        idle_runners = [bid for bid in idle_bots if self._bot_role.get(bid) == 'runner']
        idle_producers = [bid for bid in idle_bots if self._bot_role.get(bid) == 'producer']

        if not idle_runners or not idle_producers:
            return

        # Calculate available money (subtract committed spending)
        committed_spending = 0
        for bid in bots:
            t = self.bot_tasks[bid]
            if t.get('recipe') and t.get('order_id') is not None:
                recipe = t.get('recipe', [])
                step = t.get('step', 0)
                for i in range(step, len(recipe)):
                    action = recipe[i]
                    if action[0] == 'buy':
                        item = action[1]
                        cost = getattr(item, 'buy_cost', 0)
                        committed_spending += cost
        available_money = money - committed_spending

        # Use split-specific order ranking with relaxed filters
        candidates = self._prioritize_split_orders(orders, turn, c)

        for runner_bid in idle_runners:
            if not idle_producers:
                break
            producer_bid = idle_producers[0]
            for cand in candidates:
                oid = cand['order']['order_id']
                if oid in self.claimed_order_ids:
                    continue
                if available_money < cand['cost'] + 2:
                    continue
                est = self._predict_split_completion(cand['order'])
                if est > cand['tleft'] * 1.48:
                    continue
                if turn + est > GameConstants.TOTAL_TURNS:
                    continue
                result = self._generate_split_recipe(cand['order'], c, runner_bid, producer_bid)
                if not result:
                    continue
                runner_recipe, producer_recipe = result
                rt = self.bot_tasks[runner_bid]
                rt['recipe'] = runner_recipe
                rt['step'] = 0
                rt['order_id'] = oid
                rt['stuck_count'] = 0
                rt['last_progress'] = turn
                rt['is_split_order'] = True
                rt['partner_bid'] = producer_bid
                pt = self.bot_tasks[producer_bid]
                pt['recipe'] = producer_recipe
                pt['step'] = 0
                pt['order_id'] = oid
                pt['stuck_count'] = 0
                pt['last_progress'] = turn
                pt['is_split_order'] = True
                pt['partner_bid'] = runner_bid
                self.claimed_order_ids.add(oid)
                self._stat_orders_started += 1
                available_money -= cand['cost']
                idle_producers.pop(0)
                self._write_log(f"SPLIT ASSIGNED Order #{oid} to Runner={runner_bid}, Producer={producer_bid}")
                break
