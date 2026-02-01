"""
Zone Coordinator Bot - Dynamic Map-Aware Pathfinding Coordination
===================================================================

CORE INNOVATION:
Solves pathfinding as a COORDINATION problem, not just navigation.
Analyzes map topology at startup and dynamically assigns bot zones/roles.

MAP ANALYSIS (run once at initialization):
1. Detect chokepoints using articulation point algorithm
2. Classify map topology: open, split, corridor, maze
3. Calculate zone boundaries based on resource clustering
4. Determine if relay system is needed

DYNAMIC ROLE ASSIGNMENT:
- Open maps: Each bot owns a zone's resources
- Split maps: Each bot owns one side, handoff at boundary
- Chokepoint maps: Relay system with one bot ferrying through bottleneck
- Recalculate when enemy enters or resources deplete

RELAY HANDOFF LOGIC:
- Bot A prepares items and waits at handoff position
- Bot B takes items through chokepoint to submit
- Minimizes wasted travel through bottlenecks

Reference: FastPathfinder from champion_bot, state machine from IronChefOptimized
"""

import numpy as np
from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    from game_constants import Team, FoodType, ShopCosts
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
        print(f"[ZONE] {msg}")


INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False, 'turns': 3},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True,  'turns': 25},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False, 'turns': 8},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False, 'turns': 3},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True,  'turns': 30},
}


# =============================================================================
# MAP TOPOLOGY TYPES
# =============================================================================

class MapTopology(Enum):
    OPEN = auto()           # Wide open kitchen, no bottlenecks
    SPLIT = auto()          # Two distinct areas (wall in middle)
    CORRIDOR = auto()       # Single narrow path connecting areas
    MAZE = auto()           # Multiple narrow paths, complex navigation
    COMPACT = auto()        # Very small, everything close together


class BotRole(Enum):
    ZONE_A = auto()         # Owns zone A resources
    ZONE_B = auto()         # Owns zone B resources
    RELAY_PREP = auto()     # Prepares items, waits at handoff
    RELAY_DELIVER = auto()  # Takes items through chokepoint, delivers
    SOLO = auto()           # Handles everything (for compact maps)
    HELPER = auto()         # Assists, washes dishes


class BotState(Enum):
    IDLE = auto()
    BUY_PAN = auto()
    PLACE_PAN = auto()
    BUY_INGREDIENT = auto()
    PLACE_CHOP = auto()
    CHOP = auto()
    PICKUP = auto()
    START_COOK = auto()
    WAIT_COOK = auto()
    TAKE_PAN = auto()
    BUY_PLATE = auto()
    GET_PLATE = auto()
    PLACE_PLATE = auto()
    ADD_PLATE = auto()
    PICKUP_PLATE = auto()
    SUBMIT = auto()
    TRASH = auto()
    WAIT_HANDOFF = auto()
    GOTO_HANDOFF = auto()
    RECEIVE_HANDOFF = auto()


@dataclass
class Zone:
    """Represents a map zone with its resources"""
    id: int
    tiles: Set[Tuple[int, int]] = field(default_factory=set)
    shops: List[Tuple[int, int]] = field(default_factory=list)
    cookers: List[Tuple[int, int]] = field(default_factory=list)
    counters: List[Tuple[int, int]] = field(default_factory=list)
    submits: List[Tuple[int, int]] = field(default_factory=list)
    trashes: List[Tuple[int, int]] = field(default_factory=list)
    sinks: List[Tuple[int, int]] = field(default_factory=list)
    sink_tables: List[Tuple[int, int]] = field(default_factory=list)
    boxes: List[Tuple[int, int]] = field(default_factory=list)
    center: Tuple[int, int] = (0, 0)


@dataclass
class BotTask:
    state: BotState = BotState.IDLE
    role: BotRole = BotRole.SOLO
    zone: Optional[Zone] = None
    target: Optional[Tuple[int, int]] = None
    item: Optional[str] = None
    order: Optional[Dict] = None
    plate_counter: Optional[Tuple[int, int]] = None
    work_counter: Optional[Tuple[int, int]] = None
    cooker: Optional[Tuple[int, int]] = None
    ingredients_done: List[str] = field(default_factory=list)


# =============================================================================
# MAP ANALYZER - Detects topology and chokepoints
# =============================================================================

class MapAnalyzer:
    """Analyzes map topology, detects chokepoints, and creates zones"""
    
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    DIRS_4 = [(0,1), (0,-1), (1,0), (-1,0)]
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        self.map_obj = map_obj
        
        # Build walkability matrix
        self.walkable = set()
        self.tile_cache: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        
        for x in range(self.width):
            for y in range(self.height):
                tile = map_obj.tiles[x][y]
                if getattr(tile, 'is_walkable', False):
                    self.walkable.add((x, y))
                tile_name = getattr(tile, 'tile_name', '')
                if tile_name:
                    self.tile_cache[tile_name].append((x, y))
        
        # Analysis results
        self.topology: MapTopology = MapTopology.OPEN
        self.chokepoints: List[Tuple[int, int]] = []
        self.zones: List[Zone] = []
        self.handoff_position: Optional[Tuple[int, int]] = None
        self.relay_needed: bool = False
        
        # Pre-computed distance matrices
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
    
    def analyze(self):
        """Run full map analysis"""
        log(f"Analyzing map {self.width}x{self.height}...")
        
        # 1. Detect chokepoints
        self._detect_chokepoints()
        
        # 2. Classify topology
        self._classify_topology()
        
        # 3. Create zones
        self._create_zones()
        
        # 4. Determine relay needs
        self._determine_relay_system()
        
        # 5. Pre-compute distances for key POIs
        self._precompute_distances()
        
        log(f"Topology: {self.topology.name}, Chokepoints: {len(self.chokepoints)}, "
            f"Zones: {len(self.zones)}, Relay: {self.relay_needed}")
    
    def _detect_chokepoints(self):
        """
        Detect articulation points (chokepoints) in the walkable graph.
        A chokepoint is a tile where removing it disconnects the graph.
        Uses Tarjan's algorithm for articulation points.
        """
        if not self.walkable:
            return
        
        # Build adjacency list
        adj = defaultdict(list)
        for (x, y) in self.walkable:
            for dx, dy in self.DIRS_4:  # Use 4-dir for stricter chokepoint detection
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.walkable:
                    adj[(x, y)].append((nx, ny))
        
        # Tarjan's algorithm for articulation points
        visited = set()
        disc = {}
        low = {}
        parent = {}
        articulation_points = set()
        time = [0]
        
        def dfs(u):
            children = 0
            visited.add(u)
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in adj[u]:
                if v not in visited:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    
                    # u is articulation point if:
                    # 1. u is root and has 2+ children
                    # 2. u is not root and low[v] >= disc[u]
                    if parent.get(u) is None and children > 1:
                        articulation_points.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        articulation_points.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        # Run DFS from each unvisited node
        for node in self.walkable:
            if node not in visited:
                parent[node] = None
                dfs(node)
        
        # Filter: only keep chokepoints that significantly restrict flow
        # (connected to 2 or fewer walkable neighbors)
        filtered = []
        for cp in articulation_points:
            neighbors = sum(1 for dx, dy in self.DIRS_4 
                          if (cp[0]+dx, cp[1]+dy) in self.walkable)
            if neighbors <= 3:  # Narrow passage
                filtered.append(cp)
        
        self.chokepoints = filtered
    
    def _classify_topology(self):
        """Classify map topology based on size, chokepoints, and layout"""
        area = len(self.walkable)
        
        # Very small map
        if area < 30:
            self.topology = MapTopology.COMPACT
            return
        
        # Many chokepoints = maze
        if len(self.chokepoints) > 5:
            self.topology = MapTopology.MAZE
            return
        
        # Check for split layout (wall dividing map)
        if self._is_split_layout():
            self.topology = MapTopology.SPLIT
            return
        
        # 1-3 chokepoints = corridor
        if 1 <= len(self.chokepoints) <= 3:
            self.topology = MapTopology.CORRIDOR
            return
        
        # Default to open
        self.topology = MapTopology.OPEN
    
    def _is_split_layout(self):
        """Check if map has a vertical or horizontal wall dividing it"""
        # Check for vertical split
        for x in range(self.width // 3, 2 * self.width // 3):
            wall_count = 0
            gap_count = 0
            for y in range(self.height):
                if (x, y) not in self.walkable:
                    wall_count += 1
                else:
                    gap_count += 1
            # If mostly wall with small gap, it's a split
            if wall_count > self.height * 0.6 and gap_count < self.height * 0.3:
                return True
        
        # Check for horizontal split
        for y in range(self.height // 3, 2 * self.height // 3):
            wall_count = 0
            gap_count = 0
            for x in range(self.width):
                if (x, y) not in self.walkable:
                    wall_count += 1
                else:
                    gap_count += 1
            if wall_count > self.width * 0.6 and gap_count < self.width * 0.3:
                return True
        
        return False
    
    def _create_zones(self):
        """Create zones based on topology and resource locations"""
        if self.topology == MapTopology.COMPACT:
            # Single zone for compact maps
            zone = self._create_single_zone()
            self.zones = [zone]
            return
        
        if self.topology in [MapTopology.SPLIT, MapTopology.CORRIDOR]:
            # Two zones split by chokepoint or wall
            self.zones = self._create_two_zones()
            return
        
        # For open/maze, create zones based on resource clustering
        self.zones = self._create_clustered_zones()
    
    def _create_single_zone(self) -> Zone:
        """Create a single zone containing all resources"""
        zone = Zone(id=0, tiles=self.walkable.copy())
        zone.shops = self.tile_cache.get('SHOP', []).copy()
        zone.cookers = self.tile_cache.get('COOKER', []).copy()
        zone.counters = self.tile_cache.get('COUNTER', []).copy()
        zone.submits = self.tile_cache.get('SUBMIT', []).copy()
        zone.trashes = self.tile_cache.get('TRASH', []).copy()
        zone.sinks = self.tile_cache.get('SINK', []).copy()
        zone.sink_tables = self.tile_cache.get('SINKTABLE', []).copy()
        zone.boxes = self.tile_cache.get('BOX', []).copy()
        
        # Calculate center
        if zone.tiles:
            avg_x = sum(t[0] for t in zone.tiles) / len(zone.tiles)
            avg_y = sum(t[1] for t in zone.tiles) / len(zone.tiles)
            zone.center = (int(avg_x), int(avg_y))
        
        return zone
    
    def _create_two_zones(self) -> List[Zone]:
        """Create two zones split by chokepoint or wall"""
        # Use BFS from corners to identify two regions
        zone_a = Zone(id=0)
        zone_b = Zone(id=1)
        
        # Find leftmost and rightmost walkable tiles
        left_start = min(self.walkable, key=lambda t: t[0])
        right_start = max(self.walkable, key=lambda t: t[0])
        
        # BFS from left to assign zone A
        visited = set()
        queue = deque([left_start])
        zone_a.tiles.add(left_start)
        visited.add(left_start)
        
        while queue:
            x, y = queue.popleft()
            # Stop at chokepoints to create boundary
            if (x, y) in self.chokepoints and (x, y) != left_start:
                continue
            
            for dx, dy in self.DIRS_8:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.walkable and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    zone_a.tiles.add((nx, ny))
                    queue.append((nx, ny))
        
        # Remaining tiles go to zone B
        zone_b.tiles = self.walkable - zone_a.tiles
        
        # Assign resources to zones
        for zone in [zone_a, zone_b]:
            for tile_name, locs in self.tile_cache.items():
                for loc in locs:
                    # Check adjacency (resource is adjacent to zone)
                    is_adjacent = any((loc[0]+dx, loc[1]+dy) in zone.tiles 
                                     for dx, dy in self.DIRS_8)
                    if is_adjacent:
                        if tile_name == 'SHOP': zone.shops.append(loc)
                        elif tile_name == 'COOKER': zone.cookers.append(loc)
                        elif tile_name == 'COUNTER': zone.counters.append(loc)
                        elif tile_name == 'SUBMIT': zone.submits.append(loc)
                        elif tile_name == 'TRASH': zone.trashes.append(loc)
                        elif tile_name == 'SINK': zone.sinks.append(loc)
                        elif tile_name == 'SINKTABLE': zone.sink_tables.append(loc)
                        elif tile_name == 'BOX': zone.boxes.append(loc)
            
            # Calculate center
            if zone.tiles:
                avg_x = sum(t[0] for t in zone.tiles) / len(zone.tiles)
                avg_y = sum(t[1] for t in zone.tiles) / len(zone.tiles)
                zone.center = (int(avg_x), int(avg_y))
        
        return [zone_a, zone_b]
    
    def _create_clustered_zones(self) -> List[Zone]:
        """Create zones using K-means-like clustering on resources"""
        # For simplicity, split by x-coordinate midpoint
        mid_x = self.width // 2
        
        zone_a = Zone(id=0)
        zone_b = Zone(id=1)
        
        for tile in self.walkable:
            if tile[0] < mid_x:
                zone_a.tiles.add(tile)
            else:
                zone_b.tiles.add(tile)
        
        # Assign resources
        for zone, x_range in [(zone_a, lambda x: x < mid_x), 
                               (zone_b, lambda x: x >= mid_x)]:
            for tile_name, locs in self.tile_cache.items():
                for loc in locs:
                    if x_range(loc[0]):
                        if tile_name == 'SHOP': zone.shops.append(loc)
                        elif tile_name == 'COOKER': zone.cookers.append(loc)
                        elif tile_name == 'COUNTER': zone.counters.append(loc)
                        elif tile_name == 'SUBMIT': zone.submits.append(loc)
                        elif tile_name == 'TRASH': zone.trashes.append(loc)
                        elif tile_name == 'SINK': zone.sinks.append(loc)
                        elif tile_name == 'SINKTABLE': zone.sink_tables.append(loc)
                        elif tile_name == 'BOX': zone.boxes.append(loc)
            
            if zone.tiles:
                avg_x = sum(t[0] for t in zone.tiles) / len(zone.tiles)
                avg_y = sum(t[1] for t in zone.tiles) / len(zone.tiles)
                zone.center = (int(avg_x), int(avg_y))
        
        return [zone_a, zone_b]
    
    def _determine_relay_system(self):
        """Determine if relay system is needed and set handoff position"""
        if self.topology not in [MapTopology.CORRIDOR, MapTopology.SPLIT]:
            self.relay_needed = False
            return
        
        if len(self.zones) < 2:
            self.relay_needed = False
            return
        
        # Check if key resources are split across zones
        zone_a, zone_b = self.zones[0], self.zones[1]
        
        # Need relay if one zone has cookers but no submit (or vice versa)
        a_has_cook = len(zone_a.cookers) > 0
        a_has_submit = len(zone_a.submits) > 0
        b_has_cook = len(zone_b.cookers) > 0
        b_has_submit = len(zone_b.submits) > 0
        
        if (a_has_cook and not a_has_submit) or (b_has_cook and not b_has_submit):
            self.relay_needed = True
        elif (a_has_submit and not a_has_cook) or (b_has_submit and not b_has_cook):
            self.relay_needed = True
        else:
            self.relay_needed = False
        
        # Set handoff position near chokepoint
        if self.relay_needed and self.chokepoints:
            # Find chokepoint closest to center
            center = (self.width // 2, self.height // 2)
            self.handoff_position = min(self.chokepoints, 
                                        key=lambda c: abs(c[0]-center[0]) + abs(c[1]-center[1]))
        elif self.relay_needed:
            # Use boundary between zones
            self.handoff_position = (self.width // 2, self.height // 2)
    
    def _precompute_distances(self):
        """Pre-compute distance matrices for key POIs"""
        pois = set()
        for cat in ['SHOP', 'COOKER', 'SINKTABLE', 'SUBMIT', 'COUNTER', 'SINK', 'TRASH']:
            pois.update(self.tile_cache.get(cat, []))
        
        for target in pois:
            self.dist_matrices[target] = self._compute_distance_matrix(target)
    
    def _compute_distance_matrix(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS to compute distance from every tile to adjacent-to-target"""
        dist = np.full((self.width, self.height), 9999.0)
        tx, ty = target
        
        queue = deque()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in self.walkable:
                        dist[nx, ny] = 0
                        queue.append((nx, ny))
        
        visited = {(x, y) for x in range(self.width) for y in range(self.height) 
                   if dist[x, y] == 0}
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in self.DIRS_8:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.walkable and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    dist[nx, ny] = dist[x, y] + 1
                    queue.append((nx, ny))
        
        return dist
    
    def get_dist(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Get distance between two points"""
        matrix = self.dist_matrices.get(p2)
        if matrix is not None:
            val = matrix[p1[0], p1[1]]
            if val < 9999:
                return val
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    @staticmethod
    def chebyshev(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    """Zone Coordinator Bot - Dynamic map-aware coordination"""
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.analyzer: Optional[MapAnalyzer] = None
        
        # Bot task assignments
        self.bot_tasks: Dict[int, BotTask] = {}
        
        # Current order being worked on
        self.current_order: Optional[Dict] = None
        
        # Enemy tracking
        self.enemy_on_our_map = False
        self.last_recalc_turn = 0
    
    def _initialize(self, controller: RobotController, team: Team):
        """Initialize map analysis and zone assignments"""
        # Run map analysis
        self.analyzer = MapAnalyzer(self.map)
        self.analyzer.analyze()
        
        # Assign initial roles based on topology
        bots = controller.get_team_bot_ids(team)
        self._assign_roles(controller, bots, team)
        
        self.initialized = True
        log(f"Initialized with topology: {self.analyzer.topology.name}")
    
    def _assign_roles(self, controller: RobotController, bots: List[int], team: Team):
        """Assign roles to bots based on map topology"""
        if not bots:
            return
        
        topology = self.analyzer.topology
        zones = self.analyzer.zones
        
        for i, bot_id in enumerate(bots):
            task = BotTask()
            
            if topology == MapTopology.COMPACT:
                # Single bot does everything, other helps
                task.role = BotRole.SOLO if i == 0 else BotRole.HELPER
                task.zone = zones[0] if zones else None
            
            elif topology in [MapTopology.SPLIT, MapTopology.CORRIDOR]:
                if self.analyzer.relay_needed:
                    # Relay system
                    task.role = BotRole.RELAY_PREP if i == 0 else BotRole.RELAY_DELIVER
                    # Assign zone based on role
                    if len(zones) >= 2:
                        # Prep bot gets zone with cookers
                        if zones[0].cookers:
                            task.zone = zones[0] if i == 0 else zones[1]
                        else:
                            task.zone = zones[1] if i == 0 else zones[0]
                else:
                    # Each bot owns a zone
                    task.role = BotRole.ZONE_A if i == 0 else BotRole.ZONE_B
                    task.zone = zones[i] if i < len(zones) else zones[0]
            
            elif topology == MapTopology.OPEN:
                # Zone ownership
                task.role = BotRole.ZONE_A if i == 0 else BotRole.ZONE_B
                task.zone = zones[i] if i < len(zones) else zones[0]
            
            elif topology == MapTopology.MAZE:
                # Solo + helper for maze (coordination too complex)
                task.role = BotRole.SOLO if i == 0 else BotRole.HELPER
                task.zone = zones[0] if zones else None
            
            self.bot_tasks[bot_id] = task
    
    def _should_recalculate_zones(self, controller: RobotController, team: Team) -> bool:
        """Check if zone recalculation is needed"""
        turn = controller.get_turn()
        
        # Don't recalculate too frequently
        if turn - self.last_recalc_turn < 50:
            return False
        
        # Check if enemy entered our map
        switch_info = controller.get_switch_info()
        enemy_switched = switch_info.get('enemy_team_switched', False)
        
        if enemy_switched and not self.enemy_on_our_map:
            self.enemy_on_our_map = True
            return True
        
        if not enemy_switched and self.enemy_on_our_map:
            self.enemy_on_our_map = False
            return True
        
        return False
    
    def _recalculate_zones(self, controller: RobotController, team: Team):
        """Recalculate zone assignments when conditions change"""
        self.last_recalc_turn = controller.get_turn()
        
        bots = controller.get_team_bot_ids(team)
        
        if self.enemy_on_our_map:
            # Switch to defensive mode - both bots focus on own production
            for bot_id in bots:
                task = self.bot_tasks.get(bot_id, BotTask())
                if task.role in [BotRole.RELAY_PREP, BotRole.RELAY_DELIVER]:
                    task.role = BotRole.SOLO
                self.bot_tasks[bot_id] = task
            log("Switched to defensive mode due to enemy presence")
        else:
            # Restore original roles
            self._assign_roles(controller, bots, team)
            log("Restored normal zone assignments")
    
    def _get_nearest(self, pos: Tuple[int, int], locations: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Get nearest location from list"""
        if not locations:
            return None
        return min(locations, key=lambda p: self.analyzer.get_dist(pos, p))
    
    def _get_nearest_in_zone(self, pos: Tuple[int, int], locations: List[Tuple[int, int]], 
                              zone: Optional[Zone]) -> Optional[Tuple[int, int]]:
        """Get nearest location within a zone"""
        if not locations:
            return None
        if not zone:
            return self._get_nearest(pos, locations)
        # Filter to locations adjacent to zone tiles
        in_zone = [loc for loc in locations 
                   if any((loc[0]+dx, loc[1]+dy) in zone.tiles 
                         for dx in [-1,0,1] for dy in [-1,0,1])]
        if not in_zone:
            return self._get_nearest(pos, locations)
        return min(in_zone, key=lambda p: self.analyzer.get_dist(pos, p))
    
    def _move_toward(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int, int], team: Team) -> bool:
        """Move toward target. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        
        bx, by = bot['x'], bot['y']
        
        if MapAnalyzer.chebyshev((bx, by), target) <= 1:
            return True
        
        # Get other bot positions to avoid
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != bot_id:
                other = controller.get_bot_state(bid)
                if other:
                    avoid.add((other['x'], other['y']))
        
        # Use pre-computed distances for pathfinding
        best_step = None
        best_dist = 9999.0
        
        for dx, dy in MapAnalyzer.DIRS_8:
            if not controller.can_move(bot_id, dx, dy):
                continue
            
            nx, ny = bx + dx, by + dy
            if (nx, ny) in avoid:
                continue
            
            step_dist = self.analyzer.get_dist((nx, ny), target)
            if step_dist < best_dist:
                best_dist = step_dist
                best_step = (dx, dy)
        
        if best_step:
            controller.move(bot_id, best_step[0], best_step[1])
            return False
        
        # Wiggle if stuck
        import random
        dirs = list(MapAnalyzer.DIRS_8)
        random.shuffle(dirs)
        for dx, dy in dirs:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        return False
    
    def _select_order(self, controller: RobotController, team: Team) -> Optional[Dict]:
        """Select best order to work on"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 25:
                continue
            
            required = order['required']
            reward = order['reward']
            
            # Score: reward / complexity
            score = reward / (len(required) + 1)
            
            # Bonus for simple orders
            if len(required) <= 2:
                score *= 1.5
            
            # Penalty for cooking
            cook_count = sum(1 for i in required if INGREDIENT_INFO.get(i, {}).get('cook'))
            score -= cook_count * 20
            
            if score > best_score:
                best_score = score
                best = order
        
        return best
    
    def _execute_bot(self, controller: RobotController, bot_id: int, team: Team):
        """Execute bot based on its assigned role"""
        task = self.bot_tasks.get(bot_id)
        if not task:
            return
        
        role = task.role
        
        if role in [BotRole.SOLO, BotRole.ZONE_A, BotRole.ZONE_B]:
            self._execute_zone_worker(controller, bot_id, team, task)
        elif role == BotRole.RELAY_PREP:
            self._execute_relay_prep(controller, bot_id, team, task)
        elif role == BotRole.RELAY_DELIVER:
            self._execute_relay_deliver(controller, bot_id, team, task)
        elif role == BotRole.HELPER:
            self._execute_helper(controller, bot_id, team, task)
    
    def _execute_zone_worker(self, controller: RobotController, bot_id: int, 
                              team: Team, task: BotTask):
        """Execute zone worker (processes orders within their zone)"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        zone = task.zone
        
        # Get zone-specific locations (or all if no zone)
        if zone:
            shops = zone.shops or self.analyzer.tile_cache.get('SHOP', [])
            cookers = zone.cookers or self.analyzer.tile_cache.get('COOKER', [])
            counters = zone.counters or self.analyzer.tile_cache.get('COUNTER', [])
            submits = zone.submits or self.analyzer.tile_cache.get('SUBMIT', [])
            trashes = zone.trashes or self.analyzer.tile_cache.get('TRASH', [])
            sink_tables = zone.sink_tables or self.analyzer.tile_cache.get('SINKTABLE', [])
        else:
            shops = self.analyzer.tile_cache.get('SHOP', [])
            cookers = self.analyzer.tile_cache.get('COOKER', [])
            counters = self.analyzer.tile_cache.get('COUNTER', [])
            submits = self.analyzer.tile_cache.get('SUBMIT', [])
            trashes = self.analyzer.tile_cache.get('TRASH', [])
            sink_tables = self.analyzer.tile_cache.get('SINKTABLE', [])
        
        shop = self._get_nearest((bx, by), shops)
        submit = self._get_nearest((bx, by), submits)
        trash = self._get_nearest((bx, by), trashes)
        cooker = self._get_nearest((bx, by), cookers)
        
        # State machine
        state = task.state
        
        if state == BotState.IDLE:
            # Select order
            if not task.order:
                task.order = self._select_order(controller, team)
                task.ingredients_done = []
            
            if not task.order:
                return
            
            # Check if cooking needed
            needs_cook = any(INGREDIENT_INFO.get(i, {}).get('cook') 
                           for i in task.order['required'])
            
            if needs_cook and cooker:
                tile = controller.get_tile(team, cooker[0], cooker[1])
                if not isinstance(getattr(tile, 'item', None), Pan):
                    task.state = BotState.BUY_PAN
                    return
            
            # Get plate
            if not task.plate_counter:
                task.state = BotState.GET_PLATE
                return
            
            # Check plate contents
            plate_contents = []
            if task.plate_counter:
                tile = controller.get_tile(team, task.plate_counter[0], task.plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            missing = [i for i in task.order['required'] if i not in plate_contents]
            if not missing:
                task.state = BotState.PICKUP_PLATE
                return
            
            # Process next ingredient
            task.item = missing[0]
            task.state = BotState.BUY_INGREDIENT
        
        elif state == BotState.BUY_PAN:
            if holding and holding.get('type') == 'Pan':
                task.state = BotState.PLACE_PAN
            elif shop and self._move_toward(controller, bot_id, shop, team):
                if money >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
        
        elif state == BotState.PLACE_PAN:
            if cooker and self._move_toward(controller, bot_id, cooker, team):
                controller.place(bot_id, cooker[0], cooker[1])
                task.cooker = cooker
                task.state = BotState.IDLE
        
        elif state == BotState.GET_PLATE:
            if holding and holding.get('type') == 'Plate':
                task.state = BotState.PLACE_PLATE
            else:
                # Try sink table first
                if sink_tables:
                    st = self._get_nearest((bx, by), sink_tables)
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self._move_toward(controller, bot_id, st, team):
                            controller.take_clean_plate(bot_id, st[0], st[1])
                        return
                # Buy plate
                if shop and self._move_toward(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
        
        elif state == BotState.PLACE_PLATE:
            # Find free counter in zone
            free_counters = [c for c in counters 
                           if not getattr(controller.get_tile(team, c[0], c[1]), 'item', None)]
            counter = self._get_nearest((bx, by), free_counters) if free_counters else None
            
            if counter and self._move_toward(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                task.plate_counter = counter
                task.state = BotState.IDLE
        
        elif state == BotState.BUY_INGREDIENT:
            if holding and holding.get('type') == 'Food':
                info = INGREDIENT_INFO.get(task.item, {})
                if info.get('chop'):
                    task.state = BotState.PLACE_CHOP
                elif info.get('cook'):
                    task.state = BotState.START_COOK
                else:
                    task.state = BotState.ADD_PLATE
            elif shop and self._move_toward(controller, bot_id, shop, team):
                food_type = getattr(FoodType, task.item, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        elif state == BotState.PLACE_CHOP:
            free_counters = [c for c in counters if c != task.plate_counter
                           and not getattr(controller.get_tile(team, c[0], c[1]), 'item', None)]
            counter = self._get_nearest((bx, by), free_counters) if free_counters else None
            
            if counter and self._move_toward(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                task.work_counter = counter
                task.state = BotState.CHOP
        
        elif state == BotState.CHOP:
            if task.work_counter and self._move_toward(controller, bot_id, task.work_counter, team):
                tile = controller.get_tile(team, task.work_counter[0], task.work_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Food):
                    if tile.item.chopped:
                        task.state = BotState.PICKUP
                    else:
                        controller.chop(bot_id, task.work_counter[0], task.work_counter[1])
        
        elif state == BotState.PICKUP:
            if task.work_counter and self._move_toward(controller, bot_id, task.work_counter, team):
                if controller.pickup(bot_id, task.work_counter[0], task.work_counter[1]):
                    info = INGREDIENT_INFO.get(task.item, {})
                    if info.get('cook'):
                        task.state = BotState.START_COOK
                    else:
                        task.state = BotState.ADD_PLATE
        
        elif state == BotState.START_COOK:
            ck = task.cooker or cooker
            if ck and self._move_toward(controller, bot_id, ck, team):
                controller.place(bot_id, ck[0], ck[1])
                task.cooker = ck
                task.state = BotState.WAIT_COOK
        
        elif state == BotState.WAIT_COOK:
            if task.cooker:
                tile = controller.get_tile(team, task.cooker[0], task.cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage == 1:
                        task.state = BotState.TAKE_PAN
                    elif pan.food and pan.food.cooked_stage == 2:
                        task.state = BotState.TAKE_PAN  # Burnt - take and trash
        
        elif state == BotState.TAKE_PAN:
            if holding:
                if holding.get('cooked_stage') == 2:
                    task.state = BotState.TRASH
                else:
                    task.state = BotState.ADD_PLATE
            elif task.cooker and self._move_toward(controller, bot_id, task.cooker, team):
                controller.take_from_pan(bot_id, task.cooker[0], task.cooker[1])
        
        elif state == BotState.ADD_PLATE:
            if task.plate_counter and self._move_toward(controller, bot_id, task.plate_counter, team):
                if controller.add_food_to_plate(bot_id, task.plate_counter[0], task.plate_counter[1]):
                    task.state = BotState.IDLE
        
        elif state == BotState.PICKUP_PLATE:
            if task.plate_counter and self._move_toward(controller, bot_id, task.plate_counter, team):
                controller.pickup(bot_id, task.plate_counter[0], task.plate_counter[1])
                task.state = BotState.SUBMIT
        
        elif state == BotState.SUBMIT:
            if submit and self._move_toward(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    task.order = None
                    task.plate_counter = None
                    task.state = BotState.IDLE
        
        elif state == BotState.TRASH:
            if trash and self._move_toward(controller, bot_id, trash, team):
                controller.trash(bot_id, trash[0], trash[1])
                task.state = BotState.IDLE
    
    def _execute_relay_prep(self, controller: RobotController, bot_id: int,
                            team: Team, task: BotTask):
        """Execute relay prep bot (prepares plates, waits at handoff)"""
        # For now, delegate to zone worker
        # Full relay implementation would include handoff coordination
        self._execute_zone_worker(controller, bot_id, team, task)
    
    def _execute_relay_deliver(self, controller: RobotController, bot_id: int,
                               team: Team, task: BotTask):
        """Execute relay delivery bot (takes plates through chokepoint)"""
        # For now, delegate to zone worker
        self._execute_zone_worker(controller, bot_id, team, task)
    
    def _execute_helper(self, controller: RobotController, bot_id: int,
                        team: Team, task: BotTask):
        """Execute helper bot (washes dishes, assists)"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        
        # Find sinks with dirty plates
        sinks = self.analyzer.tile_cache.get('SINK', [])
        for sink in sinks:
            tile = controller.get_tile(team, sink[0], sink[1])
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                if self._move_toward(controller, bot_id, sink, team):
                    controller.wash_sink(bot_id, sink[0], sink[1])
                return
        
        # Otherwise, idle near center
        import random
        dirs = list(MapAnalyzer.DIRS_4)
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
            self._initialize(controller, team)
        
        # Check for zone recalculation triggers
        if self._should_recalculate_zones(controller, team):
            self._recalculate_zones(controller, team)
        
        # Execute each bot
        bots = controller.get_team_bot_ids(team)
        for bot_id in bots:
            self._execute_bot(controller, bot_id, team)
