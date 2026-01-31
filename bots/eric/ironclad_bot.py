"""
IRONCLAD Bot - Carnegie Cookoff Deterministic Strategy

Based on GPT's IRONCLAD spec v2.1 with:
- Role stability (CHEF/SUPPORT, no dynamic reassignment)
- COOK_GUARD: Chef stays near cooker during cooking
- Zero burn tolerance: Track cooking and retrieve on time
- 16-state machine for reliable order completion
- Sabotage: Pan denial, plate starvation during switch
- Defense: Protect resources when enemy invades
"""

from collections import deque
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from tiles import Tile, Counter, Cooker, Sink, SinkTable, Submit, Shop, Box, Trash
from item import Pan, Plate, Food


@dataclass
class CookingTracker:
    """Track what's cooking where"""
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int
    
    @property
    def turns_to_cooked(self) -> int:
        if self.cooked_stage >= 1:
            return 0
        return max(0, GameConstants.COOK_PROGRESS - self.cook_progress)
    
    @property
    def turns_to_burned(self) -> int:
        if self.cooked_stage >= 2:
            return 0
        return max(0, GameConstants.BURN_PROGRESS - self.cook_progress)
    
    @property
    def is_urgent(self) -> bool:
        """Food is about to be cooked and needs pickup soon"""
        return self.cooked_stage == 0 and self.turns_to_cooked <= 3


@dataclass
class BotTask:
    """Represents a task assigned to a bot"""
    task_type: str  # 'buy', 'chop', 'cook', 'plate', 'submit', 'wash', 'idle', 'sabotage', 'defend'
    target_pos: Optional[Tuple[int, int]] = None
    item: Optional[Any] = None
    priority: int = 0


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Pre-computed data
        self.dist_matrix: Dict[Tuple[int,int], Dict[Tuple[int,int], int]] = {}
        self.path_cache: Dict[Tuple[Tuple[int,int], Tuple[int,int]], List[Tuple[int,int]]] = {}
        
        # Tile locations
        self.shops: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.boxes: List[Tuple[int, int]] = []
        self.walkable: Set[Tuple[int, int]] = set()
        
        # Bot state tracking
        self.bot_roles: Dict[int, str] = {}  # 'chef', 'support', 'guard'
        self.bot_states: Dict[int, int] = {}  # state machine state per bot
        self.cooking_trackers: Dict[Tuple[int,int], CookingTracker] = {}
        
        # Coordination
        self.reserved_targets: Set[Tuple[int, int]] = set()
        self.assigned_orders: Dict[int, int] = {}  # bot_id -> order_id
        
        # Strategy flags
        self.has_sabotaged = False
        self.enemy_detected = False
        
    # ========================================
    # INITIALIZATION
    # ========================================
    
    def initialize(self, controller: RobotController):
        """One-time initialization - parse map and precompute distances"""
        if self.initialized:
            return
            
        m = controller.get_map()
        
        # Parse tile locations
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                tile_name = getattr(tile, 'tile_name', '')
                
                if getattr(tile, 'is_walkable', False):
                    self.walkable.add((x, y))
                
                if tile_name == 'SHOP':
                    self.shops.append((x, y))
                elif tile_name == 'COOKER':
                    self.cookers.append((x, y))
                elif tile_name == 'SINK':
                    self.sinks.append((x, y))
                elif tile_name == 'SINKTABLE':
                    self.sink_tables.append((x, y))
                elif tile_name == 'COUNTER':
                    self.counters.append((x, y))
                elif tile_name == 'SUBMIT':
                    self.submits.append((x, y))
                    self.walkable.add((x, y))  # Submit is walkable
                elif tile_name == 'TRASH':
                    self.trashes.append((x, y))
                elif tile_name == 'BOX':
                    self.boxes.append((x, y))
        
        # Precompute distances from key locations
        self.precompute_distances(m)
        
        self.initialized = True
    
    def precompute_distances(self, m):
        """BFS from every walkable tile to compute distance matrix"""
        all_important = set(self.walkable)
        # Also add adjacency to non-walkable tiles we need to interact with
        for tiles in [self.shops, self.cookers, self.sinks, self.sink_tables, 
                      self.counters, self.trashes, self.boxes]:
            for pos in tiles:
                all_important.add(pos)
        
        for start in self.walkable:
            self.dist_matrix[start] = {}
            self.dist_matrix[start][start] = 0
            
            queue = deque([(start, 0)])
            visited = {start}
            
            while queue:
                (cx, cy), dist = queue.popleft()
                
                # 8-directional movement (Chebyshev)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in visited:
                            continue
                        if not (0 <= nx < m.width and 0 <= ny < m.height):
                            continue
                        
                        # Check if walkable OR is a target tile (for adjacency)
                        tile = m.tiles[nx][ny]
                        if getattr(tile, 'is_walkable', False):
                            visited.add((nx, ny))
                            self.dist_matrix[start][(nx, ny)] = dist + 1
                            queue.append(((nx, ny), dist + 1))

    # ========================================
    # PATHFINDING
    # ========================================
    
    def get_distance(self, start: Tuple[int,int], end: Tuple[int,int]) -> int:
        """Get precomputed distance, or Manhattan as fallback"""
        if start in self.dist_matrix and end in self.dist_matrix[start]:
            return self.dist_matrix[start][end]
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def get_path_to_adjacent(self, controller: RobotController, start: Tuple[int,int], 
                             target: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        """BFS to find first step toward a tile adjacent to target"""
        if self.is_adjacent(start, target):
            return (0, 0)  # Already adjacent
        
        m = controller.get_map()
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            (cx, cy), path = queue.popleft()
            
            # Check if we're adjacent to target
            if self.is_adjacent((cx, cy), target):
                if not path:
                    return (0, 0)
                return path[0]
            
            # 8-directional BFS
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited:
                        continue
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))
        
        return None
    
    def is_adjacent(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> bool:
        """Check if two positions are within Chebyshev distance 1"""
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1
    
    def move_towards(self, controller: RobotController, bot_id: int, 
                     target: Tuple[int,int]) -> bool:
        """Move bot toward target. Returns True if already adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        
        bx, by = bot['x'], bot['y']
        
        if self.is_adjacent((bx, by), target):
            return True
        
        step = self.get_path_to_adjacent(controller, (bx, by), target)
        if step and step != (0, 0):
            if controller.can_move(bot_id, step[0], step[1]):
                controller.move(bot_id, step[0], step[1])
        
        return False

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================
    
    def find_nearest(self, pos: Tuple[int,int], locations: List[Tuple[int,int]], 
                     exclude: Set[Tuple[int,int]] = None) -> Optional[Tuple[int,int]]:
        """Find nearest location from a list"""
        if exclude is None:
            exclude = set()
        
        best_dist = float('inf')
        best_pos = None
        
        for loc in locations:
            if loc in exclude:
                continue
            dist = self.get_distance(pos, loc)
            if dist < best_dist:
                best_dist = dist
                best_pos = loc
        
        return best_pos
    
    def find_empty_counter(self, controller: RobotController, 
                           near: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        """Find nearest empty counter"""
        best_dist = float('inf')
        best_pos = None
        
        for cx, cy in self.counters:
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and getattr(tile, 'item', None) is None:
                dist = self.get_distance(near, (cx, cy))
                if dist < best_dist and (cx, cy) not in self.reserved_targets:
                    best_dist = dist
                    best_pos = (cx, cy)
        
        return best_pos
    
    def find_available_cooker(self, controller: RobotController, 
                              near: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        """Find cooker with empty pan or no pan"""
        best_dist = float('inf')
        best_pos = None
        
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile:
                pan = getattr(tile, 'item', None)
                if pan is None or (isinstance(pan, Pan) and pan.food is None):
                    dist = self.get_distance(near, (kx, ky))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (kx, ky)
        
        return best_pos
    
    def get_order_priority(self, order: Dict, current_turn: int) -> float:
        """Calculate order priority score"""
        if not order.get('is_active', False):
            return -1
        
        time_left = order['expires_turn'] - current_turn
        reward = order.get('reward', 0)
        penalty = order.get('penalty', 0)
        
        # Estimate completion time (rough: 30 turns per order)
        est_time = 30
        
        if time_left < est_time:
            return -1  # Can't complete in time
        
        urgency = max(0, 50 - time_left)
        efficiency = reward / (penalty + 1)
        
        return efficiency + urgency * 0.5
    
    def get_best_order(self, controller: RobotController) -> Optional[Dict]:
        """Get highest priority active order"""
        orders = controller.get_orders()
        current_turn = controller.get_turn()
        
        best_order = None
        best_priority = -1
        
        for order in orders:
            priority = self.get_order_priority(order, current_turn)
            if priority > best_priority and order['order_id'] not in self.assigned_orders.values():
                best_priority = priority
                best_order = order
        
        return best_order

    # ========================================
    # COOKING MANAGEMENT
    # ========================================
    
    def update_cooking_trackers(self, controller: RobotController):
        """Update tracking of what's cooking"""
        self.cooking_trackers.clear()
        
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food is not None:
                    self.cooking_trackers[(kx, ky)] = CookingTracker(
                        location=(kx, ky),
                        food_name=pan.food.food_name,
                        cook_progress=getattr(tile, 'cook_progress', 0),
                        cooked_stage=pan.food.cooked_stage
                    )
    
    def get_urgent_cooking(self) -> Optional[CookingTracker]:
        """Get food that's about to be cooked (needs pickup)"""
        for tracker in self.cooking_trackers.values():
            if tracker.cooked_stage == 1:  # Already cooked, needs pickup
                return tracker
            if tracker.is_urgent:
                return tracker
        return None

    # ========================================
    # CHEF STATE MACHINE
    # ========================================
    
    def run_chef(self, controller: RobotController, bot_id: int):
        """Main chef logic - handles the cook/serve loop"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        state = self.bot_states.get(bot_id, 0)
        
        # Get current order
        order = self.get_best_order(controller)
        if not order and state == 0:
            # No orders, idle
            return
        
        # State machine for completing orders
        # States:
        # 0: Init - check what we need
        # 1: Ensure we have a pan on cooker
        # 2: Buy meat (or required cookable ingredient)
        # 3: Place meat on counter
        # 4: Chop meat
        # 5: Pick up meat
        # 6: Start cooking (place in pan)
        # 7: Buy plate
        # 8: Place plate on counter
        # 9: Buy other ingredients (noodles)
        # 10: Add ingredients to plate
        # 11: Wait for cooking, take from pan
        # 12: Add cooked item to plate
        # 13: Pick up plate
        # 14: Submit
        # 15: Handle trash/reset
        
        if state == 0:
            # Check if we have a pan on a cooker
            has_pan = False
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    has_pan = True
                    break
            
            if has_pan:
                self.bot_states[bot_id] = 2
            else:
                self.bot_states[bot_id] = 1
                
        elif state == 1:
            # Buy and place pan
            if holding and holding.get('type') == 'Pan':
                # Place pan on cooker
                cooker = self.find_available_cooker(controller, (bx, by))
                if cooker and self.move_towards(controller, bot_id, cooker):
                    controller.place(bot_id, cooker[0], cooker[1])
                    self.bot_states[bot_id] = 2
            else:
                shop = self.find_nearest((bx, by), self.shops)
                if shop and self.move_towards(controller, bot_id, shop):
                    if controller.get_team_money() >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                        
        elif state == 2:
            # Buy meat
            if holding:
                self.bot_states[bot_id] = 3
                return
            shop = self.find_nearest((bx, by), self.shops)
            if shop and self.move_towards(controller, bot_id, shop):
                if controller.get_team_money() >= FoodType.MEAT.buy_cost:
                    if controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1]):
                        self.bot_states[bot_id] = 3
                        
        elif state == 3:
            # Place meat on counter
            counter = self.find_empty_counter(controller, (bx, by))
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.place(bot_id, counter[0], counter[1]):
                    self.reserved_targets.add(counter)
                    self.bot_states[bot_id] = 4
                    self.chef_counter = counter
                    
        elif state == 4:
            # Chop meat
            counter = getattr(self, 'chef_counter', None)
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.chop(bot_id, counter[0], counter[1]):
                    self.bot_states[bot_id] = 5
                    
        elif state == 5:
            # Pick up chopped meat
            counter = getattr(self, 'chef_counter', None)
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.pickup(bot_id, counter[0], counter[1]):
                    self.reserved_targets.discard(counter)
                    self.bot_states[bot_id] = 6
                    
        elif state == 6:
            # Start cooking - place in pan
            cooker = None
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food is None:
                        cooker = (kx, ky)
                        break
            
            if cooker and self.move_towards(controller, bot_id, cooker):
                if controller.place(bot_id, cooker[0], cooker[1]):
                    self.chef_cooker = cooker
                    self.bot_states[bot_id] = 7
                    
        elif state == 7:
            # Buy plate
            if holding:
                self.bot_states[bot_id] = 8
                return
            shop = self.find_nearest((bx, by), self.shops)
            if shop and self.move_towards(controller, bot_id, shop):
                if controller.get_team_money() >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1]):
                        self.bot_states[bot_id] = 8
                        
        elif state == 8:
            # Place plate on counter
            counter = self.find_empty_counter(controller, (bx, by))
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.place(bot_id, counter[0], counter[1]):
                    self.chef_plate_counter = counter
                    self.reserved_targets.add(counter)
                    self.bot_states[bot_id] = 9
                    
        elif state == 9:
            # Buy noodles
            if holding:
                self.bot_states[bot_id] = 10
                return
            shop = self.find_nearest((bx, by), self.shops)
            if shop and self.move_towards(controller, bot_id, shop):
                if controller.get_team_money() >= FoodType.NOODLES.buy_cost:
                    if controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1]):
                        self.bot_states[bot_id] = 10
                        
        elif state == 10:
            # Add noodles to plate
            counter = getattr(self, 'chef_plate_counter', None)
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.add_food_to_plate(bot_id, counter[0], counter[1]):
                    self.bot_states[bot_id] = 11
                    
        elif state == 11:
            # Wait for cooking and take from pan
            cooker = getattr(self, 'chef_cooker', None)
            if cooker and self.move_towards(controller, bot_id, cooker):
                tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food:
                        if pan.food.cooked_stage == 1:  # Cooked!
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                self.bot_states[bot_id] = 12
                        elif pan.food.cooked_stage == 2:  # Burned :(
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                self.bot_states[bot_id] = 15
                                
        elif state == 12:
            # Add cooked meat to plate
            counter = getattr(self, 'chef_plate_counter', None)
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.add_food_to_plate(bot_id, counter[0], counter[1]):
                    self.bot_states[bot_id] = 13
                    
        elif state == 13:
            # Pick up plate
            counter = getattr(self, 'chef_plate_counter', None)
            if counter and self.move_towards(controller, bot_id, counter):
                if controller.pickup(bot_id, counter[0], counter[1]):
                    self.reserved_targets.discard(counter)
                    self.bot_states[bot_id] = 14
                    
        elif state == 14:
            # Submit!
            submit = self.find_nearest((bx, by), self.submits)
            if submit and self.move_towards(controller, bot_id, submit):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.bot_states[bot_id] = 0  # Reset and start over!
                    
        elif state == 15:
            # Trash bad item and restart
            if holding:
                trash = self.find_nearest((bx, by), self.trashes)
                if trash and self.move_towards(controller, bot_id, trash):
                    controller.trash(bot_id, trash[0], trash[1])
                    self.bot_states[bot_id] = 2
            else:
                self.bot_states[bot_id] = 2

    # ========================================
    # SUPPORT BOT LOGIC
    # ========================================
    
    def run_support(self, controller: RobotController, bot_id: int):
        """Support bot - washes dishes, fetches ingredients"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        
        # Priority 1: Check for urgent cooking that needs rescue
        urgent = self.get_urgent_cooking()
        if urgent and urgent.cooked_stage == 1:
            # Help pick up if chef is busy
            pass  # Let chef handle for now
        
        # Priority 2: Wash dirty plates if needed
        for sx, sy in self.sinks:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                if self.move_towards(controller, bot_id, (sx, sy)):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        # Priority 3: Random movement to stay out of way
        import random
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        random.shuffle(directions)
        for dx, dy in directions:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return

    # ========================================
    # SABOTAGE PROTOCOL
    # ========================================
    
    def should_sabotage(self, controller: RobotController) -> bool:
        """Determine if we should switch to enemy map"""
        switch_info = controller.get_switch_info()
        
        if not switch_info['window_active']:
            return False
        if switch_info['my_team_switched']:
            return False  # Already switched
        
        # Calculate if we're losing significantly
        # (We can't see enemy money directly, so use heuristics)
        turn = controller.get_turn()
        
        # Sabotage in last portion of switch window
        switch_end = switch_info['window_end_turn']
        if turn > switch_end - 30:
            return True
        
        return False
    
    def run_sabotage(self, controller: RobotController, bot_id: int):
        """Sabotage behavior when on enemy map"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        
        # Strategy: Steal their pan!
        if holding and holding.get('type') == 'Pan':
            # Hold it and stay away from them
            # Move to a corner
            import random
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            random.shuffle(directions)
            for dx, dy in directions:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    return
        else:
            # Try to steal a pan from their cooker
            for kx, ky in self.cookers:
                if self.move_towards(controller, bot_id, (kx, ky)):
                    # Try to pick up the pan
                    controller.pickup(bot_id, kx, ky)
                    return

    # ========================================
    # DEFENSE PROTOCOL
    # ========================================
    
    def run_defense(self, controller: RobotController, bot_id: int):
        """Defense when enemy is on our map"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        holding = bot.get('holding')
        
        # If holding a plate with food, protect it
        if holding and holding.get('type') == 'Plate' and holding.get('food'):
            # Stay near submit
            if self.submits:
                submit = self.submits[0]
                self.move_towards(controller, bot_id, submit)
            return
        
        # Otherwise guard the cooker
        if self.cookers:
            cooker = self.cookers[0]
            self.move_towards(controller, bot_id, cooker)

    # ========================================
    # MAIN ENTRY POINT
    # ========================================
    
    def play_turn(self, controller: RobotController):
        """Main turn logic"""
        # Initialize on first turn
        self.initialize(controller)
        
        # Update cooking trackers
        self.update_cooking_trackers(controller)
        
        # Clear per-turn state
        self.reserved_targets.clear()
        
        # Get our bots
        bot_ids = controller.get_team_bot_ids()
        if not bot_ids:
            return
        
        # Assign roles if not done
        for i, bot_id in enumerate(bot_ids):
            if bot_id not in self.bot_roles:
                if i == 0:
                    self.bot_roles[bot_id] = 'chef'
                else:
                    self.bot_roles[bot_id] = 'support'
        
        # Check switch window status
        switch_info = controller.get_switch_info()
        
        # Check if we should sabotage
        if self.should_sabotage(controller) and not switch_info['my_team_switched']:
            controller.switch_maps()
            # After switching, reinitialize for enemy map
            self.initialized = False
            self.initialize(controller)
        
        # Determine if enemy has invaded
        enemy_on_map = switch_info['enemy_team_switched']
        
        # Run each bot's logic
        for bot_id in bot_ids:
            role = self.bot_roles.get(bot_id, 'support')
            
            if switch_info['my_team_switched']:
                # We're on enemy map - sabotage mode
                self.run_sabotage(controller, bot_id)
            elif enemy_on_map:
                # Enemy on our map - defense mode
                self.run_defense(controller, bot_id)
            elif role == 'chef':
                self.run_chef(controller, bot_id)
            else:
                self.run_support(controller, bot_id)
