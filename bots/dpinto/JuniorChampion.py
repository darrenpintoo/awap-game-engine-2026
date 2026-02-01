"""
JuniorChampion Bot - AWAP 2026
A kitchen management bot with dual-worker coordination.
"""
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


# Ingredient database with processing requirements
INGREDIENTS = {
    "EGG": {"kind": FoodType.EGG, "needs_knife": False, "needs_heat": True, "price": 20},
    "ONIONS": {"kind": FoodType.ONIONS, "needs_knife": True, "needs_heat": False, "price": 30},
    "MEAT": {"kind": FoodType.MEAT, "needs_knife": True, "needs_heat": True, "price": 80},
    "NOODLES": {"kind": FoodType.NOODLES, "needs_knife": False, "needs_heat": False, "price": 40},
    "SAUCE": {"kind": FoodType.SAUCE, "needs_knife": False, "needs_heat": False, "price": 10},
}


class BotPlayer:
    """Kitchen management bot with chef/assistant worker roles."""
    
    def __init__(self, map_copy):
        self.grid_w = map_copy.width
        self.grid_h = map_copy.height
        self.grid_size = self.grid_w * self.grid_h
        
        # Scan map for locations and walkable tiles
        self.locations = {}
        self.floor_tiles = set()
        for col in range(self.grid_w):
            for row in range(self.grid_h):
                cell = map_copy.tiles[col][row]
                self.locations.setdefault(cell.tile_name, []).append((col, row))
                if cell.is_walkable:
                    self.floor_tiles.add((col, row))
        
        # Build pathfinding cache
        self._distances = {}
        self._directions = {}
        self._build_pathfinder()
        
        # Cache facility positions
        self._stores = self.locations.get('SHOP', [])
        self._delivery = self.locations.get('SUBMIT', [])
        self._surfaces = self.locations.get('COUNTER', [])
        self._stoves = self.locations.get('COOKER', [])
        self._bins = self.locations.get('TRASH', [])
        self._basins = self.locations.get('SINK', [])
        self._drying_racks = self.locations.get('SINKTABLE', [])
        
        # Designate key workstations
        self.plating_station = None
        self.prep_station = None
        self._designate_workstations()
        
        # Cache route metrics for scoring
        self.route_store_to_surface = None
        self.route_store_to_stove = None
        self.route_surface_to_delivery = None
        self._cache_route_metrics()
        
        # Bot job tracking
        self.bot_jobs = {}
        self.claimed_orders = set()
        self.finished_orders = set()
        self.cooldowns = {}
        self.my_team = None
    
    # ─────────────────────────────────────────────────────────────
    # Pathfinding System
    # ─────────────────────────────────────────────────────────────
    
    def _build_pathfinder(self):
        """Build BFS distance/direction cache from all walkable tiles."""
        for origin in self.floor_tiles:
            dist_map = {origin: 0}
            dir_map = {origin: (0, 0)}
            frontier = deque([origin])
            
            while frontier:
                cx, cy = frontier.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in self.floor_tiles and (nx, ny) not in dist_map:
                            dist_map[(nx, ny)] = dist_map[(cx, cy)] + 1
                            dir_map[(nx, ny)] = dir_map[(cx, cy)] if (cx, cy) != origin else (dx, dy)
                            frontier.append((nx, ny))
            
            self._distances[origin] = dist_map
            self._directions[origin] = dir_map
    
    def _neighbors(self, tx, ty):
        """Get walkable tiles adjacent to a facility tile."""
        adjacent = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos = (tx + dx, ty + dy)
                if pos in self.floor_tiles:
                    adjacent.append(pos)
        return adjacent
    
    def _path_cost(self, sx, sy, tx, ty):
        """Get minimum path cost from (sx,sy) to adjacent of (tx,ty)."""
        adjacent = self._neighbors(tx, ty)
        if not adjacent or (sx, sy) not in self._distances:
            return 9999
        return min((self._distances[(sx, sy)].get(a, 9999) for a in adjacent), default=9999)
    
    def _station_distance(self, pos_a, pos_b):
        """Distance between adjacent walkable tiles of two stations."""
        if pos_a is None or pos_b is None:
            return 9999
        adj_a = self._neighbors(pos_a[0], pos_a[1])
        adj_b = self._neighbors(pos_b[0], pos_b[1])
        if not adj_a or not adj_b:
            return 9999
        
        best = 9999
        for a in adj_a:
            if a not in self._distances:
                continue
            for b in adj_b:
                cost = self._distances[a].get(b, 9999)
                if cost < best:
                    best = cost
        return best
    
    def _closest_of_type(self, x, y, station_type):
        """Find nearest reachable station of given type."""
        candidates = self.locations.get(station_type, [])
        if not candidates:
            return None
        
        reachable = []
        for loc in candidates:
            cost = self._path_cost(x, y, loc[0], loc[1])
            if cost < 9999:
                reachable.append((loc, cost))
        
        if not reachable:
            return None
        return min(reachable, key=lambda item: item[1])[0]
    
    def _navigate(self, ctrl, bot_id, bot_state, target):
        """Move bot toward target tile, return True if adjacent."""
        bx, by = bot_state['x'], bot_state['y']
        tx, ty = target
        
        # Check if already adjacent
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return True
        
        adjacent = self._neighbors(tx, ty)
        if not adjacent:
            return False
        
        origin = (bx, by)
        if origin not in self._distances:
            return False
        
        # Find best adjacent tile to move toward
        goal = min(adjacent, key=lambda a: self._distances[origin].get(a, 9999))
        if goal not in self._directions.get(origin, {}):
            return False
        
        dx, dy = self._directions[origin][goal]
        if ctrl.can_move(bot_id, dx, dy):
            ctrl.move(bot_id, dx, dy)
        return False
    
    # ─────────────────────────────────────────────────────────────
    # Workstation Management
    # ─────────────────────────────────────────────────────────────
    
    def _designate_workstations(self):
        """Assign plating station near delivery, prep station near store/stoves."""
        if not self._surfaces:
            return
        
        store = self._stores[0] if self._stores else None
        delivery = self._delivery[0] if self._delivery else None
        stove = self._stoves[0] if self._stoves else None
        
        if len(self._surfaces) >= 2 and delivery:
            # Find best plating station (close to delivery)
            best_plating = None
            best_val = 9999
            for surface in self._surfaces:
                d_delivery = self._station_distance(surface, delivery)
                d_store = self._station_distance(surface, store) if store else 0
                val = d_delivery * 1.0 + d_store * 0.5
                if val < best_val:
                    best_val = val
                    best_plating = surface
            self.plating_station = best_plating
            
            # Find best prep station (close to store/stoves)
            best_prep = None
            best_val = 9999
            for surface in self._surfaces:
                if surface == self.plating_station:
                    continue
                d_store = self._station_distance(surface, store) if store else 0
                d_stove = self._station_distance(surface, stove) if stove else 0
                val = d_store * 1.0 + d_stove * 0.5
                if val < best_val:
                    best_val = val
                    best_prep = surface
            self.prep_station = best_prep
        elif self._surfaces:
            self.plating_station = self._surfaces[0]
            self.prep_station = self._surfaces[0] if len(self._surfaces) == 1 else self._surfaces[1]
    
    def _cache_route_metrics(self):
        """Pre-compute key route distances for order scoring."""
        store = self._stores[0] if self._stores else None
        delivery = self._delivery[0] if self._delivery else None
        
        if store and self._surfaces:
            dists = [self._station_distance(store, s) for s in self._surfaces]
            valid = [d for d in dists if d < 9999]
            self.route_store_to_surface = min(valid) if valid else None
        
        if store and self._stoves:
            dists = [self._station_distance(store, s) for s in self._stoves]
            valid = [d for d in dists if d < 9999]
            self.route_store_to_stove = min(valid) if valid else None
        
        if self._surfaces and delivery:
            dists = [self._station_distance(s, delivery) for s in self._surfaces]
            valid = [d for d in dists if d < 9999]
            self.route_surface_to_delivery = min(valid) if valid else None
    
    def _locate_free_surface(self, ctrl, x, y, exclude=None):
        """Find empty counter surface, preferring prep station."""
        exclude = exclude or []
        team = ctrl.get_team()
        
        # Try prep station first
        if self.prep_station and self.prep_station not in exclude:
            tile = ctrl.get_tile(team, self.prep_station[0], self.prep_station[1])
            if tile and getattr(tile, 'item', None) is None:
                cost = self._path_cost(x, y, self.prep_station[0], self.prep_station[1])
                if cost < 9999:
                    return self.prep_station
        
        # Search other surfaces
        for surface in self._surfaces:
            if surface in exclude:
                continue
            tile = ctrl.get_tile(team, surface[0], surface[1])
            if tile and getattr(tile, 'item', None) is None:
                cost = self._path_cost(x, y, surface[0], surface[1])
                if cost < 9999:
                    return surface
        return None
    
    def _locate_ready_stove(self, ctrl, x, y):
        """Find stove with empty pan that's reachable."""
        team = ctrl.get_team()
        best = None
        best_cost = 9999
        
        for stove in self._stoves:
            cost = self._path_cost(x, y, stove[0], stove[1])
            if cost >= 9999:
                continue
            tile = ctrl.get_tile(team, stove[0], stove[1])
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    if cost < best_cost:
                        best_cost = cost
                        best = stove
        
        # Fallback to nearest reachable stove
        if best is None:
            for stove in self._stoves:
                cost = self._path_cost(x, y, stove[0], stove[1])
                if cost < best_cost:
                    best_cost = cost
                    best = stove
        return best
    
    # ─────────────────────────────────────────────────────────────
    # Main Turn Logic
    # ─────────────────────────────────────────────────────────────
    
    def play_turn(self, ctrl: RobotController):
        turn_num = ctrl.get_turn()
        team = ctrl.get_team()
        if self.my_team is None:
            self.my_team = team
        
        bot_ids = ctrl.get_team_bot_ids(team)
        orders = ctrl.get_orders(team)
        
        # Initialize job tracking for each bot
        for bid in bot_ids:
            if bid not in self.bot_jobs:
                self.bot_jobs[bid] = {
                    'action_list': [], 'action_idx': 0,
                    'target_order': None, 'blocked_count': 0, 'worker_type': 'chef'
                }
        
        # Assign worker roles: first bot is chef, second is assistant
        if len(bot_ids) >= 2:
            self.bot_jobs[bot_ids[0]]['worker_type'] = 'chef'
            self.bot_jobs[bot_ids[1]]['worker_type'] = 'assistant'
        
        # Clean up expired order claims
        active_ids = {o['order_id'] for o in orders if o['is_active']}
        for bid, job in self.bot_jobs.items():
            if job['target_order'] and job['target_order'] not in active_ids:
                self.claimed_orders.discard(job['target_order'])
                job['action_list'] = []
                job['action_idx'] = 0
                job['target_order'] = None
        
        # Execute each bot's routine
        for bid in bot_ids:
            job = self.bot_jobs[bid]
            if job['worker_type'] == 'assistant':
                self._assistant_routine(ctrl, bid, turn_num, orders)
            else:
                self._chef_routine(ctrl, bid, turn_num, orders)
    
    # ─────────────────────────────────────────────────────────────
    # Worker Routines
    # ─────────────────────────────────────────────────────────────
    
    def _chef_routine(self, ctrl, bid, turn_num, orders):
        """Chef bot: fulfill customer orders."""
        job = self.bot_jobs[bid]
        bot_state = ctrl.get_bot_state(bid)
        if not bot_state:
            return
        
        # Continue existing action list
        if job['action_list'] and job['action_idx'] < len(job['action_list']):
            self._perform_step(ctrl, bid, turn_num)
            return
        
        # Clear hands before taking new order
        if bot_state.get('holding'):
            trash = self._closest_of_type(bot_state['x'], bot_state['y'], 'TRASH')
            if trash:
                job['action_list'] = [('discard', trash)]
                job['action_idx'] = 0
                job['target_order'] = None
            return
        
        # Select best available order
        best_order = None
        best_rating = -9999
        
        for order in orders:
            if not order['is_active']:
                continue
            if order['order_id'] in self.claimed_orders:
                continue
            if order['order_id'] in self.finished_orders:
                continue
            if order['order_id'] in self.cooldowns:
                if turn_num - self.cooldowns[order['order_id']] < 20:
                    continue
            
            rating = self._evaluate_order(order, bot_state['x'], bot_state['y'], turn_num)
            if rating > best_rating:
                best_rating = rating
                best_order = order
        
        if best_order:
            action_list = self._plan_order(ctrl, bid, best_order, bot_state)
            if action_list:
                job['action_list'] = action_list
                job['action_idx'] = 0
                job['target_order'] = best_order['order_id']
                self.claimed_orders.add(best_order['order_id'])
    
    def _assistant_routine(self, ctrl, bid, turn_num, orders):
        """Assistant bot: wash dishes, prepare plates, support chef."""
        bot_state = ctrl.get_bot_state(bid)
        if not bot_state:
            return
        
        team = ctrl.get_team()
        holding = bot_state.get('holding')
        holding_type = holding.get('type') if holding else None
        
        # If holding dirty plate, deposit in sink
        if holding and holding.get('type') == 'Plate' and holding.get('dirty'):
            basin = self._closest_of_type(bot_state['x'], bot_state['y'], 'SINK')
            if basin and self._navigate(ctrl, bid, bot_state, basin):
                ctrl.put_dirty_plate_in_sink(bid, basin[0], basin[1])
            return
        
        # Wash dirty dishes in sinks
        for basin in self._basins:
            cost = self._path_cost(bot_state['x'], bot_state['y'], basin[0], basin[1])
            if cost >= 9999:
                continue
            tile = ctrl.get_tile(team, basin[0], basin[1])
            if tile and hasattr(tile, 'num_dirty_plates') and tile.num_dirty_plates > 0:
                if self._navigate(ctrl, bid, bot_state, basin):
                    ctrl.wash_sink(bid, basin[0], basin[1])
                return
        
        # Pre-stage plate at plating station (larger maps only)
        if self.plating_station and self.grid_size >= 200:
            cost = self._path_cost(bot_state['x'], bot_state['y'],
                                   self.plating_station[0], self.plating_station[1])
            if cost < 9999:
                tile = ctrl.get_tile(team, self.plating_station[0], self.plating_station[1])
                plate_item = getattr(tile, 'item', None) if tile else None
                has_clean_plate = isinstance(plate_item, Plate) and not plate_item.dirty
                
                if not has_clean_plate:
                    # Place held clean plate
                    if holding_type == 'Plate' and not holding.get('dirty', True):
                        if self._navigate(ctrl, bid, bot_state, self.plating_station):
                            ctrl.place(bid, self.plating_station[0], self.plating_station[1])
                        return
                    
                    # Discard unwanted item
                    if holding:
                        trash = self._closest_of_type(bot_state['x'], bot_state['y'], 'TRASH')
                        if trash and self._navigate(ctrl, bid, bot_state, trash):
                            ctrl.trash(bid, trash[0], trash[1])
                        return
                    
                    # Grab clean plate from drying rack
                    for rack in self._drying_racks:
                        cost = self._path_cost(bot_state['x'], bot_state['y'], rack[0], rack[1])
                        if cost >= 9999:
                            continue
                        tile = ctrl.get_tile(team, rack[0], rack[1])
                        if tile and hasattr(tile, 'num_clean_plates') and tile.num_clean_plates > 0:
                            if self._navigate(ctrl, bid, bot_state, rack):
                                ctrl.take_clean_plate(bid, rack[0], rack[1])
                            return
                    
                    # Buy new plate if affordable
                    money = ctrl.get_team_money(team)
                    store = self._closest_of_type(bot_state['x'], bot_state['y'], 'SHOP')
                    if store and money >= ShopCosts.PLATE.buy_cost + 150:
                        if self._navigate(ctrl, bid, bot_state, store):
                            ctrl.buy(bid, ShopCosts.PLATE, store[0], store[1])
                        return
        
        # Idle near sink
        basin = self._closest_of_type(bot_state['x'], bot_state['y'], 'SINK')
        if basin:
            self._navigate(ctrl, bid, bot_state, basin)
    
    # ─────────────────────────────────────────────────────────────
    # Order Evaluation & Planning
    # ─────────────────────────────────────────────────────────────
    
    def _evaluate_order(self, order, x, y, turn_num):
        """Rate an order based on profit, time, and complexity."""
        required = order['required']
        
        # Calculate ingredient cost
        total_cost = ShopCosts.PLATE.buy_cost
        for item_name in required:
            info = INGREDIENTS.get(item_name, {})
            total_cost += info.get('price', 0)
        
        profit = order['reward'] - total_cost
        if profit <= 0:
            return -9999
        
        remaining_turns = order['expires_turn'] - turn_num
        if remaining_turns < 12:
            return -9999
        
        # Count processing requirements
        heat_count = sum(1 for n in required if INGREDIENTS.get(n, {}).get('needs_heat', False))
        knife_count = sum(1 for n in required if INGREDIENTS.get(n, {}).get('needs_knife', False))
        
        # Check stove reachability for cooking orders
        if heat_count > 0:
            stove_reachable = any(self._path_cost(x, y, s[0], s[1]) < 9999 for s in self._stoves)
            if not stove_reachable:
                return -9999
        
        # Estimate completion time
        base_time = 12
        per_ingredient = 3
        heat_time = 22
        knife_time = 4
        
        estimated = base_time + len(required) * per_ingredient + heat_count * heat_time + knife_count * knife_time
        
        # Add travel time
        travel_cost = 0
        if self.route_store_to_surface:
            travel_cost += self.route_store_to_surface * len(required)
        if self.route_store_to_stove and heat_count > 0:
            travel_cost += self.route_store_to_stove * heat_count
        if self.route_surface_to_delivery:
            travel_cost += self.route_surface_to_delivery
        
        # Scale travel by map size
        if self.grid_size >= 250:
            estimated += travel_cost * 0.8
        else:
            estimated += travel_cost * 0.4
        
        # Reject complex orders on maps with limited surfaces
        if len(self._surfaces) <= 1 and knife_count >= 2:
            return -9999
        if self.grid_size >= 300 and len(self._surfaces) <= 8:
            if len(required) >= 4:
                return -9999
            if len(required) == 3 and (knife_count + heat_count) >= 2:
                return -9999
        
        # Feasibility check
        if estimated > remaining_turns * 0.75:
            return -9999
        
        # Calculate final rating with urgency bonus
        rating = profit / max(estimated, 1)
        if remaining_turns < 80:
            rating += 0.5
        
        return rating
    
    def _plan_order(self, ctrl, bid, order, bot_state):
        """Generate action sequence to fulfill an order."""
        bx, by = bot_state['x'], bot_state['y']
        team = ctrl.get_team()
        
        store = self._closest_of_type(bx, by, 'SHOP')
        delivery = self._closest_of_type(bx, by, 'SUBMIT')
        if not store or not delivery:
            return None
        
        # Categorize ingredients by processing needs
        heat_and_knife = []
        heat_only = []
        knife_only = []
        ready_to_use = []
        
        for item_name in order['required']:
            info = INGREDIENTS.get(item_name, {})
            kind = info.get('kind')
            if not kind:
                continue
            
            needs_heat = info.get('needs_heat', False)
            needs_knife = info.get('needs_knife', False)
            
            if needs_heat and needs_knife:
                heat_and_knife.append((item_name, kind))
            elif needs_heat:
                heat_only.append((item_name, kind))
            elif needs_knife:
                knife_only.append((item_name, kind))
            else:
                ready_to_use.append((item_name, kind))
        
        heated_items = heat_and_knife + heat_only
        knifed_items = heat_and_knife + knife_only
        
        # Find workstations - must be reachable
        plate_loc = self.plating_station
        if plate_loc:
            cost = self._path_cost(bx, by, plate_loc[0], plate_loc[1])
            if cost >= 9999:
                plate_loc = None
        if not plate_loc:
            plate_loc = self._closest_of_type(bx, by, 'COUNTER')
        
        prep_loc = self.prep_station
        if prep_loc:
            cost = self._path_cost(bx, by, prep_loc[0], prep_loc[1])
            if cost >= 9999:
                prep_loc = None
        if not prep_loc and knifed_items:
            prep_loc = self._locate_free_surface(ctrl, bx, by, exclude=[plate_loc])
        
        stove_loc = self._locate_ready_stove(ctrl, bx, by) if heated_items else None
        
        if not plate_loc:
            return None
        if knifed_items and not prep_loc:
            prep_loc = plate_loc
        if heated_items and not stove_loc:
            return None
        
        actions = []
        
        # Check if plate already staged at plating station
        tile = ctrl.get_tile(team, plate_loc[0], plate_loc[1])
        plate_item = getattr(tile, 'item', None) if tile else None
        plate_staged = isinstance(plate_item, Plate) and not plate_item.dirty
        
        if not plate_staged:
            actions.append(('purchase', ShopCosts.PLATE, store))
            actions.append(('deposit', plate_loc))
        
        # Ensure pan on stove for heated items
        if heated_items and stove_loc:
            tile = ctrl.get_tile(team, stove_loc[0], stove_loc[1])
            pan_item = getattr(tile, 'item', None) if tile else None
            if not isinstance(pan_item, Pan):
                actions.append(('purchase', ShopCosts.PAN, store))
                actions.append(('deposit', stove_loc))
        
        # Ready-to-use ingredients
        for item_name, kind in ready_to_use:
            actions.append(('purchase', kind, store))
            actions.append(('plate_add', plate_loc))
        
        # Knife-only ingredients
        for item_name, kind in knife_only:
            actions.append(('purchase', kind, store))
            actions.append(('deposit', prep_loc))
            actions.append(('slice', prep_loc))
            actions.append(('grab', prep_loc))
            actions.append(('plate_add', plate_loc))
        
        # Heated ingredients
        for item_name, kind in heated_items:
            info = INGREDIENTS.get(item_name, {})
            actions.append(('purchase', kind, store))
            if info.get('needs_knife'):
                actions.append(('deposit', prep_loc))
                actions.append(('slice', prep_loc))
                actions.append(('grab', prep_loc))
            actions.append(('heat_start', stove_loc))
            actions.append(('heat_wait', stove_loc))
            actions.append(('plate_add', plate_loc))
        
        # Deliver order
        actions.append(('grab', plate_loc))
        actions.append(('deliver', delivery))
        
        return actions
    
    def _perform_step(self, ctrl, bid, turn_num):
        """Execute the current action in bot's action list."""
        job = self.bot_jobs[bid]
        if not job['action_list'] or job['action_idx'] >= len(job['action_list']):
            return
        
        bot_state = ctrl.get_bot_state(bid)
        if not bot_state:
            return
        
        current_action = job['action_list'][job['action_idx']]
        action_type = current_action[0]
        completed = False
        
        if action_type == 'goto':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = True
        
        elif action_type == 'purchase':
            item, store = current_action[1], current_action[2]
            if self._navigate(ctrl, bid, bot_state, store):
                completed = ctrl.buy(bid, item, store[0], store[1])
        
        elif action_type == 'deposit':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = ctrl.place(bid, target[0], target[1])
        
        elif action_type == 'grab':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = ctrl.pickup(bid, target[0], target[1])
        
        elif action_type == 'slice':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                ctrl.chop(bid, target[0], target[1])
                tile = ctrl.get_tile(ctrl.get_team(), target[0], target[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Food):
                    completed = tile.item.chopped
        
        elif action_type == 'heat_start':
            stove = current_action[1]
            if self._navigate(ctrl, bid, bot_state, stove):
                completed = ctrl.start_cook(bid, stove[0], stove[1]) or ctrl.place(bid, stove[0], stove[1])
        
        elif action_type == 'heat_wait':
            stove = current_action[1]
            if self._navigate(ctrl, bid, bot_state, stove):
                tile = ctrl.get_tile(ctrl.get_team(), stove[0], stove[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage >= 1:
                        completed = ctrl.take_from_pan(bid, stove[0], stove[1])
        
        elif action_type == 'plate_add':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = ctrl.add_food_to_plate(bid, target[0], target[1])
        
        elif action_type == 'deliver':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = ctrl.submit(bid, target[0], target[1])
        
        elif action_type == 'discard':
            target = current_action[1]
            if self._navigate(ctrl, bid, bot_state, target):
                completed = ctrl.trash(bid, target[0], target[1])
        
        if completed:
            job['action_idx'] += 1
            job['blocked_count'] = 0
            
            if job['action_idx'] >= len(job['action_list']):
                if job['target_order']:
                    self.finished_orders.add(job['target_order'])
                    self.claimed_orders.discard(job['target_order'])
                job['action_list'] = []
                job['action_idx'] = 0
                job['target_order'] = None
        else:
            pos = (bot_state['x'], bot_state['y'])
            if pos == job.get('_prev_pos'):
                job['blocked_count'] += 1
            else:
                job['blocked_count'] = 0
            job['_prev_pos'] = pos
            
            if job['blocked_count'] > 25:
                if job['target_order']:
                    self.claimed_orders.discard(job['target_order'])
                    self.cooldowns[job['target_order']] = turn_num
                job['action_list'] = []
                job['action_idx'] = 0
                job['target_order'] = None
