"""
JuniorChampion Bot - AWAP 2026
==============================
Recipe-based execution with cook-first parallelism.
Inspired by test.py architecture.
"""
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Any

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


FOOD_MAP = {
    "EGG": FoodType.EGG, "ONIONS": FoodType.ONIONS,
    "MEAT": FoodType.MEAT, "NOODLES": FoodType.NOODLES,
    "SAUCE": FoodType.SAUCE,
}


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.w = map_copy.width
        self.h = map_copy.height
        self.team = None
        
        # Build tile locations and walkable set
        self.tile_locs: Dict[str, List[Tuple[int, int]]] = {}
        self.walkable: Set[Tuple[int, int]] = set()
        for x in range(self.w):
            for y in range(self.h):
                t = map_copy.tiles[x][y]
                self.tile_locs.setdefault(t.tile_name, []).append((x, y))
                if t.is_walkable:
                    self.walkable.add((x, y))
        
        # Precompute BFS
        self._dist: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._next: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        self._precompute_bfs()
        
        # Key locations
        self._shops = self.tile_locs.get('SHOP', [])
        self._submits = self.tile_locs.get('SUBMIT', [])
        self._counters = self.tile_locs.get('COUNTER', [])
        self._cookers = self.tile_locs.get('COOKER', [])
        self._trash = self.tile_locs.get('TRASH', [])
        
        # Analyze map for feasibility multiplier
        self._analyze_map()
        
        # Task tracking
        self.tasks: Dict[int, Dict] = {}
        self.assigned_orders: Set[int] = set()
        self.completed_orders: Set[int] = set()
        self.failed_orders: Dict[int, int] = {}
    
    def _precompute_bfs(self):
        """BFS from every walkable tile"""
        for src in self.walkable:
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
                        if (nx, ny) in self.walkable and (nx, ny) not in dist:
                            dist[(nx, ny)] = dist[(cx, cy)] + 1
                            first_step[(nx, ny)] = first_step[(cx, cy)] if (cx, cy) != src else (dx, dy)
                            q.append((nx, ny))
            self._dist[src] = dist
            self._next[src] = first_step
    
    def _adj_walk(self, tx: int, ty: int) -> List[Tuple[int, int]]:
        """Get walkable tiles adjacent to (tx, ty)"""
        result = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pos = (tx + dx, ty + dy)
                if pos in self.walkable:
                    result.append(pos)
        return result
    
    def _dist_to_tile(self, sx: int, sy: int, tx: int, ty: int) -> int:
        """Distance from (sx,sy) to nearest walkable adjacent to (tx,ty)"""
        adj = self._adj_walk(tx, ty)
        if not adj:
            return 9999
        src = (sx, sy)
        if src not in self._dist:
            return 9999
        return min((self._dist[src].get(a, 9999) for a in adj), default=9999)
    
    def _nearest(self, bx: int, by: int, tile_type: str) -> Optional[Tuple[int, int]]:
        """Find nearest tile of type"""
        locs = self.tile_locs.get(tile_type, [])
        if not locs:
            return None
        return min(locs, key=lambda p: self._dist_to_tile(bx, by, p[0], p[1]))
    
    def _analyze_map(self):
        """Analyze map to set feasibility multiplier"""
        self._feasibility = 1.4
        self._min_tleft = 10
        
        if not self._shops or not self._submits:
            return
        
        # Calculate average infrastructure distance
        shop = self._shops[0]
        submit = self._submits[0]
        
        shop_adj = self._adj_walk(shop[0], shop[1])
        submit_adj = self._adj_walk(submit[0], submit[1])
        
        if shop_adj and submit_adj:
            d_shop_submit = min(
                self._dist.get(sa, {}).get(sb, 9999)
                for sa in shop_adj for sb in submit_adj
            )
        else:
            d_shop_submit = 9999
        
        # Compact maps
        if d_shop_submit <= 6:
            self._feasibility = 2.0
            self._min_tleft = 5
        elif d_shop_submit <= 12:
            self._feasibility = 1.6
            self._min_tleft = 8
        else:
            self._feasibility = 1.3
            self._min_tleft = 12
        
        # Adjust for resources
        if len(self._cookers) >= 3:
            self._feasibility *= 1.1
        elif len(self._cookers) <= 1:
            self._feasibility *= 0.9
    
    # ================================================================
    # RECIPE GENERATION
    # ================================================================
    
    def _make_recipe(self, order: Dict, c: RobotController, bid: int) -> Optional[List[tuple]]:
        """Generate recipe with cook-first parallelism"""
        bs = c.get_bot_state(bid)
        if not bs:
            return None
        bx, by = bs['x'], bs['y']
        team = c.get_team()
        
        shop = self._nearest(bx, by, 'SHOP')
        submit = self._nearest(bx, by, 'SUBMIT')
        if not shop or not submit:
            return None
        
        # Find assembly point (walkable near shop, closest to submit)
        shop_adj = self._adj_walk(shop[0], shop[1])
        claimed = self._get_claimed(bid)
        free_adj = [a for a in shop_adj if a not in claimed]
        if not free_adj:
            free_adj = shop_adj
        if not free_adj:
            return None
        
        assembly = min(free_adj, key=lambda a: self._dist_to_tile(a[0], a[1], submit[0], submit[1]))
        
        # Parse ingredients
        ingredients = [FOOD_MAP[fn] for fn in order['required'] if fn in FOOD_MAP]
        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]  # MEAT
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]  # EGG
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]  # ONIONS
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]  # NOODLES, SAUCE
        all_cook = cook_chop + cook_only
        needs_chop = bool(cook_chop) or bool(chop_only)
        
        # Find cooker
        cooker = None
        if all_cook:
            free_k = [k for k in self._cookers if k not in claimed]
            if free_k:
                cooker = min(free_k, key=lambda k: self._dist_to_tile(assembly[0], assembly[1], k[0], k[1]))
            elif self._cookers:
                cooker = min(self._cookers, key=lambda k: self._dist_to_tile(assembly[0], assembly[1], k[0], k[1]))
            if not cooker:
                return None
        
        # Find counter for chopping
        chop_c = None
        if needs_chop:
            free_c = [ct for ct in self._counters if ct not in claimed]
            if free_c:
                chop_c = min(free_c, key=lambda ct: self._dist_to_tile(assembly[0], assembly[1], ct[0], ct[1]))
            elif self._counters:
                chop_c = self._counters[0]
            if not chop_c:
                return None
        
        trash = self._nearest(bx, by, 'TRASH')
        steps: List[tuple] = []
        
        # Note: resources (assembly, cooker, chop) will be stored when order is actually assigned
        # Don't store them here during scoring phase
        
        # Go to assembly
        if assembly in self.walkable:
            steps.append(('goto', assembly))
        
        # Cleanup assembly if occupied
        tile_a = c.get_tile(team, assembly[0], assembly[1])
        if tile_a and getattr(tile_a, 'item', None) is not None:
            steps.append(('pickup', assembly))
            if trash:
                steps.append(('trash', trash))
                if assembly in self.walkable:
                    steps.append(('goto', assembly))
        
        # Calculate if cook-first is safe (won't burn)
        BURN_SAFE = GameConstants.COOK_PROGRESS * 2 - 5  # 35 turns
        
        d_cooker = 0
        if cooker:
            d_cooker = self._dist_to_tile(assembly[0], assembly[1], cooker[0], cooker[1])
        
        d_chop = 0
        if chop_c:
            d_chop = self._dist_to_tile(assembly[0], assembly[1], chop_c[0], chop_c[1])
        
        # Time away from cooker after placing first cook item
        parallel_work = (d_cooker +  # return to assembly
                        2 +  # buy plate + place
                        len(simple) * 2 +  # buy + add each
                        len(chop_only) * (2 * d_chop + 4) +  # chop round trips
                        d_cooker + 1)  # return to cooker + take
        
        cook_first = bool(all_cook) and bool(cooker) and parallel_work <= BURN_SAFE
        
        if cook_first:
            # === COOK-FIRST PARALLELISM ===
            fc = all_cook[0]
            steps.append(('buy', fc, shop))
            if fc.can_chop and chop_c:
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
            steps.append(('place_cook', cooker))
            if assembly in self.walkable:
                steps.append(('goto', assembly))
            
            # Buy plate, place
            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', assembly))
            
            # Simple items
            for ft in simple:
                steps.append(('buy', ft, shop))
                steps.append(('add_plate', assembly))
            
            # Chop-only items
            for ft in chop_only:
                steps.append(('buy', ft, shop))
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
                if assembly in self.walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
            
            # Collect first cooked item
            steps.append(('wait_take', cooker))
            if assembly in self.walkable:
                steps.append(('goto', assembly))
            steps.append(('add_plate', assembly))
            
            # Additional cook items (sequential)
            for ft in all_cook[1:]:
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                if assembly in self.walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
        else:
            # === SEQUENTIAL MODE ===
            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', assembly))
            
            for ft in simple:
                steps.append(('buy', ft, shop))
                steps.append(('add_plate', assembly))
            
            for ft in chop_only:
                steps.append(('buy', ft, shop))
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
                if assembly in self.walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
            
            for ft in all_cook:
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                if assembly in self.walkable:
                    steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
        
        # Pickup and submit
        steps.append(('pickup', assembly))
        steps.append(('submit', submit))
        
        return steps
    
    def _recipe_turns(self, recipe: List[tuple], bx: int, by: int) -> int:
        """Estimate turns to execute recipe"""
        x, y = bx, by
        turns = 0
        cook_start: Dict[Tuple[int, int], int] = {}
        
        for step in recipe:
            action = step[0]
            
            if action == 'goto':
                loc = step[1]
                if (x, y) != loc:
                    src = (x, y)
                    if src in self._dist and loc in self._dist[src]:
                        turns += self._dist[src][loc]
                        x, y = loc
            
            elif action == 'wait_take':
                loc = step[1]
                d = self._dist_to_tile(x, y, loc[0], loc[1])
                turns += d
                placed = cook_start.get(loc, -1)
                if placed >= 0:
                    elapsed = turns - placed
                    remaining = max(0, GameConstants.COOK_PROGRESS - elapsed)
                    turns += remaining
                else:
                    turns += GameConstants.COOK_PROGRESS
                turns += 1
                # After taking from cooker, find closest adjacent tile
                adj = self._adj_walk(loc[0], loc[1])
                if adj:
                    src = (x, y)
                    if src in self._dist:
                        x, y = min(adj, key=lambda a: self._dist[src].get(a, 9999))
            
            elif action == 'place_cook':
                loc = step[1]
                d = self._dist_to_tile(x, y, loc[0], loc[1])
                turns += max(d, 1)
                cook_start[loc] = turns
                # After placing at cooker, find closest adjacent tile
                adj = self._adj_walk(loc[0], loc[1])
                if adj:
                    src = (x, y)
                    if src in self._dist:
                        x, y = min(adj, key=lambda a: self._dist[src].get(a, 9999))
            
            else:  # buy, place, chop, pickup, add_plate, submit, trash
                loc = step[2] if action == 'buy' else step[1]
                d = self._dist_to_tile(x, y, loc[0], loc[1])
                turns += max(d, 1)
                # After action, bot is at closest adjacent tile (or stays if already there)
                adj = self._adj_walk(loc[0], loc[1])
                if adj:
                    if (x, y) in adj:
                        pass  # Already adjacent, stay in place
                    else:
                        src = (x, y)
                        if src in self._dist:
                            x, y = min(adj, key=lambda a: self._dist[src].get(a, 9999))
        
        return turns
    
    # ================================================================
    # ORDER EVALUATION
    # ================================================================
    
    def _score_order(self, order: Dict, bx: int, by: int, turn: int, c: RobotController, bid: int) -> float:
        """Score an order for selection"""
        if order['order_id'] in self.assigned_orders:
            return -9999
        if not order['is_active']:
            return -9999
        
        # Check failed cooldown
        if order['order_id'] in self.failed_orders:
            if turn - self.failed_orders[order['order_id']] < 20:
                return -9999
        
        # Calculate cost/profit
        ingredients = [FOOD_MAP[fn] for fn in order['required'] if fn in FOOD_MAP]
        cost = ShopCosts.PLATE.buy_cost + sum(f.buy_cost for f in ingredients)
        profit = order['reward'] - cost
        penalty = order.get('penalty', 0)
        
        # Only take orders with positive profit
        # (Negative profit orders are traps - even if penalty is high,
        # completing them loses money which may be worse than paying penalty
        # when factoring in opportunity cost)
        if profit <= 0:
            return -9999
        
        # Time check
        tleft = order['expires_turn'] - turn
        if tleft < self._min_tleft:
            return -9999
        
        # Try to make recipe and estimate time
        recipe = self._make_recipe(order, c, bid)
        if not recipe:
            return -9999
        
        est = self._recipe_turns(recipe, bx, by)
        if est >= 9999:
            return -9999
        
        # Feasibility check - be conservative
        # For short orders, need strict buffer; for long orders, can be more relaxed
        if tleft < 60:
            # Short duration orders: need estimate < time_left - 15
            if est > tleft - 15:
                return -9999
        else:
            # Long duration orders: use multiplier
            if est > tleft * self._feasibility:
                return -9999
        
        # Score = efficiency (value per turn)
        value = profit + penalty
        efficiency = value / max(est, 1)
        
        return efficiency
    
    def _get_claimed(self, exclude_bid: int) -> Set[Tuple[int, int]]:
        """Get resources claimed by other bots"""
        claimed = set()
        for bid, t in self.tasks.items():
            if bid == exclude_bid:
                continue
            for key in ('assembly', 'cooker', 'chop'):
                v = t.get(key)
                if v:
                    claimed.add(v)
        return claimed
    
    # ================================================================
    # EXECUTION
    # ================================================================
    
    def _execute_step(self, c: RobotController, bid: int, step: tuple) -> bool:
        """Execute one step. Returns True if completed."""
        action = step[0]
        bs = c.get_bot_state(bid)
        if not bs:
            return False
        bx, by = bs['x'], bs['y']
        
        if action == 'goto':
            target = step[1]
            if (bx, by) == target:
                return True
            src = (bx, by)
            if src in self._next and target in self._next[src]:
                dx, dy = self._next[src][target]
                if c.can_move(bid, dx, dy):
                    c.move(bid, dx, dy)
            return False
        
        elif action == 'buy':
            item, shop = step[1], step[2]
            d = self._dist_to_tile(bx, by, shop[0], shop[1])
            if d > 0:
                self._move_toward(c, bid, shop[0], shop[1])
                return False
            return c.buy(bid, item, shop[0], shop[1])
        
        elif action == 'place':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            return c.place(bid, target[0], target[1])
        
        elif action == 'pickup':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            return c.pickup(bid, target[0], target[1])
        
        elif action == 'chop':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            return c.chop(bid, target[0], target[1])
        
        elif action == 'place_cook':
            cooker = step[1]
            d = self._dist_to_tile(bx, by, cooker[0], cooker[1])
            if d > 0:
                self._move_toward(c, bid, cooker[0], cooker[1])
                return False
            if c.start_cook(bid, cooker[0], cooker[1]):
                return True
            return c.place(bid, cooker[0], cooker[1])
        
        elif action == 'wait_take':
            cooker = step[1]
            d = self._dist_to_tile(bx, by, cooker[0], cooker[1])
            if d > 0:
                self._move_toward(c, bid, cooker[0], cooker[1])
                return False
            team = c.get_team()
            tile = c.get_tile(team, cooker[0], cooker[1])
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food:
                    if pan.food.cooked_stage >= 1:
                        return c.take_from_pan(bid, cooker[0], cooker[1])
            return False
        
        elif action == 'add_plate':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            return c.add_food_to_plate(bid, target[0], target[1])
        
        elif action == 'trash':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            return c.trash(bid, target[0], target[1])
        
        elif action == 'submit':
            target = step[1]
            d = self._dist_to_tile(bx, by, target[0], target[1])
            if d > 0:
                self._move_toward(c, bid, target[0], target[1])
                return False
            
            # For future orders, wait until order is active before submitting
            t = self.tasks.get(bid, {})
            oid = t.get('order_id')
            if oid:
                orders = c.get_orders(c.get_team())
                order = next((o for o in orders if o['order_id'] == oid), None)
                if order and not order['is_active']:
                    # Order not active yet, wait
                    return False
            
            return c.submit(bid, target[0], target[1])
        
        return False
    
    def _move_toward(self, c: RobotController, bid: int, tx: int, ty: int):
        """Move bot toward target tile"""
        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']
        
        adj = self._adj_walk(tx, ty)
        if not adj:
            return
        
        src = (bx, by)
        if src not in self._dist:
            return
        
        # Find closest adjacent tile
        best = min(adj, key=lambda a: self._dist[src].get(a, 9999))
        if best not in self._next.get(src, {}):
            # Fallback: try any move that gets closer
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (dx, dy) != (0, 0) and c.can_move(bid, dx, dy):
                        c.move(bid, dx, dy)
                        return
            return
        
        dx, dy = self._next[src][best]
        if c.can_move(bid, dx, dy):
            c.move(bid, dx, dy)
    
    # ================================================================
    # MAIN TURN
    # ================================================================
    
    def play_turn(self, c: RobotController):
        turn = c.get_turn()
        team = c.get_team()
        if self.team is None:
            self.team = team
        
        bots = c.get_team_bot_ids(team)
        orders = c.get_orders(team)
        
        # Initialize tasks
        for bid in bots:
            if bid not in self.tasks:
                self.tasks[bid] = {
                    'recipe': [], 'step': 0, 'order_id': None,
                    'assembly': None, 'cooker': None, 'chop': None,
                    'stuck': 0, 'last_progress': turn
                }
        
        # Clean up expired assignments
        # Keep orders that are active OR not yet active (future orders being prepared)
        valid_ids = set()
        for o in orders:
            if o['is_active']:
                valid_ids.add(o['order_id'])
            elif o['created_turn'] > turn:  # Future order not yet started
                valid_ids.add(o['order_id'])
        
        for bid in bots:
            t = self.tasks[bid]
            oid = t.get('order_id')
            if oid and oid not in valid_ids:
                self._abort_task(bid, turn)
        
        # Detect stuck bots
        for bid in bots:
            t = self.tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                no_progress = turn - t.get('last_progress', turn)
                if t['stuck'] > 15 or no_progress > 40:
                    self._abort_task(bid, turn)
        
        # Handle idle bots with items
        idle_bots = []
        for bid in bots:
            t = self.tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                trash = self._nearest(bs['x'], bs['y'], 'TRASH')
                if trash:
                    t['recipe'] = [('trash', trash)]
                    t['step'] = 0
                    t['order_id'] = None
                    t['stuck'] = 0
                    t['last_progress'] = turn
                    continue
            
            idle_bots.append(bid)
        
        # Assign orders to idle bots
        for bid in idle_bots:
            bs = c.get_bot_state(bid)
            if not bs:
                continue
            bx, by = bs['x'], bs['y']
            
            # Score active orders
            scored = []
            for o in orders:
                score = self._score_order(o, bx, by, turn, c, bid)
                if score > -9999:
                    scored.append((score, o))
            
            # Also consider future orders (not yet active but highly profitable)
            for o in orders:
                if o['is_active'] or o['order_id'] in self.assigned_orders:
                    continue
                if o.get('completed_turn') is not None:
                    continue
                
                # Check if future order is worth preparing
                created = o['created_turn']
                wait_turns = created - turn
                if wait_turns <= 0 or wait_turns > 50:
                    continue  # Too far or already past
                
                # Only prepare high-value future orders
                ingredients = [FOOD_MAP[fn] for fn in o['required'] if fn in FOOD_MAP]
                cost = ShopCosts.PLATE.buy_cost + sum(f.buy_cost for f in ingredients)
                profit = o['reward'] - cost
                
                # Must be very profitable to prepare early
                if profit < 500:
                    continue
                
                # Check if we can complete in time
                recipe = self._make_recipe(o, c, bid)
                if not recipe:
                    continue
                est = self._recipe_turns(recipe, bx, by)
                if est >= 9999:
                    continue
                
                # Need recipe time + wait_turns to fit in order duration
                duration = o['expires_turn'] - o['created_turn']
                # Start early enough that we can submit soon after order activates
                if est > wait_turns + duration * 0.9:
                    continue
                
                # High priority for valuable future orders
                score = profit / max(est, 1) * 2  # Double weight for future orders
                scored.append((score, o))
            
            if not scored:
                continue
            
            scored.sort(key=lambda x: -x[0])
            best_order = scored[0][1]
            
            # Generate recipe
            recipe = self._make_recipe(best_order, c, bid)
            if not recipe:
                continue
            
            # Assign
            t = self.tasks[bid]
            t['recipe'] = recipe
            t['step'] = 0
            t['order_id'] = best_order['order_id']
            t['stuck'] = 0
            t['last_progress'] = turn
            
            # Extract resources from recipe
            t['assembly'] = None
            t['cooker'] = None
            t['chop'] = None
            for step in recipe:
                action = step[0]
                if action == 'goto' and t['assembly'] is None:
                    t['assembly'] = step[1]
                elif action == 'place_cook':
                    t['cooker'] = step[1]
                elif action == 'chop' and t['chop'] is None:
                    t['chop'] = step[1]
            
            self.assigned_orders.add(best_order['order_id'])
        
        # Execute all bots
        for bid in bots:
            self._execute_bot(c, bid, turn)
    
    def _execute_bot(self, c: RobotController, bid: int, turn: int):
        """Execute current task"""
        t = self.tasks[bid]
        
        if not t['recipe'] or t['step'] >= len(t['recipe']):
            return
        
        bs = c.get_bot_state(bid)
        if not bs:
            return
        
        step = t['recipe'][t['step']]
        completed = self._execute_step(c, bid, step)
        
        if completed:
            t['step'] += 1
            t['stuck'] = 0
            t['last_progress'] = turn
            
            if t['step'] >= len(t['recipe']):
                oid = t.get('order_id')
                if oid:
                    self.completed_orders.add(oid)
                    self.assigned_orders.discard(oid)
                self._reset_task(bid)
        else:
            # Don't count as stuck if waiting for cooking
            current_action = step[0] if step else None
            if current_action == 'wait_take':
                # Waiting for cook is expected, not stuck
                t['last_progress'] = turn  # Keep progress timer fresh
            else:
                old_pos = t.get('_last_pos')
                new_pos = (bs['x'], bs['y'])
                if old_pos == new_pos:
                    t['stuck'] = t.get('stuck', 0) + 1
                t['_last_pos'] = new_pos
    
    def _abort_task(self, bid: int, turn: int):
        """Abort current task"""
        t = self.tasks[bid]
        oid = t.get('order_id')
        if oid:
            self.assigned_orders.discard(oid)
            self.failed_orders[oid] = turn
        self._reset_task(bid)
    
    def _reset_task(self, bid: int):
        """Reset task"""
        self.tasks[bid] = {
            'recipe': [], 'step': 0, 'order_id': None,
            'assembly': None, 'cooker': None, 'chop': None,
            'stuck': 0, 'last_progress': 0
        }
