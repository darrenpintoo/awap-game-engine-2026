"""
APEX: Autonomous Prioritized EXecution Bot
AWAP 2026 Tournament - Utility AI with Goal-Oriented Action Planning

Architecture:
├── Navigator (A* pathfinding with caching)
├── Planner (goal decomposition into action sequences)
├── Evaluator (utility-based order scoring with urgency)
├── Executor (action dispatch with failure recovery)
└── Tactician (sabotage, recycling, defense)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque
from heapq import heappush, heappop

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


# =============================================================================
# CONSTANTS & MAPPINGS
# =============================================================================

FOOD_MAP = {
    "EGG": FoodType.EGG, "ONIONS": FoodType.ONIONS,
    "MEAT": FoodType.MEAT, "NOODLES": FoodType.NOODLES,
    "SAUCE": FoodType.SAUCE,
}


# =============================================================================
# NAVIGATOR - Pathfinding System
# =============================================================================

class Navigator:
    """Handles all spatial queries and pathfinding"""
    
    def __init__(self, game_map):
        self.w = game_map.width
        self.h = game_map.height
        self.walkable: Set[Tuple[int, int]] = set()
        self.tiles: Dict[str, List[Tuple[int, int]]] = {}
        
        # Scan map
        for x in range(self.w):
            for y in range(self.h):
                tile = game_map.tiles[x][y]
                self.tiles.setdefault(tile.tile_name, []).append((x, y))
                if tile.is_walkable:
                    self.walkable.add((x, y))
        
        # Precompute BFS from every walkable tile
        self._dist: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._next: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        self._precompute()
        
        # Tile accessors
        self.shops = self.tiles.get('SHOP', [])
        self.submits = self.tiles.get('SUBMIT', [])
        self.cookers = self.tiles.get('COOKER', [])
        self.counters = self.tiles.get('COUNTER', [])
        self.trash = self.tiles.get('TRASH', [])
        self.sinks = self.tiles.get('SINK', [])
        self.sink_tables = self.tiles.get('SINKTABLE', [])
    
    def _precompute(self):
        """BFS from every walkable source"""
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
    
    def adjacent_walkable(self, tx: int, ty: int) -> List[Tuple[int, int]]:
        """Get walkable tiles adjacent to target (including target if walkable)"""
        result = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pos = (tx + dx, ty + dy)
                if pos in self.walkable:
                    result.append(pos)
        return result
    
    def distance_to(self, sx: int, sy: int, tx: int, ty: int) -> int:
        """Distance from (sx,sy) to nearest tile adjacent to (tx,ty)"""
        adj = self.adjacent_walkable(tx, ty)
        if not adj:
            return 9999
        src = (sx, sy)
        if src not in self._dist:
            return 9999
        return min((self._dist[src].get(a, 9999) for a in adj), default=9999)
    
    def nearest(self, bx: int, by: int, tile_type: str) -> Optional[Tuple[int, int]]:
        """Find nearest tile of given type"""
        locs = self.tiles.get(tile_type, [])
        if not locs:
            return None
        return min(locs, key=lambda p: self.distance_to(bx, by, p[0], p[1]))
    
    def step_toward(self, sx: int, sy: int, tx: int, ty: int) -> Optional[Tuple[int, int]]:
        """Get (dx, dy) for first step toward target tile (for adjacent access)"""
        adj = self.adjacent_walkable(tx, ty)
        if not adj:
            return None
        src = (sx, sy)
        if src in adj:
            return None  # Already adjacent
        if src not in self._next:
            return None
        
        best_adj = min(adj, key=lambda a: self._dist[src].get(a, 9999))
        if best_adj not in self._next[src]:
            return None
        return self._next[src][best_adj]
    
    def step_onto(self, sx: int, sy: int, tx: int, ty: int) -> Optional[Tuple[int, int]]:
        """Get (dx, dy) for first step toward stepping ONTO target tile"""
        src = (sx, sy)
        target = (tx, ty)
        
        if src == target:
            return None  # Already there
        if target not in self.walkable:
            return None  # Can't step onto non-walkable
        if src not in self._next:
            return None
        if target not in self._next[src]:
            return None
        
        return self._next[src][target]


# =============================================================================
# EVALUATOR - Order Scoring System
# =============================================================================

class Evaluator:
    """Utility-based scoring for order selection"""
    
    def __init__(self, nav: Navigator):
        self.nav = nav
        self._analyze_map()
    
    def _analyze_map(self):
        """Compute map characteristics for strategy tuning"""
        self.aggression = 1.3
        
        if not self.nav.shops or not self.nav.submits:
            return
        
        shop = self.nav.shops[0]
        submit = self.nav.submits[0]
        
        d_shop_submit = self.nav.distance_to(shop[0], shop[1], submit[0], submit[1])
        
        # Classify map
        if d_shop_submit <= 6:
            self.aggression = 1.8
        elif d_shop_submit <= 12:
            self.aggression = 1.5
        else:
            self.aggression = 1.2
        
        # Adjust for resources
        if len(self.nav.cookers) >= 3:
            self.aggression *= 1.15
        elif len(self.nav.cookers) <= 1:
            self.aggression *= 0.85
    
    def score_order(self, order: Dict, bx: int, by: int, turn: int,
                    assigned: Set[int], n_cooking: int) -> float:
        """Compute utility score for an order"""
        oid = order['order_id']
        if oid in assigned:
            return -9999
        if not order['is_active']:
            return -9999
        
        # Parse ingredients
        ingredients = [FOOD_MAP[fn] for fn in order['required'] if fn in FOOD_MAP]
        cost = ShopCosts.PLATE.buy_cost + sum(f.buy_cost for f in ingredients)
        profit = order['reward'] - cost
        penalty = order.get('penalty', 0)
        
        if profit + penalty <= 0:
            return -9999
        
        # Time estimation
        turns_left = order['expires_turn'] - turn
        n_cook = sum(1 for f in ingredients if f.can_cook)
        n_chop = sum(1 for f in ingredients if f.can_chop)
        
        base_time = 10 + len(ingredients) * 3 + n_cook * 22 + n_chop * 5
        
        # Add travel time from shop
        if self.nav.shops:
            shop = self.nav.shops[0]
            d = self.nav.distance_to(bx, by, shop[0], shop[1])
            base_time += d
        
        # Feasibility check
        if base_time > turns_left * self.aggression:
            return -9999
        
        # Efficiency score
        efficiency = profit / max(base_time, 1)
        
        # Urgency boost for expiring orders with penalties
        urgency = 0.0
        if turns_left < 60 and penalty > 15:
            urgency = penalty * (60 - turns_left) / 60 * 0.4
        
        # Cooker contention
        if n_cook > 0 and n_cooking >= len(self.nav.cookers):
            efficiency *= 0.5
        
        return efficiency + urgency


# =============================================================================
# PLANNER - Goal Decomposition
# =============================================================================

class Planner:
    """Generates action sequences for orders using cook-first parallelism"""
    
    def __init__(self, nav: Navigator):
        self.nav = nav
    
    def plan_order(self, order: Dict, c: RobotController, bid: int,
                   claimed: Set[Tuple[int, int]]) -> Optional[List[tuple]]:
        """Generate recipe for completing an order with cook-first optimization"""
        bs = c.get_bot_state(bid)
        if not bs:
            return None
        
        bx, by = bs['x'], bs['y']
        team = c.get_team()
        
        shop = self.nav.nearest(bx, by, 'SHOP')
        submit = self.nav.nearest(bx, by, 'SUBMIT')
        trash = self.nav.nearest(bx, by, 'TRASH')
        
        if not shop or not submit:
            return None
        
        # Choose assembly point (walkable near shop, closest to submit, not claimed)
        shop_adj = self.nav.adjacent_walkable(shop[0], shop[1])
        free_adj = [a for a in shop_adj if a not in claimed and a in self.nav.walkable]
        if free_adj:
            assembly = min(free_adj, key=lambda a: self.nav.distance_to(a[0], a[1], submit[0], submit[1]))
        elif shop_adj:
            # All claimed - pick one that's different from what's claimed
            unclaimed = [a for a in shop_adj if a in self.nav.walkable]
            if unclaimed:
                assembly = unclaimed[0]
            else:
                return None
        else:
            return None
        
        # Parse ingredients
        ingredients = [FOOD_MAP[fn] for fn in order['required'] if fn in FOOD_MAP]
        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]  # MEAT
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]  # EGG
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]  # ONIONS
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]  # NOODLES, SAUCE
        
        all_cook = cook_chop + cook_only
        needs_chop = bool(cook_chop) or bool(chop_only)
        
        # Find cooker if needed
        cooker = None
        if all_cook:
            for ck in self.nav.cookers:
                if ck not in claimed:
                    cooker = ck
                    break
            if not cooker and self.nav.cookers:
                cooker = self.nav.cookers[0]
            if not cooker:
                return None
        
        # Find counter for chopping
        chop_c = None
        if needs_chop:
            for ct in self.nav.counters:
                if ct not in claimed:
                    chop_c = ct
                    break
            if not chop_c and self.nav.counters:
                chop_c = self.nav.counters[0]
            if needs_chop and not chop_c:
                return None
        
        steps: List[tuple] = []
        
        # === PHASE 1: Go to assembly ===
        steps.append(('goto', assembly))
        
        # === PHASE 2: Clean up assembly if occupied ===
        tile_a = c.get_tile(team, assembly[0], assembly[1])
        if tile_a and getattr(tile_a, 'item', None) is not None:
            steps.append(('pickup', assembly))
            if trash:
                steps.append(('trash', trash))
                steps.append(('goto', assembly))
        
        # === PHASE 3: Clean up cooker if it has leftover food ===
        if cooker:
            tile_k = c.get_tile(team, cooker[0], cooker[1])
            if tile_k and isinstance(getattr(tile_k, 'item', None), Pan):
                if tile_k.item.food is not None:
                    steps.append(('take_pan', cooker))
                    if trash:
                        steps.append(('trash', trash))
                        steps.append(('goto', assembly))
        
        # === COOK-FIRST PARALLELISM ===
        # If we have cookable items, start the first one cooking, then do plate work
        # while it cooks. This saves ~20 turns per cook item.
        
        if all_cook and cooker:
            # Start first cook item
            fc = all_cook[0]
            steps.append(('buy', fc, shop))
            if fc.can_chop and chop_c:
                steps.append(('place', chop_c))
                steps.append(('chop', chop_c))
                steps.append(('pickup', chop_c))
            steps.append(('place_cook', cooker))
            steps.append(('goto', assembly))
            
            # Now do plate + simple + chop work while first item cooks
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
                steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
            
            # Collect first cooked item
            steps.append(('wait_take', cooker))
            steps.append(('goto', assembly))
            steps.append(('add_plate', assembly))
            
            # Process remaining cook items sequentially
            for ft in all_cook[1:]:
                steps.append(('buy', ft, shop))
                if ft.can_chop and chop_c:
                    steps.append(('place', chop_c))
                    steps.append(('chop', chop_c))
                    steps.append(('pickup', chop_c))
                steps.append(('place_cook', cooker))
                steps.append(('wait_take', cooker))
                steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
        else:
            # No cook items - just do plate + simple + chop
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
                steps.append(('goto', assembly))
                steps.append(('add_plate', assembly))
        
        # === PHASE 8: Pickup plate and submit ===
        steps.append(('pickup', assembly))
        steps.append(('submit', submit))
        
        return steps


# =============================================================================
# EXECUTOR - Action Dispatch
# =============================================================================

class Executor:
    """Executes action sequences with move/action budget tracking"""
    
    def __init__(self, nav: Navigator):
        self.nav = nav
    
    def _move_with_avoidance(self, c: RobotController, bid: int, 
                              dxy: Optional[Tuple[int, int]]) -> bool:
        """Try to move in direction dxy, or find alternative if blocked"""
        if dxy and c.can_move(bid, dxy[0], dxy[1]):
            return c.move(bid, dxy[0], dxy[1])
        
        # Try alternative directions
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) != (0, 0) and c.can_move(bid, dx, dy):
                    return c.move(bid, dx, dy)
        return False
    
    def execute_step(self, c: RobotController, bid: int, step: tuple,
                     task: Dict) -> bool:
        """
        Execute a single step. Returns True if step completed.
        """
        action = step[0]
        bs = c.get_bot_state(bid)
        if not bs:
            return False
        
        bx, by = bs['x'], bs['y']
        
        # === GOTO ===
        if action == 'goto':
            target = step[1]
            if (bx, by) == target:
                return True
            
            dxy = self.nav.step_onto(bx, by, target[0], target[1])
            self._move_with_avoidance(c, bid, dxy)
            return False
        
        # === BUY ===
        elif action == 'buy':
            item, shop = step[1], step[2]
            
            # Navigate to shop first
            d = self.nav.distance_to(bx, by, shop[0], shop[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, shop[0], shop[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            # Adjacent - buy
            if c.buy(bid, item, shop[0], shop[1]):
                return True
            return False
        
        # === PLACE ===
        elif action == 'place':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.place(bid, target[0], target[1]):
                return True
            return False
        
        # === PICKUP ===
        elif action == 'pickup':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.pickup(bid, target[0], target[1]):
                return True
            return False
        
        # === CHOP ===
        elif action == 'chop':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.chop(bid, target[0], target[1]):
                return True
            return False
        
        # === PLACE_COOK ===
        elif action == 'place_cook':
            cooker = step[1]
            
            d = self.nav.distance_to(bx, by, cooker[0], cooker[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, cooker[0], cooker[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            # Try start_cook first, fallback to place
            if c.start_cook(bid, cooker[0], cooker[1]):
                return True
            if c.place(bid, cooker[0], cooker[1]):
                return True
            return False
        
        # === WAIT_TAKE ===
        elif action == 'wait_take':
            cooker = step[1]
            team = c.get_team()
            
            d = self.nav.distance_to(bx, by, cooker[0], cooker[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, cooker[0], cooker[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            # Check if cooked
            tile = c.get_tile(team, cooker[0], cooker[1])
            if tile:
                progress = getattr(tile, 'cook_progress', 0)
                if progress >= GameConstants.COOK_PROGRESS:
                    if c.take_from_pan(bid, cooker[0], cooker[1]):
                        return True
            return False  # Still waiting
        
        # === TAKE_PAN ===
        elif action == 'take_pan':
            cooker = step[1]
            
            d = self.nav.distance_to(bx, by, cooker[0], cooker[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, cooker[0], cooker[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.take_from_pan(bid, cooker[0], cooker[1]):
                return True
            return False
        
        # === ADD_PLATE ===
        elif action == 'add_plate':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.add_food_to_plate(bid, target[0], target[1]):
                return True
            return False
        
        # === TRASH ===
        elif action == 'trash':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.trash(bid, target[0], target[1]):
                return True
            return False
        
        # === SUBMIT ===
        elif action == 'submit':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.submit(bid, target[0], target[1]):
                return True
            return False
        
        # === TAKE_CLEAN_PLATE ===
        elif action == 'take_clean_plate':
            target = step[1]
            
            d = self.nav.distance_to(bx, by, target[0], target[1])
            if d > 0:
                dxy = self.nav.step_toward(bx, by, target[0], target[1])
                self._move_with_avoidance(c, bid, dxy)
                return False
            
            if c.take_clean_plate(bid, target[0], target[1]):
                return True
            return False
        
        return False


# =============================================================================
# TACTICIAN - Strategic Operations
# =============================================================================

class Tactician:
    """Handles sabotage, recycling, and defense"""
    
    def __init__(self, nav: Navigator):
        self.nav = nav
        self.has_switched = False
        self.sabotage_active = False
    
    def should_sabotage(self, c: RobotController, turn: int) -> bool:
        """Evaluate if sabotage is worthwhile"""
        if self.has_switched:
            return False
        
        info = c.get_switch_info()
        if not info['window_active']:
            return False
        
        team = c.get_team()
        enemy = c.get_enemy_team()
        our_money = c.get_team_money(team)
        enemy_money = c.get_team_money(enemy)
        
        # Sabotage if significantly behind
        if enemy_money > our_money + 100:
            return True
        if turn >= 380 and enemy_money > our_money + 60:
            return True
        
        return False
    
    def find_recycled_plate(self, c: RobotController, bx: int, by: int
                            ) -> Optional[Tuple[int, int]]:
        """Find sink table with clean plates"""
        team = c.get_team()
        best = None
        best_d = 9999
        
        for st in self.nav.sink_tables:
            tile = c.get_tile(team, st[0], st[1])
            if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                d = self.nav.distance_to(bx, by, st[0], st[1])
                if d < best_d:
                    best_d = d
                    best = st
        
        return best


# =============================================================================
# BOT PLAYER - Main Controller
# =============================================================================

class BotPlayer:
    """Main bot interface"""
    
    def __init__(self, game_map):
        self.nav = Navigator(game_map)
        self.evaluator = Evaluator(self.nav)
        self.planner = Planner(self.nav)
        self.executor = Executor(self.nav)
        self.tactician = Tactician(self.nav)
        
        # State
        self.team = None
        self.tasks: Dict[int, Dict] = {}
        self.assigned_orders: Set[int] = set()
        self.completed_orders: Set[int] = set()
        self.failed_orders: Dict[int, int] = {}  # order_id -> fail_turn
    
    def play_turn(self, c: RobotController):
        """Execute one game turn"""
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
                    'assembly': None, 'cooker': None,
                    'stuck_count': 0, 'last_progress': turn,
                }
        
        # Clean up expired assignments
        active_ids = {o['order_id'] for o in orders if o['is_active']}
        for bid in bots:
            t = self.tasks[bid]
            oid = t.get('order_id')
            if oid and oid not in active_ids:
                self._abort_task(bid, turn)
        
        # Detect stuck bots
        for bid in bots:
            t = self.tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                no_progress = turn - t.get('last_progress', turn)
                if t['stuck_count'] > 12 or no_progress > 30:
                    oid = t.get('order_id')
                    if oid:
                        self.failed_orders[oid] = turn
                    self._abort_task(bid, turn)
        
        # Handle idle bots with items
        idle_bots = []
        for bid in bots:
            t = self.tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            
            bs = c.get_bot_state(bid)
            if bs and bs.get('holding'):
                # Try to trash or place the item
                bx, by = bs['x'], bs['y']
                trash = self.nav.nearest(bx, by, 'TRASH')
                if trash and self.nav.distance_to(bx, by, trash[0], trash[1]) < 20:
                    t['recipe'] = [('trash', trash)]
                    t['step'] = 0
                    t['order_id'] = None
                    t['stuck_count'] = 0
                    t['last_progress'] = turn
                    continue
                # Otherwise try to place on counter
                team = c.get_team()
                for ct in self.nav.counters:
                    tile = c.get_tile(team, ct[0], ct[1])
                    if tile and getattr(tile, 'item', None) is None:
                        if self.nav.distance_to(bx, by, ct[0], ct[1]) < 20:
                            t['recipe'] = [('place', ct)]
                            t['step'] = 0
                            t['order_id'] = None
                            t['stuck_count'] = 0
                            t['last_progress'] = turn
                            break
                continue
            
            idle_bots.append(bid)
        
        # Check sabotage
        if self.tactician.should_sabotage(c, turn) and idle_bots:
            if c.switch_maps():
                self.tactician.has_switched = True
                self.tactician.sabotage_active = True
        
        # Count active cooking bots
        n_cooking = sum(1 for bid in bots 
                       if self.tasks[bid].get('cooker') and self.tasks[bid].get('order_id'))
        
        # Assign orders to idle bots
        for bid in idle_bots:
            bs = c.get_bot_state(bid)
            if not bs:
                continue
            bx, by = bs['x'], bs['y']
            
            # Score orders
            scored = []
            for o in orders:
                if not o['is_active']:
                    continue
                if o['order_id'] in self.failed_orders:
                    if turn - self.failed_orders[o['order_id']] < 20:
                        continue
                
                score = self.evaluator.score_order(
                    o, bx, by, turn, self.assigned_orders, n_cooking
                )
                if score > -9999:
                    scored.append((score, o))
            
            if not scored:
                continue
            
            scored.sort(key=lambda x: -x[0])
            best_order = scored[0][1]
            
            # Get claimed resources
            claimed = self._get_claimed(bid)
            
            # Generate recipe
            recipe = self.planner.plan_order(best_order, c, bid, claimed)
            if not recipe:
                continue
            
            # Assign
            t = self.tasks[bid]
            t['recipe'] = recipe
            t['step'] = 0
            t['order_id'] = best_order['order_id']
            t['stuck_count'] = 0
            t['last_progress'] = turn
            
            # Track resources
            for step in recipe:
                if step[0] == 'goto' and len(step) > 1:
                    t['assembly'] = step[1]
                    break
            for step in recipe:
                if step[0] in ('place_cook', 'wait_take') and len(step) > 1:
                    t['cooker'] = step[1]
                    break
            
            self.assigned_orders.add(best_order['order_id'])
            
            # Update cooking count
            if t.get('cooker'):
                n_cooking += 1
        
        # Execute all bot actions
        for bid in bots:
            self._execute_bot(c, bid, turn)
    
    def _execute_bot(self, c: RobotController, bid: int, turn: int):
        """Execute current task for a bot"""
        t = self.tasks[bid]
        
        if not t['recipe'] or t['step'] >= len(t['recipe']):
            return
        
        bs = c.get_bot_state(bid)
        if not bs:
            return
        
        step = t['recipe'][t['step']]
        
        completed = self.executor.execute_step(c, bid, step, t)
        
        if completed:
            t['step'] += 1
            t['stuck_count'] = 0
            t['last_progress'] = turn
            
            # Check if order completed
            if t['step'] >= len(t['recipe']):
                oid = t.get('order_id')
                if oid:
                    self.completed_orders.add(oid)
                    self.assigned_orders.discard(oid)
                self._reset_task(bid)
        else:
            # Increment stuck counter if position unchanged
            old_pos = t.get('_last_pos')
            new_pos = (bs['x'], bs['y'])
            if old_pos == new_pos:
                t['stuck_count'] = t.get('stuck_count', 0) + 1
            t['_last_pos'] = new_pos
    
    def _get_claimed(self, exclude_bid: int) -> Set[Tuple[int, int]]:
        """Get all resources claimed by other bots"""
        claimed = set()
        for bid, t in self.tasks.items():
            if bid == exclude_bid:
                continue
            if t.get('assembly'):
                claimed.add(t['assembly'])
            if t.get('cooker'):
                claimed.add(t['cooker'])
        return claimed
    
    def _abort_task(self, bid: int, turn: int):
        """Abort current task"""
        t = self.tasks[bid]
        oid = t.get('order_id')
        if oid:
            self.assigned_orders.discard(oid)
            self.failed_orders[oid] = turn
        self._reset_task(bid)
    
    def _reset_task(self, bid: int):
        """Reset task state"""
        self.tasks[bid] = {
            'recipe': [], 'step': 0, 'order_id': None,
            'assembly': None, 'cooker': None,
            'stuck_count': 0, 'last_progress': 0,
        }
