"""
ATLAS BOT - Maximum Disruption Counter-Strategy
=================================================

CORE INSIGHT: test.py wins by being extremely efficient at order completion.
They NEVER sabotage. Our strategy:

1. Match their efficiency as much as possible (copy proven patterns)
2. IMMEDIATELY switch at turn 250 with BOTH bots
3. MAXIMUM DISRUPTION: Steal ALL pans first, then ALL plates
4. Make their losses exceed the time we spend sabotaging

Target: Their cooker dependency is their Achilles heel. No pans = no cooking orders.
"""

from collections import deque
from typing import Tuple, Optional, List, Dict, Set, Any
from enum import Enum, auto

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food

DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[ATLAS] {msg}")

INGREDIENT_INFO = {
    'SAUCE':   {'cost': 2, 'chop': False, 'cook': False},
    'EGG':     {'cost': 10, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 4, 'chop': True, 'cook': False},
    'NOODLES': {'cost': 3, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 12, 'chop': True, 'cook': True},
}

class Phase(Enum):
    PRODUCTION = auto()
    SABOTAGE = auto()

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.w = map_copy.width
        self.h = map_copy.height
        self.team = None
        
        # Parse tiles
        self.tile_locs: Dict[str, List[Tuple[int, int]]] = {}
        self.walkable: Set[Tuple[int, int]] = set()
        for x in range(self.w):
            for y in range(self.h):
                t = map_copy.tiles[x][y]
                self.tile_locs.setdefault(t.tile_name, []).append((x, y))
                if t.is_walkable:
                    self.walkable.add((x, y))
        
        # Pathfinding
        self._adj_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._dist: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._next_step: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        self._precompute_bfs()
        
        # Key locations
        self._shops = self.tile_locs.get('SHOP', [])
        self._submits = self.tile_locs.get('SUBMIT', [])
        self._counters = self.tile_locs.get('COUNTER', [])
        self._cookers = self.tile_locs.get('COOKER', [])
        self._trash = self.tile_locs.get('TRASH', [])
        self._sink_tables = self.tile_locs.get('SINK_TABLE', [])
        
        # State tracking
        self.phase = Phase.PRODUCTION
        self.has_switched = False
        self.sabotage_start = 0
        self.pans_stolen = 0
        self.plates_stolen = 0
        
        # Order tracking
        self.current_orders: Dict[int, Dict] = {}  # bot_id -> order info
        self.completed_orders: Set[int] = set()
        self.bot_states: Dict[int, Dict] = {}
        
        # Resource tracking
        self.pan_placed = False
        self.plate_assembly: Dict[int, Tuple[int, int]] = {}  # bot_id -> assembly loc
    
    def _precompute_bfs(self):
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
                        if (nx, ny) not in self.walkable or (nx, ny) in dist:
                            continue
                        dist[(nx, ny)] = dist[(cx, cy)] + 1
                        if (cx, cy) == src:
                            first_step[(nx, ny)] = (dx, dy)
                        else:
                            first_step[(nx, ny)] = first_step[(cx, cy)]
                        q.append((nx, ny))
            self._dist[src] = dist
            self._next_step[src] = first_step
    
    def _adj_walk(self, tx: int, ty: int) -> List[Tuple[int, int]]:
        key = (tx, ty)
        if key not in self._adj_cache:
            result = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = tx + dx, ty + dy
                    if (nx, ny) in self.walkable:
                        result.append((nx, ny))
            self._adj_cache[key] = result
        return self._adj_cache[key]
    
    def _dist_to_tile(self, sx: int, sy: int, tx: int, ty: int) -> int:
        adj = self._adj_walk(tx, ty)
        if not adj:
            return 9999
        src = (sx, sy)
        if src not in self._dist:
            return 9999
        d = self._dist[src]
        return min((d.get(a, 9999) for a in adj), default=9999)
    
    def _nearest(self, bx: int, by: int, name: str) -> Optional[Tuple[int, int]]:
        locs = self.tile_locs.get(name, [])
        if not locs:
            return None
        return min(locs, key=lambda p: self._dist_to_tile(bx, by, p[0], p[1]))
    
    def _move_toward(self, c: RobotController, bid: int, 
                     target: Tuple[int, int]) -> bool:
        """Move bot toward target. Returns True if adjacent."""
        bs = c.get_bot_state(bid)
        if not bs:
            return False
        bx, by = bs['x'], bs['y']
        tx, ty = target
        
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return True
        
        adj = self._adj_walk(tx, ty)
        if not adj:
            return False
        
        src = (bx, by)
        if src not in self._dist:
            return False
        
        best_target = min(adj, key=lambda a: self._dist[src].get(a, 9999))
        if best_target not in self._dist[src]:
            return False
        
        if best_target in self._next_step.get(src, {}):
            dx, dy = self._next_step[src][best_target]
            if c.can_move(bid, dx, dy):
                c.move(bid, dx, dy)
                return max(abs(bx + dx - tx), abs(by + dy - ty)) <= 1
        
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if c.can_move(bid, dx, dy):
                    c.move(bid, dx, dy)
                    return False
        
        return False
    
    # ================================================================
    # SABOTAGE EXECUTION - MAXIMUM DISRUPTION
    # ================================================================
    
    def _execute_sabotage(self, c: RobotController, bid: int, team: Team):
        """Execute aggressive sabotage.
        
        After switching, we're on the ENEMY map! Need to use enemy team for get_tile.
        
        Priority:
        1. If holding something -> trash it immediately
        2. Steal pans from cookers (highest impact)
        3. Steal plates from sink tables
        4. Steal anything from counters
        """
        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']
        holding = bs.get('holding')
        
        # CRITICAL: After switching, we're on enemy map!
        # The bot's map_team tells us which map we're on
        map_team_name = bs.get('map_team', team.name)
        if map_team_name == 'RED':
            map_team = Team.RED
        else:
            map_team = Team.BLUE
        
        # We're sabotaging - so we want to look at the current map (enemy's map)
        # After switch, map_team should be the enemy team
        enemy_team = c.get_enemy_team()
        
        trash = self._trash[0] if self._trash else None
        
        # If holding something, we need to trash it to free hands
        if holding:
            log(f"Bot {bid} at ({bx},{by}) holding {holding.get('type')}, need to trash at {trash}")
            if trash:
                dist = max(abs(bx - trash[0]), abs(by - trash[1]))
                log(f"  Distance to trash: {dist}")
                if dist <= 1:
                    # Adjacent - try to trash
                    result = c.trash(bid, trash[0], trash[1])
                    log(f"  Trash result: {result}")
                else:
                    # Move toward trash
                    self._move_toward(c, bid, trash)
                    log(f"  Moving toward trash")
            else:
                log(f"No trash found!")
            return
        
        # Priority 1: Steal pans from cookers (MAXIMUM IMPACT)
        log(f"Bot {bid} checking {len(self._cookers)} cookers for pans... (enemy={enemy_team.name}, map_team={map_team_name})")
        for ck in self._cookers:
            # Use enemy_team to look at tiles on the enemy's map
            tile = c.get_tile(enemy_team, ck[0], ck[1])
            log(f"  Cooker {ck}: tile={tile}, item={getattr(tile, 'item', None) if tile else None}")
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Pan):
                    log(f"Found pan at {ck}, moving to steal...")
                    if self._move_toward(c, bid, ck):
                        # Try to take cooked food first, else pickup the pan
                        if item.food and item.food.is_cooked:
                            if c.take_from_pan(bid, ck[0], ck[1]):
                                log(f"Took food from pan at {ck}!")
                        else:
                            if c.pickup(bid, ck[0], ck[1]):
                                self.pans_stolen += 1
                                log(f"Stole pan from {ck}!")
                    return
        
        # Priority 2: Steal plates from sink tables
        for st in self._sink_tables:
            tile = c.get_tile(enemy_team, st[0], st[1])
            if tile:
                item = getattr(tile, 'item', None)
                if isinstance(item, Plate):
                    log(f"Found plate at {st}, moving to steal...")
                    if self._move_toward(c, bid, st):
                        if c.pickup(bid, st[0], st[1]):
                            self.plates_stolen += 1
                            log(f"Stole plate from {st}!")
                    return
        
        # Priority 3: Steal from counters (plates, food, anything)
        for ct in self._counters:
            tile = c.get_tile(enemy_team, ct[0], ct[1])
            if tile and getattr(tile, 'item', None) is not None:
                log(f"Found item on counter {ct}, moving to steal...")
                if self._move_toward(c, bid, ct):
                    if c.pickup(bid, ct[0], ct[1]):
                        log(f"Stole item from counter {ct}!")
                return
        
        # Nothing to steal - patrol near cookers waiting for pans
        log(f"Nothing to steal, patrolling...")
        if self._cookers:
            target = min(self._cookers, 
                        key=lambda ck: self._dist_to_tile(bx, by, ck[0], ck[1]))
            self._move_toward(c, bid, target)
    
    # ================================================================
    # EFFICIENT ORDER COMPLETION (Simplified but effective)
    # ================================================================
    
    def _select_best_order(self, c: RobotController, team: Team, 
                           bid: int) -> Optional[Dict]:
        """Select best order - prioritize simple non-cooking orders."""
        orders = c.get_orders(team)
        turn = c.get_turn()
        money = c.get_team_money(team)
        
        best = None
        best_score = -999
        
        for o in orders:
            if not o['is_active']:
                continue
            if o.get('completed_turn') is not None:
                continue
            if o['order_id'] in self.completed_orders:
                continue
            if any(self.current_orders.get(b, {}).get('order_id') == o['order_id'] 
                   for b in self.current_orders if b != bid):
                continue
            
            # Calculate cost and check affordability
            cost = ShopCosts.PLATE.buy_cost
            needs_cook = False
            needs_chop = False
            for fn in o['required']:
                info = INGREDIENT_INFO.get(fn, {})
                cost += info.get('cost', 5)
                if info.get('cook'):
                    needs_cook = True
                if info.get('chop'):
                    needs_chop = True
            
            if needs_cook and not self.pan_placed:
                cost += ShopCosts.PAN.buy_cost
            
            if cost > money:
                continue
            
            # Check time feasibility
            tleft = o['expires_turn'] - turn
            if tleft < 20:
                continue
            
            # Score - heavily favor simple orders
            profit = o['reward'] - cost
            score = profit / max(tleft, 1) * 10
            
            # Bonuses for simplicity
            if not needs_cook:
                score += 15  # Big bonus for no cooking
            if not needs_chop:
                score += 5
            if len(o['required']) == 1:
                score += 10
            elif len(o['required']) == 2:
                score += 5
            
            if score > best_score:
                best_score = score
                best = o
        
        return best
    
    def _is_tile_empty(self, c: RobotController, team: Team, x: int, y: int) -> bool:
        tile = c.get_tile(team, x, y)
        if tile:
            return getattr(tile, 'item', None) is None
        return True
    
    def _find_empty_counter(self, c: RobotController, team: Team, 
                           bx: int, by: int) -> Optional[Tuple[int, int]]:
        for ct in sorted(self._counters, 
                        key=lambda cc: self._dist_to_tile(bx, by, cc[0], cc[1])):
            if self._is_tile_empty(c, team, ct[0], ct[1]):
                return ct
        return None
    
    def _execute_production(self, c: RobotController, bid: int, team: Team):
        """Execute order completion - simplified state machine."""
        bs = c.get_bot_state(bid)
        if not bs:
            return
        bx, by = bs['x'], bs['y']
        holding = bs.get('holding')
        money = c.get_team_money(team)
        
        state = self.bot_states.get(bid, {'phase': 'IDLE'})
        phase = state.get('phase', 'IDLE')
        
        shop = self._shops[0] if self._shops else None
        submit = self._submits[0] if self._submits else None
        trash = self._trash[0] if self._trash else None
        
        if not shop or not submit:
            return
        
        sx, sy = shop
        
        # Get or assign order
        order_info = self.current_orders.get(bid, {})
        order = order_info.get('order')
        
        if phase == 'IDLE':
            if not order:
                order = self._select_best_order(c, team, bid)
                if order:
                    self.current_orders[bid] = {
                        'order': order,
                        'order_id': order['order_id'],
                        'ingredients_done': set(),
                        'plate_placed': False,
                        'assembly': None,
                    }
                    order_info = self.current_orders[bid]
                    log(f"Bot {bid} selected order {order['order_id']}: {order['required']}")
            
            if order:
                # Determine what we need
                needs_cook = any(INGREDIENT_INFO.get(ing, {}).get('cook') 
                                for ing in order['required'])
                
                if needs_cook and not self.pan_placed:
                    state['phase'] = 'BUY_PAN'
                elif not order_info.get('plate_placed'):
                    state['phase'] = 'BUY_PLATE'
                else:
                    # Work on next ingredient
                    for ing in order['required']:
                        if ing not in order_info.get('ingredients_done', set()):
                            state['phase'] = 'BUY_INGREDIENT'
                            state['current_ing'] = ing
                            break
                    else:
                        state['phase'] = 'PICKUP_PLATE'
        
        elif phase == 'BUY_PAN':
            if holding and holding.get('type') == 'Pan':
                state['phase'] = 'PLACE_PAN'
            elif not holding:
                if self._move_toward(c, bid, shop):
                    if money >= ShopCosts.PAN.buy_cost:
                        c.buy(bid, ShopCosts.PAN, sx, sy)
            else:
                if trash and self._move_toward(c, bid, trash):
                    c.trash(bid, trash[0], trash[1])
        
        elif phase == 'PLACE_PAN':
            cooker = self._cookers[0] if self._cookers else None
            if cooker:
                if self._move_toward(c, bid, cooker):
                    if c.place(bid, cooker[0], cooker[1]):
                        self.pan_placed = True
                        state['phase'] = 'IDLE'
        
        elif phase == 'BUY_PLATE':
            if holding and holding.get('type') == 'Plate':
                state['phase'] = 'PLACE_PLATE'
            elif not holding:
                if self._move_toward(c, bid, shop):
                    if money >= ShopCosts.PLATE.buy_cost:
                        c.buy(bid, ShopCosts.PLATE, sx, sy)
            else:
                if trash and self._move_toward(c, bid, trash):
                    c.trash(bid, trash[0], trash[1])
        
        elif phase == 'PLACE_PLATE':
            assembly = order_info.get('assembly')
            if not assembly:
                assembly = self._find_empty_counter(c, team, bx, by)
                if order_info:
                    order_info['assembly'] = assembly
            
            if assembly:
                if self._move_toward(c, bid, assembly):
                    if self._is_tile_empty(c, team, assembly[0], assembly[1]):
                        if c.place(bid, assembly[0], assembly[1]):
                            if order_info:
                                order_info['plate_placed'] = True
                            state['phase'] = 'IDLE'
        
        elif phase == 'BUY_INGREDIENT':
            ing_name = state.get('current_ing')
            food_type = getattr(FoodType, ing_name, None) if ing_name else None
            info = INGREDIENT_INFO.get(ing_name, {})
            
            if holding and holding.get('type') == 'Food':
                if info.get('chop'):
                    state['phase'] = 'PLACE_FOR_CHOP'
                elif info.get('cook'):
                    state['phase'] = 'START_COOK'
                else:
                    state['phase'] = 'ADD_TO_PLATE'
            elif not holding:
                if self._move_toward(c, bid, shop):
                    if food_type and money >= food_type.buy_cost:
                        c.buy(bid, food_type, sx, sy)
            else:
                if trash and self._move_toward(c, bid, trash):
                    c.trash(bid, trash[0], trash[1])
        
        elif phase == 'PLACE_FOR_CHOP':
            counter = self._find_empty_counter(c, team, bx, by)
            if counter:
                state['chop_loc'] = counter
                if self._move_toward(c, bid, counter):
                    if c.place(bid, counter[0], counter[1]):
                        state['phase'] = 'CHOP'
        
        elif phase == 'CHOP':
            chop_loc = state.get('chop_loc')
            if chop_loc:
                if self._move_toward(c, bid, chop_loc):
                    tile = c.get_tile(team, chop_loc[0], chop_loc[1])
                    if tile and isinstance(getattr(tile, 'item', None), Food):
                        if tile.item.is_chopped:
                            state['phase'] = 'PICKUP_CHOPPED'
                        else:
                            c.chop(bid, chop_loc[0], chop_loc[1])
        
        elif phase == 'PICKUP_CHOPPED':
            chop_loc = state.get('chop_loc')
            if chop_loc:
                if self._move_toward(c, bid, chop_loc):
                    if c.pickup(bid, chop_loc[0], chop_loc[1]):
                        info = INGREDIENT_INFO.get(state.get('current_ing'), {})
                        if info.get('cook'):
                            state['phase'] = 'START_COOK'
                        else:
                            state['phase'] = 'ADD_TO_PLATE'
        
        elif phase == 'START_COOK':
            cooker = self._cookers[0] if self._cookers else None
            if cooker:
                if self._move_toward(c, bid, cooker):
                    if c.place(bid, cooker[0], cooker[1]):
                        state['phase'] = 'WAIT_COOK'
                        state['cook_loc'] = cooker
        
        elif phase == 'WAIT_COOK':
            cook_loc = state.get('cook_loc')
            if cook_loc:
                tile = c.get_tile(team, cook_loc[0], cook_loc[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food:
                        stage = getattr(pan.food, 'cooked_stage', 0)
                        if stage >= 1:  # Cooked or burnt
                            state['phase'] = 'TAKE_FROM_PAN'
        
        elif phase == 'TAKE_FROM_PAN':
            cook_loc = state.get('cook_loc')
            if cook_loc:
                if holding:
                    stage = holding.get('cooked_stage', 0)
                    if stage == 2:  # Burnt
                        state['phase'] = 'TRASH_BURNT'
                    else:
                        state['phase'] = 'ADD_TO_PLATE'
                elif self._move_toward(c, bid, cook_loc):
                    c.take_from_pan(bid, cook_loc[0], cook_loc[1])
        
        elif phase == 'TRASH_BURNT':
            if trash:
                if self._move_toward(c, bid, trash):
                    c.trash(bid, trash[0], trash[1])
                    state['phase'] = 'BUY_INGREDIENT'  # Retry
        
        elif phase == 'ADD_TO_PLATE':
            assembly = order_info.get('assembly') if order_info else None
            if assembly:
                if self._move_toward(c, bid, assembly):
                    if c.place(bid, assembly[0], assembly[1]):
                        ing = state.get('current_ing')
                        if ing and order_info:
                            order_info.setdefault('ingredients_done', set()).add(ing)
                        state['phase'] = 'IDLE'
        
        elif phase == 'PICKUP_PLATE':
            assembly = order_info.get('assembly') if order_info else None
            if assembly:
                if self._move_toward(c, bid, assembly):
                    if c.pickup(bid, assembly[0], assembly[1]):
                        state['phase'] = 'SUBMIT'
        
        elif phase == 'SUBMIT':
            ux, uy = submit
            if self._move_toward(c, bid, submit):
                if c.submit(bid, ux, uy):
                    if order_info and order_info.get('order_id'):
                        self.completed_orders.add(order_info['order_id'])
                        log(f"Bot {bid} completed order {order_info['order_id']}!")
                    self.current_orders.pop(bid, None)
                    state['phase'] = 'IDLE'
        
        self.bot_states[bid] = state
    
    # ================================================================
    # MAIN TURN LOGIC
    # ================================================================
    
    def play_turn(self, c: RobotController):
        turn = c.get_turn()
        team = c.get_team()
        if self.team is None:
            self.team = team
        
        bots = c.get_team_bot_ids(team)
        
        # Initialize bot states
        for bid in bots:
            if bid not in self.bot_states:
                self.bot_states[bid] = {'phase': 'IDLE'}
        
        # Check for sabotage opportunity
        switch_info = c.get_switch_info()
        switch_turn = switch_info.get('switch_turn', 250)
        switch_duration = switch_info.get('switch_duration', 100)
        
        in_window = switch_turn <= turn < switch_turn + switch_duration
        
        # AGGRESSIVE SABOTAGE STRATEGY
        # Switch IMMEDIATELY at start of window and stay for ~60 turns
        if in_window and not self.has_switched and self.phase == Phase.PRODUCTION:
            if c.can_switch_maps():
                turns_in = turn - switch_turn
                # Switch in first 5 turns of window
                if turns_in < 5:
                    if c.switch_maps():
                        self.has_switched = True
                        self.phase = Phase.SABOTAGE
                        self.sabotage_start = turn
                        log(f"SWITCHED TO ENEMY MAP at turn {turn}!")
                        # Reset all bots to sabotage mode
                        for bid in bots:
                            self.bot_states[bid] = {'phase': 'SABOTAGE'}
        
        # Return to production after 60 turns of sabotage
        if self.phase == Phase.SABOTAGE:
            if turn - self.sabotage_start > 60:
                self.phase = Phase.PRODUCTION
                log(f"Returning to production at turn {turn}")
                for bid in bots:
                    self.bot_states[bid] = {'phase': 'IDLE'}
                    self.current_orders.pop(bid, None)
        
        # Execute based on phase
        for bid in bots:
            if self.phase == Phase.SABOTAGE:
                self._execute_sabotage(c, bid, team)
            else:
                self._execute_production(c, bid, team)
        
        if DEBUG and turn == 499:
            log(f"Final stats: pans_stolen={self.pans_stolen}, plates_stolen={self.plates_stolen}")
