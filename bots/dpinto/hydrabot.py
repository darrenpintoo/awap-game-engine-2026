import os
import sys
import collections
from typing import List, Tuple, Dict, Optional, Set, Any
try:
    from src.robot_controller import RobotController
    from src.game_constants import Team, FoodType, ShopCosts
    from src.item import Pan, Plate, Food
except ImportError:
    try:
        from robot_controller import RobotController
        from game_constants import Team, FoodType, ShopCosts
        from item import Pan, Plate, Food
    except ImportError:
        pass

INGREDIENT_INFO = {
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False},
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False},
}

DEBUG_LOG_ENABLED = True

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy; self.initialized = False; self.team = None; self.log_path = None
        self.shops = []; self.cookers = []; self.counters = []; self.submits = []; self.trashes = []; self.sink_tables = []
        self.bot_ids = []; self.bot_tasks = {}; self.order_states = {}
        self.reserved_tiles = {}; self.reserved_stations = {}
        
    def _log(self, msg):
        if not DEBUG_LOG_ENABLED or not self.log_path: return
        with open(self.log_path, "a") as f: f.write(msg + "\n")

    def _init_log(self, team):
        if not DEBUG_LOG_ENABLED: return
        self.log_path = os.path.join(os.path.dirname(__file__), f"hydrabot_{team.name}.log")
        with open(self.log_path, "w") as f: f.write(f"=== HYDRA BOT LOG INITIALIZED: {team.name} ===\n")

    def _get_bot_pos(self, c, bid): b = c.get_bot_state(bid); return (b['x'], b['y'])
    def _chebyshev_dist(self, a, b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def _init_map(self, c, team):
        self._init_log(team); self.team = team; gm = c.get_map(team)
        for y in range(gm.height):
            for x in range(gm.width):
                tn = gm.tiles[x][y].tile_name
                if tn == "SHOP": self.shops.append((x, y))
                elif tn == "COOKER": self.cookers.append((x, y))
                elif tn == "COUNTER": self.counters.append((x, y))
                elif tn == "SUBMIT": self.submits.append((x, y))
                elif tn == "TRASH": self.trashes.append((x, y))
                elif tn == "SINKTABLE": self.sink_tables.append((x, y))
        self.initialized = True

    def _move_toward(self, c, bid, target, team):
        pos = self._get_bot_pos(c, bid)
        if self._chebyshev_dist(pos, target) <= 1: return True
        gm = c.get_map(team); turn = c.get_turn()
        t_res = self.reserved_tiles.get(turn, set())
        occupied = t_res.union({self._get_other_bot_pos(c, bid)})
        adj = set(); 
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx, ny = target[0]+dx, target[1]+dy
                if gm.in_bounds(nx, ny) and gm.is_tile_walkable(nx, ny): adj.add((nx, ny))
        if target in adj: adj.add(target)
        path = self._bfs_path(pos, adj, occupied, gm)
        if path and len(path) > 0:
            ns = path[0]; dx, dy = ns[0]-pos[0], ns[1]-pos[1]
            if turn not in self.reserved_tiles: self.reserved_tiles[turn] = set()
            self.reserved_tiles[turn].add(ns); c.move(bid, dx, dy); return False
        return False

    def _bfs_path(self, start, goals, occupied, gm):
        if start in goals: return [start]
        q = collections.deque([(start, [])]); vis = {start}
        while q:
            (cx, cy), p = q.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cx+dx, cy+dy
                if gm.in_bounds(nx, ny) and (nx, ny) not in vis and gm.is_tile_walkable(nx, ny):
                    if (nx, ny) in occupied and len(p) == 0: continue
                    np = p + [(nx, ny)]
                    if (nx, ny) in goals: return np
                    vis.add((nx, ny)); q.append(((nx, ny), np))
        return None

    def _get_other_bot_id(self, bid): return self.bot_ids[1] if bid == self.bot_ids[0] else self.bot_ids[0]
    def _get_other_bot_pos(self, c, bid): return self._get_bot_pos(c, self._get_other_bot_id(bid))
    def _get_nearest(self, p, locs): return min(locs, key=lambda l: self._chebyshev_dist(p, l)) if locs else None

    def play_turn(self, controller):
        team = controller.get_team(); bids = controller.get_team_bot_ids(team); self.bot_ids = bids
        if not self.initialized: self._init_map(controller, team)
        orders = controller.get_orders(team); active = [o for o in orders if o['is_active']]
        cur_t = controller.get_turn()
        self._log(f"--- TURN {cur_t} ---")
        for bid in bids:
            bt = self.bot_tasks.get(bid)
            if bt and bt['type'] == 'DELIVER_ORDER':
                if not any(o['order_id'] == bt['order_id'] and o['is_active'] for o in orders):
                    self._log(f"Bot {bid}: Order {bt['order_id']} expired/done.")
                    bt['status'] = 'COMPLETE'
        for bid in bids:
            if bid not in self.bot_tasks or self.bot_tasks[bid].get('status') == 'COMPLETE': self._assign_best_task(bid, active)
            self._execute_task(controller, bid, team)

    def _assign_best_task(self, bid, active):
        for o in active:
            self.bot_tasks[bid] = {'type': 'DELIVER_ORDER', 'order': o, 'order_id': o['order_id'], 'status': 'IN_PROGRESS'}
            self._log(f"Bot {bid} newly assigned to Order {o['order_id']}")
            return
        self.bot_tasks[bid] = {'type': 'IDLE', 'status': 'IN_PROGRESS'}

    def _execute_task(self, c, bid, team):
        t = self.bot_tasks.get(bid)
        if t['type'] == 'DELIVER_ORDER': self._execute_deliver_order(c, bid, t['order'], team)
        else: self._role_idle(c, bid, team)

    def _execute_deliver_order(self, c, bid, order, team):
        oid = order['order_id']; bs = c.get_bot_state(bid); bpos = (bs['x'], bs['y']); h = bs.get('holding')
        if oid not in self.order_states:
            self.order_states[oid] = {'plate_pos': None, 'plate_held_by': None, 'ingredients_ready': [], 'ingredients_in_progress': [], 'ingredients_on_plate': []}
        ostate = self.order_states[oid]; self._sync_order_state(c, team, order, ostate)
        
        needed = [ing for ing in order['required'] if ing not in ostate['ingredients_on_plate']]
        if not needed:
             self._log(f"Bot {bid} Order {oid} complete, submitting.")
             self._state_submit(c, bid, order, ostate, team); return

        is_assembler = False
        if h and h.get('type') == 'Plate': is_assembler = True
        elif not ostate.get('plate_held_by') and not ostate.get('plate_pos'):
            if bid == min(self.bot_ids): is_assembler = True
        elif ostate.get('plate_held_by') == bid: is_assembler = True
        elif ostate.get('plate_pos') and self._chebyshev_dist(bpos, ostate['plate_pos']) < 3: is_assembler = True
        
        if is_assembler: self._role_assembler(c, bid, order, ostate, team)
        else: self._role_fetcher(c, bid, order, ostate, team)

    def _sync_order_state(self, c, team, order, ostate):
        req = list(order['required'])
        ostate['plate_pos'] = None; ostate['ingredients_ready'] = []; ostate['ingredients_in_progress'] = []
        ostate['ingredients_on_plate'] = []; ostate['plate_held_by'] = None
        
        def _is_prep_ready(f_dict):
             fn = f_dict.get('food_name')
             info = INGREDIENT_INFO.get(fn, {})
             if info.get('chop') and not f_dict.get('chopped'): return False
             if info.get('cook') and f_dict.get('cooked_stage', 0) < 1: return False
             return True

        for cpos in (self.counters + self.sink_tables):
            tile = c.get_tile(team, cpos[0], cpos[1]); item = getattr(tile, 'item', None)
            if not item: continue
            if isinstance(item, Plate) and not item.dirty:
                ostate['plate_pos'] = cpos
                plated = []
                for f in item.food:
                     f_dict = {'food_name': getattr(f, 'food_name', f), 'chopped': getattr(f, 'chopped', False), 'cooked_stage': getattr(f, 'cooked_stage', 0)}
                     if _is_prep_ready(f_dict): plated.append(f_dict['food_name'])
                ostate['ingredients_on_plate'] = plated
            elif hasattr(item, 'food_name'):
                fn = item.food_name
                if fn in req:
                     f_dict = {'food_name': fn, 'chopped': getattr(item, 'chopped', False), 'cooked_stage': getattr(item, 'cooked_stage', 0)}
                     ready = _is_prep_ready(f_dict)
                     if ready: ostate['ingredients_ready'].append((fn, cpos, None))
                     else: ostate['ingredients_in_progress'].append((fn, cpos, None, "MAP"))

        for cpos in self.cookers:
            tile = c.get_tile(team, cpos[0], cpos[1]); item = getattr(tile, 'item', None)
            if item and isinstance(item, Pan) and item.food and item.food.food_name in req:
                fn = item.food.food_name
                if item.food.cooked_stage >= 1: ostate['ingredients_ready'].append((fn, cpos, None))
                else: ostate['ingredients_in_progress'].append((fn, cpos, None, "COOKING"))

        for bid in self.bot_ids:
            bs = c.get_bot_state(bid); h = bs.get('holding')
            if h and h.get('type') == 'Plate' and not h.get('dirty'):
                ostate['plate_held_by'] = bid
                plated = []
                for f in h.get('food', []):
                     if _is_prep_ready(f): plated.append(f['food_name'])
                ostate['ingredients_on_plate'] = plated
            elif h and h.get('type') == 'Food' and h['food_name'] in req:
                fn = h['food_name']; ready = _is_prep_ready(h)
                if ready: ostate['ingredients_ready'].append((fn, (bs['x'], bs['y']), bid))
                else: ostate['ingredients_in_progress'].append((fn, (bs['x'], bs['y']), bid, "HAND"))

    def _reserve(self, c, pos):
        t = c.get_turn()
        if t not in self.reserved_stations: self.reserved_stations[t] = set()
        self.reserved_stations[t].add(pos)
    def _is_res(self, c, pos): return pos in self.reserved_stations.get(c.get_turn(), set())

    def _role_assembler(self, c, bid, order, ostate, team):
        bs = c.get_bot_state(bid); h = bs.get('holding'); bpos = (bs['x'], bs['y'])
        if h and h.get('type') == 'Plate' and h.get('dirty'):
             sinks = [s for s in self.map_sinks() if not self._is_res(c, s)]
             if sinks:
                  t = self._get_nearest(bpos, sinks)
                  if t and self._move_toward(c, bid, t, team): self._reserve(c, t); c.put_dirty_plate_in_sink(bid, t[0], t[1])
                  return
             trash = self._get_nearest(bpos, self.trashes)
             if trash and self._move_toward(c, bid, trash, team): c.trash(bid, trash[0], trash[1])
             return
        if h and h.get('type') == 'Food':
            fn = h['food_name']; info = INGREDIENT_INFO.get(fn, {})
            ready = True
            if info.get('chop') and not h.get('chopped'): ready = False
            if info.get('cook') and h.get('cooked_stage', 0) < 1: ready = False
            
            if ready and fn in order['required'] and fn not in ostate['ingredients_on_plate']:
                if ostate['plate_pos']:
                    if self._move_toward(c, bid, ostate['plate_pos'], team): c.add_food_to_plate(bid, ostate['plate_pos'][0], ostate['plate_pos'][1])
                    return
                elif ostate.get('plate_held_by') == bid: c.add_food_to_plate(bid, bpos[0], bpos[1]); return
            trash = self._get_nearest(bpos, self.trashes)
            if trash and self._move_toward(c, bid, trash, team): c.trash(bid, trash[0], trash[1])
            return
        if h and h.get('type') == 'Plate':
             ready = [x for x in ostate['ingredients_ready'] if x[2] is None and not self._is_res(c, x[1])]
             if ready:
                  ti = min(ready, key=lambda x: self._chebyshev_dist(bpos, x[1]))
                  if self._move_toward(c, bid, ti[1], team): self._reserve(c, ti[1]); c.add_food_to_plate(bid, ti[1][0], ti[1][1])
                  return
             still = [ing for ing in order['required'] if ing not in ostate['ingredients_on_plate'] and not INGREDIENT_INFO[ing].get('chop') and not INGREDIENT_INFO[ing].get('cook')]
             if still and self.shops:
                  sh = self._get_nearest(bpos, [s for s in self.shops if not self._is_res(c, s)])
                  if sh and self._move_toward(c, bid, sh, team): self._reserve(c, sh); c.buy(bid, FoodType[still[0]], sh[0], sh[1])
                  return
             self._role_idle(c, bid, team); return
        if not ostate['plate_pos'] and not ostate.get('plate_held_by'):
            stables = [s for s in self.sink_tables if getattr(c.get_tile(team, s[0], s[1]), 'num_clean_plates', 0) > 0 and not self._is_res(c, s)]
            if stables:
                t = self._get_nearest(bpos, stables); 
                if t and self._move_toward(c, bid, t, team): self._reserve(c, t); c.take_clean_plate(bid, t[0], t[1])
                return
            sinks = [s for s in self.map_sinks() if getattr(c.get_tile(team, s[0], s[1]), 'num_dirty_plates', 0) > 0 and not self._is_res(c, s)]
            if sinks:
                t = self._get_nearest(bpos, sinks)
                if t and self._move_toward(c, bid, t, team): self._reserve(c, t); c.wash_sink(bid, t[0], t[1])
                return
            if self.shops:
                s = self._get_nearest(bpos, [sh for sh in self.shops if not self._is_res(c, sh)])
                if s and self._move_toward(c, bid, s, team): self._reserve(c, s); c.buy(bid, ShopCosts.PLATE, s[0], s[1])
            return
        ready = [x for x in ostate['ingredients_ready'] if x[2] is None and not self._is_res(c, x[1])]
        if ready:
            fi = min(ready, key=lambda x: self._chebyshev_dist(bpos, x[1]))
            if self._move_toward(c, bid, fi[1], team):
                self._reserve(c, fi[1]); tile = c.get_tile(team, fi[1][0], fi[1][1])
                if tile and isinstance(tile.item, Pan): c.take_from_pan(bid, fi[1][0], fi[1][1])
                else: c.pickup(bid, fi[1][0], fi[1][1])
            return
        self._role_idle(c, bid, team)

    def _role_fetcher(self, c, bid, order, ostate, team):
        bs = c.get_bot_state(bid); h = bs.get('holding'); bpos = (bs['x'], bs['y'])
        if h and h.get('type') == 'Food':
            fn = h['food_name']; info = INGREDIENT_INFO.get(fn, {})
            if fn not in order['required']:
                trash = self._get_nearest(bpos, self.trashes)
                if trash and self._move_toward(c, bid, trash, team): c.trash(bid, trash[0], trash[1])
                return
            if info.get('chop') and not h.get('chopped'):
                ct = self._get_free_stat(c, team, bpos, self.counters)
                if ct and self._move_toward(c, bid, ct, team): self._reserve(c, ct); c.place(bid, ct[0], ct[1])
                return
            if info.get('cook') and h.get('cooked_stage', 0) < 1:
                ck = self._get_avail_ck(c, team, bpos)
                if ck and self._move_toward(c, bid, ck, team): self._reserve(c, ck); c.place(bid, ck[0], ck[1])
                return
            if ostate['plate_pos']:
                  if self._move_toward(c, bid, ostate['plate_pos'], team): c.add_food_to_plate(bid, ostate['plate_pos'][0], ostate['plate_pos'][1])
                  return
            ct = self._get_free_stat(c, team, bpos, self.counters)
            if ct and self._move_toward(c, bid, ct, team): self._reserve(c, ct); c.place(bid, ct[0], ct[1])
            return
        for cpos in self.counters:
             if self._is_res(c, cpos): continue
             t = c.get_tile(team, cpos[0], cpos[1]); item = getattr(t, 'item', None)
             if hasattr(item, 'food_name') and item.food_name in order['required'] and not item.chopped and INGREDIENT_INFO[item.food_name].get('chop'):
                  if self._move_toward(c, bid, cpos, team): self._reserve(c, cpos); c.chop(bid, cpos[0], cpos[1])
                  return
        still = [ing for ing in order['required'] if ing not in ostate['ingredients_on_plate'] and ing not in [x[0] for x in ostate['ingredients_ready']] and ing not in [x[0] for x in ostate['ingredients_in_progress']]]
        if still and self.shops:
             sh = self._get_nearest(bpos, [s for s in self.shops if not self._is_res(c, s)])
             if sh and self._move_toward(c, bid, sh, team): self._reserve(c, sh); c.buy(bid, FoodType[still[0]], sh[0], sh[1])
             return
        for ii in ostate['ingredients_in_progress']:
             if ii[2] is None and ii[3] == "MAP" and INGREDIENT_INFO[ii[0]].get('chop'):
                  if not self._is_res(c, ii[1]) and self._move_toward(c, bid, ii[1], team): self._reserve(c, ii[1]); c.chop(bid, ii[1][0], ii[1][1])
                  return
        self._role_idle(c, bid, team)

    def _role_idle(self, c, bid, team):
        bpos = self._get_bot_pos(c, bid)
        sinks = [s for s in self.map_sinks() if getattr(c.get_tile(team, s[0], s[1]), 'num_dirty_plates', 0) > 0 and not self._is_res(c, s)]
        if sinks:
             t = self._get_nearest(bpos, sinks)
             if t and self._move_toward(c, bid, t, team): self._reserve(c, t); c.wash_sink(bid, t[0], t[1])
             return
        wait_spots = self.counters + self.cookers + self.submits
        target = self._get_nearest(bpos, wait_spots)
        if target: self._move_toward(c, bid, target, team)

    def _state_submit(self, c, bid, order, ostate, team):
        bs = c.get_bot_state(bid); h = bs.get('holding'); bpos = (bs['x'], bs['y'])
        if h and h.get('type') == 'Plate':
             t = self._get_nearest(bpos, self.submits)
             if t and self._move_toward(c, bid, t, team):
                  if c.submit(bid, t[0], t[1]):
                       self._log(f"Bot {bid} SUCCESS: Order {order['order_id']} submitted.")
                       self.bot_tasks[bid]['status'] = 'COMPLETE'
             return
        if ostate['plate_pos']:
             if self._move_toward(c, bid, ostate['plate_pos'], team): c.pickup(bid, ostate['plate_pos'][0], ostate['plate_pos'][1])

    def _get_free_stat(self, c, t, p, stats):
        free = [s for s in stats if getattr(c.get_tile(t, s[0], s[1]), 'item', None) is None and not self._is_res(c, s)]
        return self._get_nearest(p, free)

    def _get_avail_ck(self, c, t, p):
        free = [ck for ck in self.cookers if isinstance(getattr(c.get_tile(t, ck[0], ck[1]), 'item', None), Pan) and c.get_tile(t, ck[0], ck[1]).item.food is None and not self._is_res(c, ck)]
        return self._get_nearest(p, free)

    def map_sinks(self):
        s = []
        for y in range(self.map.height):
            for x in range(self.map.width):
                if self.map.tiles[x][y].tile_name == "SINK": s.append((x, y))
        return s
