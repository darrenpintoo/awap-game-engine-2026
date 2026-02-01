"""
Ultimate Champion Bot v23 - Harmonic Duo Coordination.
"""

import traceback
from collections import deque
from typing import List, Tuple, Dict, Optional, Any

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[ChampionBot] {msg}")

def get_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class Goal:
    BUY, CHOP, PLACE, PICKUP, COOK, TAKE, PLATE, SUBMIT = range(8)
    def __init__(self, type, pos, data=None):
        self.type = type; self.pos = pos; self.data = data
    def __repr__(self):
        names = ["BUY", "CHOP", "PLACE", "PICKUP", "COOK", "TAKE", "PLATE", "SUBMIT"]
        return f"{names[self.type]}@{self.pos} ({self.data})"

class Simulator:
    def __init__(self, shop, sub, counters, tables, cookers, ctr_items, k_items, locks=None):
        self.shop = shop; self.sub = sub; self.counters = counters; self.tables = tables; self.cookers = cookers
        self.ctr_items = ctr_items; self.k_items = k_items; self.locks = locks or []

    def plan(self, turn, pos, holding, orders, bot_role=0) -> List[Goal]:
        goals = []
        st, sp, sh = turn, pos, self._parse_h(holding)
        all_placeable = self.counters + self.tables
        c_occ = {c: None for c in all_placeable}
        for c, it in self.ctr_items.items():
            if it: c_occ[c] = {'name': it.get('food_name'), 'chopped': it.get('chopped'), 'type': it.get('type'), 'food': it.get('food', [])}
        
        k_occ = {c: -1 for c in self.cookers}
        for c, it in self.k_items.items():
            if it and it.get('type') == 'Pan' and it.get('food'): k_occ[c] = st + 10

        o_states = []
        for o in orders:
            tasks = []
            for item in o['required']:
                state = 'PENDING'; c_loc, ckr_loc = None, None
                for c, info in c_occ.items():
                    if info and info['type'] == 'Food' and info['name'] == item and not info.get('bound'):
                        state = 'CHOPPED' if info['chopped'] else 'PLACED'; c_loc = c; info['bound'] = True; break
                if state == 'PENDING':
                    for ckr in self.cookers:
                        it = self.k_items[ckr].get('food')
                        if it and it.get('food_name') == item: state = 'COOKING'; ckr_loc = ckr; break
                if state == 'PENDING':
                    for c, info in c_occ.items():
                        if info and info['type'] == 'Plate' and not info.get('bound'):
                            for f in info['food']:
                                if f.get('food_name') == item: state = 'DONE'; c_loc = c; break
                tasks.append({'item': item, 'state': state, 'ctr': c_loc, 'ckr': ckr_loc})
            o_states.append({'info': o, 'tasks': tasks, 'p_state': 'NEED', 'p_loc': None, 'done': False})

        for c, info in c_occ.items():
             if info and info['type'] == 'Plate' and not info.get('bound'):
                 for o in o_states:
                     if o['p_state'] == 'NEED': o['p_state'] = 'ON_COUNTER'; o['p_loc'] = c; info['bound'] = True; break

        def add_g(gt, target, data=None):
            nonlocal st, sp, sh
            dist = get_dist(sp, target); goals.append(Goal(gt, target, data)); st += dist + 1; sp = target
            if gt == Goal.BUY: sh = {'type': 'Food', 'name': data} if data != 'PLATE' else {'type': 'Plate'}
            elif gt == Goal.TAKE: sh = {'type': 'Food', 'name': 'Any'}
            elif gt == Goal.PICKUP: sh = {'type': 'Any'} 
            elif gt in [Goal.PLACE, Goal.PLATE, Goal.SUBMIT, Goal.COOK]: sh = None

        while st < 500 and len(goals) < 60:
            prog = False
            # 1. Submit
            for o in o_states:
                if not o['done'] and o['p_state'] == 'READY' and st >= o['info']['created_turn']:
                    if sh is None: add_g(Goal.PICKUP, o['p_loc']); add_g(Goal.SUBMIT, self.sub); o['done'] = True; prog = True; break
                    elif sh and sh.get('type') == 'Plate': add_g(Goal.SUBMIT, self.sub); o['done'] = True; prog = True; break
            if prog: continue

            # 2. Ready
            for o in o_states:
                if not o['done'] and o['p_state'] == 'ON_COUNTER' and all(t['state'] == 'DONE' for t in o['tasks']):
                    o['p_state'] = 'READY'; prog = True; break
            if prog: continue

            # 3. Take
            for ckr, dt in k_occ.items():
                if 0 <= dt <= st:
                    for o in o_states:
                        if o['done'] or o['p_state'] != 'ON_COUNTER': continue
                        for t in o['tasks']:
                            if t['state'] == 'COOKING' and t.get('ckr') == ckr:
                                if sh is None: add_g(Goal.TAKE, ckr); add_g(Goal.PLATE, o['p_loc']); t['state'] = 'DONE'; k_occ[ckr] = -1; prog = True; break
                        if prog: break
                if prog: break
            if prog: continue

            # 4. Work (Chop/Cook/Simple)
            for o_idx, o in enumerate(o_states):
                if o['done']: continue
                # Simple logic: Bot 0 does Order 0, Bot 1 does Order 1, OR they share if one is done
                if bot_role == 0 and o_idx > 0 and len(o_states) > 0 and not o_states[0]['done']: continue
                if bot_role == 1 and o_idx == 0 and len(o_states) > 1 and not o_states[1]['done']: continue

                for t in o['tasks']:
                    if t['state'] == 'CHOPPED' and self._needs_cook(t['item']):
                        ckr = next((c for c, f in k_occ.items() if f == -1 and c not in self.locks), None)
                        if ckr:
                            if sh is None: add_g(Goal.PICKUP, t['ctr']); add_g(Goal.COOK, ckr); t['state'] = 'COOKING'; t['ckr'] = ckr; k_occ[ckr] = st + 20; prog = True; break
                    if self._needs_chop(t['item']):
                        if t['state'] == 'PLACED' and t['ctr'] in self.counters: 
                            if sh is None: add_g(Goal.CHOP, t['ctr']); t['state'] = 'CHOPPED'; prog = True; break
                        elif t['state'] == 'PENDING':
                            ctr = next((c for c in self.counters if c_occ[c] is None and c not in self.locks), None)
                            if ctr:
                                if sh is None: add_g(Goal.BUY, self.shop, t['item']); add_g(Goal.PLACE, ctr); add_g(Goal.CHOP, ctr); t['state'] = 'CHOPPED'; t['ctr'] = ctr; c_occ[ctr] = 'BUSY'; prog = True; break
                    elif t['state'] == 'PENDING' and not self._needs_cook(t['item']):
                        if o['p_state'] == 'ON_COUNTER':
                            if sh is None: add_g(Goal.BUY, self.shop, t['item']); add_g(Goal.PLATE, o['p_loc']); t['state'] = 'DONE'; prog = True; break
                if prog: break
            if prog: continue

            # 5. Plates
            for o_idx, o in enumerate(o_states):
                if o['done'] or o['p_state'] != 'NEED': continue
                if bot_role == 0 and o_idx > 0 and len(o_states) > 0 and not o_states[0]['done']: continue
                if bot_role == 1 and o_idx == 0 and len(o_states) > 1 and not o_states[1]['done']: continue
                ctr = next((c for c in all_placeable if c_occ[c] is None and c not in self.locks), None)
                if ctr:
                    if sh is None: add_g(Goal.BUY, self.shop, "PLATE"); add_g(Goal.PLACE, ctr); o['p_state'] = 'ON_COUNTER'; o['p_loc'] = ctr; c_occ[ctr] = 'BUSY'; prog = True; break
                if prog: break
            if prog: continue

            # ADVANCE
            times = [dt for dt in k_occ.values() if dt > st]
            for o in o_states:
                if not o['done'] and o['info']['created_turn'] > st: times.append(o['info']['created_turn'])
            earliest = min(times, default=None)
            if earliest and earliest < 500: st = earliest; prog = True
            else: break
        return goals

    def _parse_h(self, h):
        if not h: return None
        if h.get('type') == 'Food': return {'type': 'Food', 'name': h['food_name']}
        return {'type': h.get('type')}
    def _needs_chop(self, item): return item in ["MEAT", "ONIONS"]
    def _needs_cook(self, item): return item in ["MEAT", "EGG"]

class BotPlayer:
    def __init__(self, m):
        self.init_done = False; self.shops, self.counters, self.tables, self.cookers, self.submits = [], [], [], [], []
        self.worker_data = {} # {id: {queue, current, last_replan}}

    def _init(self, controller):
        m = controller.get_map(controller.get_team())
        self.shops, self.counters, self.tables, self.cookers, self.submits = [], [], [], [], []
        for x in range(m.width):
            for y in range(m.height):
                if m.is_tile_name(x, y, "SHOP"): self.shops.append((x,y))
                elif m.is_tile_name(x, y, "COUNTER"): self.counters.append((x,y))
                elif m.is_tile_name(x, y, "SINKTABLE"): self.tables.append((x,y))
                elif m.is_tile_name(x, y, "COOKER"): self.cookers.append((x,y))
                elif m.is_tile_name(x, y, "SUBMIT"): self.submits.append((x,y))
        self.init_done = True

    def _replan(self, controller, bid, role):
        try:
            team = controller.get_team(); turn = controller.get_turn(); w_s = controller.get_bot_state(bid)
            orders = [o for o in controller.get_orders(team) if o['completed_turn'] is None and o['expires_turn'] > turn]
            m = controller.get_map(team); ctr_it, k_it = {}, {}
            for c in self.counters + self.tables: 
                tile = m.tiles[c[0]][c[1]]; ctr_it[c] = controller.item_to_public_dict(tile.item) if tile.item else None
            for c in self.cookers:
                tile = m.tiles[c[0]][c[1]]; k_it[c] = controller.item_to_public_dict(tile.item) if tile.item else None
            
            # Locked tiles from the OTHER bot
            locks = []
            for other_id, data in self.worker_data.items():
                if other_id != bid and data.get('current'): locks.append(data['current'].pos)

            shop = min(self.shops, key=lambda s: get_dist((w_s['x'], w_s['y']), s))
            sub = min(self.submits, key=lambda s: get_dist((w_s['x'], w_s['y']), s))
            sim = Simulator(shop, sub, self.counters, self.tables, self.cookers, ctr_it, k_it, locks)
            goals = sim.plan(turn, (w_s['x'], w_s['y']), w_s['holding'], orders, role)
            self.worker_data[bid] = {'queue': deque(goals), 'current': None, 'last_replan': turn}
            log(f"Bot {bid} (Role {role}) Replanned. Goals: {len(goals)}")
        except Exception: traceback.print_exc()

    def play_turn(self, controller: RobotController):
        if not self.init_done: self._init(controller)
        turn = controller.get_turn(); team = controller.get_team(); bots = controller.get_team_bot_ids(team)
        for i, bid in enumerate(bots):
            if bid not in self.worker_data or (turn - self.worker_data[bid]['last_replan']) > 30: self._replan(controller, bid, i)
            data = self.worker_data[bid]
            while not data['current'] and data['queue']: data['current'] = data['queue'].popleft()
            if data['current']: self._execute(controller, bid, data['current'])
            elif turn % 10 == 0: self._replan(controller, bid, i)

    def _execute(self, controller, bid, goal):
        w_s = controller.get_bot_state(bid); pos = (w_s['x'], w_s['y'])
        dist = get_dist(pos, goal.pos)
        if dist > 1:
            mv = self._bfs(controller, pos, goal.pos, 1)
            if mv != (0, 0): controller.move(bid, mv[0], mv[1])
        else:
            success = False; t = goal.type
            if t == Goal.BUY:
                item = ShopCosts.PLATE if goal.data == "PLATE" else getattr(FoodType, goal.data)
                success = controller.buy(bid, item, goal.pos[0], goal.pos[1])
            elif t == Goal.CHOP: success = controller.chop(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.PLACE: success = controller.place(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.PICKUP: success = controller.pickup(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.COOK: success = controller.start_cook(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.TAKE: success = controller.take_from_pan(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.PLATE: success = controller.add_food_to_plate(bid, goal.pos[0], goal.pos[1])
            elif t == Goal.SUBMIT: success = controller.submit(bid, goal.pos[0], goal.pos[1])
            if success: self.worker_data[bid]['current'] = None
            else: self._replan(controller, bid, 0 if bid == controller.get_team_bot_ids(controller.get_team())[0] else 1)

    def _bfs(self, controller, start, target, stop_dist=0):
        if get_dist(start, target) <= stop_dist: return (0, 0)
        m = controller.get_map(controller.get_team())
        q = deque([(start, [])]); v = {start}
        while q:
            (x, y), path = q.popleft()
            if get_dist((x, y), target) <= stop_dist: return path[0]
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < m.width and 0 <= ny < m.height and (nx,ny) not in v and m.is_tile_walkable(nx,ny):
                    v.add((nx,ny)); q.append(((nx,ny), path + [(dx,dy)]))
        return (0, 0)
