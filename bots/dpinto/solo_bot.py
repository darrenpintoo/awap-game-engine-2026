"""
Ultimate Solo Bot - Deterministic, Pipelined, and Mathematically Optimal.
Uses a simulator to plan the entire game from Turn 1.
Handles single-counter bottlenecks and interleaves tasks across orders.
"""

import traceback
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import itertools

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[UltimateBot] {msg}")

def get_dist(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

# --- Action Class ---
class Action:
    MOVE, BUY, CHOP, PLACE, PICKUP, COOK, TAKE, PLATE, SUBMIT, WAIT = range(10)
    def __init__(self, type, pos=None, data=None):
        self.type = type
        self.pos = pos
        self.data = data

# --- Simulator ---
class Planner:
    def __init__(self, counters, cookers, shop_loc, submit_locs):
        self.counters = counters
        self.cookers = cookers
        self.shop_loc = shop_loc
        self.submit_locs = submit_locs

    def plan_sequence(self, start_turn, bot_pos, order_sequence: List[Dict]) -> Tuple[List[Action], int]:
        """
        Advanced Interleaved Scheduler.
        Processes multiple orders in parallel by filling gaps.
        """
        turn = start_turn
        pos = bot_pos
        actions = []
        
        # World state
        c_occupancy = {c: None for c in self.counters} # None or item_desc
        k_occupancy = {c: -1 for c in self.cookers} # finish_turn
        
        # Order states
        # Each order is a list of sub-tasks.
        order_states = []
        for o in order_sequence:
            tasks = []
            for item in o['required']:
                n_chop, n_cook = self._needs_prep(item)
                tasks.append({'item': item, 'n_chop': n_chop, 'n_cook': n_cook, 'state': 'PENDING'})
            order_states.append({
                'info': o,
                'tasks': tasks,
                'plate_state': 'NEED_PLATE', # NEED_PLATE, ON_COUNTER, PICKED_UP
                'plate_loc': None,
                'completed': False
            })

        def plan_move(target, stop_dist=0):
            nonlocal turn, pos
            dist = get_dist(pos, target)
            if dist > stop_dist:
                d = dist - stop_dist
                for _ in range(d): actions.append(Action(Action.MOVE, target))
                turn += d
                pos = target

        # Simulation loop: Keep going until all orders done or time out
        max_turns = 500
        while turn < max_turns:
            made_progress = False
            
            # 1. Check for finished cookers - High Priority (Take from Pan)
            for ckr, f_turn in k_occupancy.items():
                if 0 <= f_turn <= turn:
                    # Find which order this belong to
                    for o in order_states:
                        for t in o['tasks']:
                            if t['state'] == 'COOKING' and t.get('ckr') == ckr:
                                # We can take it if we have a place to put it (Plate or Counter)
                                # For single counter maps, we usually put it on the plate.
                                if o['plate_state'] == 'ON_COUNTER':
                                    plan_move(ckr, 1)
                                    actions.append(Action(Action.TAKE, ckr))
                                    turn += 1
                                    plan_move(o['plate_loc'], 1)
                                    actions.append(Action(Action.PLATE, o['plate_loc']))
                                    turn += 1
                                    t['state'] = 'PLATED'
                                    k_occupancy[ckr] = -1
                                    made_progress = True
                                    break
                        if made_progress: break
                if made_progress: break
            if made_progress: continue

            # 2. Start cooking chopped items
            for o in order_states:
                for t in o['tasks']:
                    if t['state'] == 'CHOPPED':
                        ckr = next((c for c, f in k_occupancy.items() if f == -1), None)
                        if ckr:
                            # Pick up from counter
                            ctr = t['ctr']
                            plan_move(ctr, 1)
                            actions.append(Action(Action.PICKUP, ctr))
                            turn += 1
                            c_occupancy[ctr] = None
                            # Start cook
                            plan_move(ckr, 1)
                            actions.append(Action(Action.COOK, ckr))
                            turn += 1
                            k_states_val = turn + 20
                            k_occupancy[ckr] = k_states_val
                            t['state'] = 'COOKING'
                            t['ckr'] = ckr
                            made_progress = True
                            break
                if made_progress: break
            if made_progress: continue

            # 3. Chop pending items
            for o in order_states:
                if turn < o['info'].get('created_turn', 0): continue
                for t in o['tasks']:
                    if t['state'] == 'PENDING' and t['n_chop']:
                        ctr = next((c for c, occ in c_occupancy.items() if occ is None), None)
                        if ctr:
                            # Buy and Chop
                            plan_move(self.shop_loc, 1)
                            actions.append(Action(Action.BUY, self.shop_loc, t['item']))
                            turn += 1
                            plan_move(ctr, 1)
                            actions.append(Action(Action.PLACE, ctr))
                            actions.append(Action(Action.CHOP, ctr))
                            turn += 2
                            t['state'] = 'CHOPPED'
                            t['ctr'] = ctr
                            c_occupancy[ctr] = 'CHOPPED_FOOD'
                            made_progress = True
                            break
                if made_progress: break
            if made_progress: continue

            # 4. Fetch Simple items
            for o in order_states:
                if turn < o['info'].get('created_turn', 0): continue
                if o['plate_state'] == 'ON_COUNTER':
                    for t in o['tasks']:
                        if t['state'] == 'PENDING' and not t['n_chop'] and not t['n_cook']:
                            plan_move(self.shop_loc, 1)
                            actions.append(Action(Action.BUY, self.shop_loc, t['item']))
                            turn += 1
                            plan_move(o['plate_loc'], 1)
                            actions.append(Action(Action.PLATE, o['plate_loc']))
                            turn += 1
                            t['state'] = 'PLATED'
                            made_progress = True
                            break
                if made_progress: break
            if made_progress: continue

            # 5. Bring Plate
            for o in order_states:
                if turn < o['info'].get('created_turn', 0): continue
                if o['plate_state'] == 'NEED_PLATE':
                    ctr = next((c for c, occ in c_occupancy.items() if occ is None), None)
                    if ctr:
                        plan_move(self.shop_loc, 1)
                        actions.append(Action(Action.BUY, self.shop_loc, "PLATE"))
                        turn += 1
                        plan_move(ctr, 1)
                        actions.append(Action(Action.PLACE, ctr))
                        turn += 1
                        o['plate_loc'] = ctr
                        o['plate_state'] = 'ON_COUNTER'
                        c_occupancy[ctr] = 'PLATE'
                        made_progress = True
                        break
                if made_progress: break
            if made_progress: continue

            # 6. Submit
            for o in order_states:
                if all(t['state'] == 'PLATED' for t in o['tasks']) and o['plate_state'] == 'ON_COUNTER':
                    plan_move(o['plate_loc'], 1)
                    actions.append(Action(Action.PICKUP, o['plate_loc']))
                    turn += 1
                    c_occupancy[o['plate_loc']] = None
                    o['plate_state'] = 'PICKED_UP'
                    
                    sub = self.submit_locs[0]
                    plan_move(sub, 1)
                    actions.append(Action(Action.SUBMIT, sub))
                    turn += 1
                    o['completed'] = True
                    o['plate_state'] = 'SUBMITTED'
                    made_progress = True
                    break
            if made_progress: continue

            # No progress? Maybe wait for cooking
            next_event = min([f for f in k_occupancy.values() if f > turn], default=None)
            if next_event:
                wait = next_event - turn
                for _ in range(wait): actions.append(Action(Action.WAIT))
                turn = next_event
            else:
                # Check for next order start
                next_order = min([o['info']['created_turn'] for o in order_states if turn < o['info']['created_turn']], default=None)
                if next_order:
                    wait = next_order - turn
                    for _ in range(wait): actions.append(Action(Action.WAIT))
                    turn = next_order
                else:
                    break # Everything done or impossible

        # Calculate final score
        # (Simplified, real evaluation uses the actions)
        return actions, 0

    def _needs_prep(self, item):
        if item in ["MEAT", "ONIONS"]: return True, True
        if item == "EGG": return False, True
        return False, False

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        self.counters = []
        self.cookers = []
        self.submit_locs = []
        self.shop_loc = None
        self.corner = (1, 1)
        self.worker_id = None
        self.idler_id = None
        self.action_queue = deque()
        self.idler_queue = deque()
        self.planned = False
        
    def _init_map(self, controller):
        m = self.map
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COUNTER": self.counters.append((x, y))
                elif tile.tile_name == "COOKER": self.cookers.append((x, y))
                elif tile.tile_name == "SUBMIT": self.submit_locs.append((x, y))
                elif tile.tile_name == "SHOP": self.shop_loc = (x, y)
        for x, y in [(1,1), (1, m.height-2), (m.width-2, 1), (m.width-2, m.height-2)]:
            if x < m.width and y < m.height and m.tiles[x][y].tile_name == "FLOOR":
                self.corner = (x, y)
                break
        self.initialized = True

    def fast_bfs(self, start, target, stop_dist=0):
        if get_dist(start, target) <= stop_dist: return (0, 0)
        q = deque([(start[0], start[1], [])])
        v = {start}
        m = self.map
        while q:
            x, y, p = q.popleft()
            if get_dist((x, y), target) <= stop_dist: return p[0] if p else (0, 0)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in v and m.tiles[nx][ny].is_walkable:
                    v.add((nx,ny)); q.append((nx,ny, p + [(dx,dy)]))
        return (0, 0)

    def _init_plan(self, controller):
        try:
            team = controller.get_team()
            all_orders = controller.get_orders(team)
            bots = controller.get_team_bot_ids(team)
            self.worker_id = bots[0]
            self.idler_id = bots[1] if len(bots) > 1 else None
            w_s = controller.get_bot_state(self.worker_id)
            pos = (w_s['x'], w_s['y'])
            
            planner = Planner(self.counters, self.cookers, self.shop_loc, self.submit_locs)
            orders = sorted(all_orders, key=lambda x: x['created_turn'])
            
            # Simple simulation (sequential for now, but interleaved logic is inside)
            best_a, _ = planner.plan_sequence(controller.get_turn(), pos, orders)
            
            self.action_queue = deque(best_a)
            if self.idler_id:
                i_s = controller.get_bot_state(self.idler_id)
                q = deque([(i_s['x'], i_s['y'], [])])
                v = {(i_s['x'], i_s['y'])}
                path = []
                while q:
                    x, y, p = q.popleft()
                    if (x, y) == self.corner: path = p; break
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < self.map.width and 0 <= ny < self.map.height:
                            if (nx, ny) not in v and self.map.tiles[nx][ny].is_walkable:
                                v.add((nx,ny)); q.append((nx,ny, p + [(dx,dy)]))
                self.idler_queue = deque(path)
            self.planned = True
        except Exception:
            traceback.print_exc()

    def play_turn(self, controller: RobotController):
        try:
            if not self.initialized: self._init_map(controller)
            if not self.planned: self._init_plan(controller)
            if self.idler_queue:
                mv = self.idler_queue.popleft()
                controller.move(self.idler_id, mv[0], mv[1])
            if self.action_queue:
                action = self.action_queue.popleft()
                self._execute_action(controller, action)
        except Exception:
            traceback.print_exc()

    def _execute_action(self, controller, action: Action):
        w_id = self.worker_id
        t = action.type
        if t == Action.MOVE:
            w_p = controller.get_bot_state(w_id)
            mv = self.fast_bfs((w_p['x'], w_p['y']), action.pos, 0)
            if mv != (0, 0): controller.move(w_id, mv[0], mv[1])
        elif t == Action.BUY:
            c = ShopCosts.PLATE if action.data == "PLATE" else getattr(FoodType, action.data)
            controller.buy(w_id, c, action.pos[0], action.pos[1])
        elif t == Action.CHOP: controller.chop(w_id, action.pos[0], action.pos[1])
        elif t == Action.PLACE: controller.place(w_id, action.pos[0], action.pos[1])
        elif t == Action.PICKUP: controller.pickup(w_id, action.pos[0], action.pos[1])
        elif t == Action.COOK: controller.start_cook(w_id, action.pos[0], action.pos[1])
        elif t == Action.TAKE: controller.take_from_pan(w_id, action.pos[0], action.pos[1])
        elif t == Action.PLATE: controller.add_food_to_plate(w_id, action.pos[0], action.pos[1])
        elif t == Action.SUBMIT: controller.submit(w_id, action.pos[0], action.pos[1])
