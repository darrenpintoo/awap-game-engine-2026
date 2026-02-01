"""
JuniorChampion Bot - AWAP 2026
Simplified recipe-based approach with assembly point pattern.
"""
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


FOOD_LUT = {
    "EGG": FoodType.EGG, "ONIONS": FoodType.ONIONS,
    "MEAT": FoodType.MEAT, "NOODLES": FoodType.NOODLES, "SAUCE": FoodType.SAUCE,
}


class BotPlayer:
    def __init__(self, map_copy):
        self.w = map_copy.width
        self.h = map_copy.height
        
        self.tile_locs = {}
        self.walkable = set()
        for x in range(self.w):
            for y in range(self.h):
                t = map_copy.tiles[x][y]
                self.tile_locs.setdefault(t.tile_name, []).append((x, y))
                if t.is_walkable:
                    self.walkable.add((x, y))
        
        self._dist = {}
        self._next = {}
        self._precompute_bfs()
        
        self._shops = self.tile_locs.get('SHOP', [])
        self._submits = self.tile_locs.get('SUBMIT', [])
        self._counters = self.tile_locs.get('COUNTER', [])
        self._cookers = self.tile_locs.get('COOKER', [])
        self._trash = self.tile_locs.get('TRASH', [])
        
        self.tasks = {}
        self.assigned = set()
        self.completed = set()
        self.failed = {}
        self.team = None
    
    def _precompute_bfs(self):
        for src in self.walkable:
            dist = {src: 0}
            first = {src: (0, 0)}
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
                            first[(nx, ny)] = first[(cx, cy)] if (cx, cy) != src else (dx, dy)
                            q.append((nx, ny))
            self._dist[src] = dist
            self._next[src] = first
    
    def _adj(self, tx, ty):
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (tx + dx, ty + dy) in self.walkable:
                    result.append((tx + dx, ty + dy))
        return result
    
    def _dist_to(self, sx, sy, tx, ty):
        adj = self._adj(tx, ty)
        if not adj or (sx, sy) not in self._dist:
            return 9999
        return min((self._dist[(sx, sy)].get(a, 9999) for a in adj), default=9999)
    
    def _nearest(self, x, y, tile_type):
        locs = self.tile_locs.get(tile_type, [])
        if not locs:
            return None
        reachable = [(loc, self._dist_to(x, y, loc[0], loc[1])) for loc in locs]
        reachable = [(loc, d) for loc, d in reachable if d < 9999]
        if not reachable:
            return None
        return min(reachable, key=lambda x: x[1])[0]
    
    def _move_to(self, c, bid, bs, tile):
        bx, by = bs['x'], bs['y']
        tx, ty = tile
        if max(abs(bx - tx), abs(by - ty)) <= 1:
            return True
        
        adj = self._adj(tx, ty)
        if not adj:
            return False
        
        src = (bx, by)
        if src not in self._dist:
            return False
        
        best = min(adj, key=lambda a: self._dist[src].get(a, 9999))
        if best not in self._next.get(src, {}):
            return False
        
        dx, dy = self._next[src][best]
        if c.can_move(bid, dx, dy):
            c.move(bid, dx, dy)
        return False
    
    def play_turn(self, c: RobotController):
        turn = c.get_turn()
        team = c.get_team()
        if self.team is None:
            self.team = team
        
        bots = c.get_team_bot_ids(team)
        orders = c.get_orders(team)
        
        for bid in bots:
            if bid not in self.tasks:
                self.tasks[bid] = {'recipe': [], 'step': 0, 'oid': None, 'stuck': 0}
        
        # Clean up expired
        active = {o['order_id'] for o in orders if o['is_active']}
        for bid, t in self.tasks.items():
            if t['oid'] and t['oid'] not in active:
                self.assigned.discard(t['oid'])
                t['recipe'] = []
                t['step'] = 0
                t['oid'] = None
        
        # Assign orders to idle bots
        for bid in bots:
            t = self.tasks[bid]
            if t['recipe'] and t['step'] < len(t['recipe']):
                continue
            
            bs = c.get_bot_state(bid)
            if not bs:
                continue
            
            # If holding, trash first
            if bs.get('holding'):
                trash = self._nearest(bs['x'], bs['y'], 'TRASH')
                if trash:
                    t['recipe'] = [('trash', trash)]
                    t['step'] = 0
                    t['oid'] = None
                continue
            
            # Find best order
            best = None
            best_score = -9999
            for o in orders:
                if not o['is_active'] or o['order_id'] in self.assigned or o['order_id'] in self.completed:
                    continue
                if o['order_id'] in self.failed and turn - self.failed[o['order_id']] < 20:
                    continue
                
                score = self._score(o, bs['x'], bs['y'], turn)
                if score > best_score:
                    best_score = score
                    best = o
            
            if best:
                recipe = self._make_recipe(c, bid, best, bs)
                if recipe:
                    t['recipe'] = recipe
                    t['step'] = 0
                    t['oid'] = best['order_id']
                    self.assigned.add(best['order_id'])
        
        # Execute bots
        for bid in bots:
            self._execute(c, bid, turn)
    
    def _score(self, order, x, y, turn):
        req = order['required']
        cost = ShopCosts.PLATE.buy_cost
        for fn in req:
            ft = FOOD_LUT.get(fn)
            if ft:
                cost += ft.buy_cost
        
        profit = order['reward'] - cost
        if profit <= 0:
            return -9999
        
        tleft = order['expires_turn'] - turn
        if tleft < 15:
            return -9999
        
        n_cook = sum(1 for fn in req if FOOD_LUT.get(fn) and FOOD_LUT[fn].can_cook)
        n_chop = sum(1 for fn in req if FOOD_LUT.get(fn) and FOOD_LUT[fn].can_chop)
        # Conservative time estimate
        est = 15 + len(req) * 4 + n_cook * 25 + n_chop * 5
        
        # Conservative feasibility check
        if est > tleft * 0.7:
            return -9999
        
        return profit / est
    
    def _make_recipe(self, c, bid, order, bs):
        bx, by = bs['x'], bs['y']
        
        shop = self._nearest(bx, by, 'SHOP')
        submit = self._nearest(bx, by, 'SUBMIT')
        if not shop or not submit:
            return None
        
        # Assembly point: walkable adjacent to shop, closest to submit
        shop_adj = self._adj(shop[0], shop[1])
        if not shop_adj:
            return None
        assembly = min(shop_adj, key=lambda a: self._dist_to(a[0], a[1], submit[0], submit[1]))
        
        # Parse ingredients
        ingredients = [FOOD_LUT[fn] for fn in order['required'] if fn in FOOD_LUT]
        cook_chop = [f for f in ingredients if f.can_cook and f.can_chop]
        cook_only = [f for f in ingredients if f.can_cook and not f.can_chop]
        chop_only = [f for f in ingredients if f.can_chop and not f.can_cook]
        simple = [f for f in ingredients if not f.can_cook and not f.can_chop]
        all_cook = cook_chop + cook_only
        
        # Find resources
        counter = self._nearest(bx, by, 'COUNTER') if (cook_chop or chop_only) else None
        cooker = self._nearest(bx, by, 'COOKER') if all_cook else None
        
        if (cook_chop or chop_only) and not counter:
            return None
        if all_cook and not cooker:
            return None
        
        steps = []
        trash = self._nearest(bx, by, 'TRASH')
        
        # Go to assembly
        steps.append(('goto', assembly))
        
        # Buy plate, place at assembly
        steps.append(('buy', ShopCosts.PLATE, shop))
        steps.append(('place', assembly))
        
        # Simple items
        for ft in simple:
            steps.append(('buy', ft, shop))
            steps.append(('add', assembly))
        
        # Chop-only items
        for ft in chop_only:
            steps.append(('buy', ft, shop))
            steps.append(('place', counter))
            steps.append(('chop', counter))
            steps.append(('pickup', counter))
            steps.append(('goto', assembly))
            steps.append(('add', assembly))
        
        # Cook items
        for ft in all_cook:
            steps.append(('buy', ft, shop))
            if ft.can_chop:
                steps.append(('place', counter))
                steps.append(('chop', counter))
                steps.append(('pickup', counter))
            steps.append(('cook', cooker))
            steps.append(('wait_cook', cooker))
            steps.append(('goto', assembly))
            steps.append(('add', assembly))
        
        # Submit
        steps.append(('pickup', assembly))
        steps.append(('submit', submit))
        
        return steps
    
    def _execute(self, c, bid, turn):
        t = self.tasks[bid]
        if not t['recipe'] or t['step'] >= len(t['recipe']):
            return
        
        bs = c.get_bot_state(bid)
        if not bs:
            return
        
        step = t['recipe'][t['step']]
        action = step[0]
        done = False
        
        if action == 'goto':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = True
        
        elif action == 'buy':
            item, shop = step[1], step[2]
            if self._move_to(c, bid, bs, shop):
                done = c.buy(bid, item, shop[0], shop[1])
        
        elif action == 'place':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = c.place(bid, target[0], target[1])
        
        elif action == 'pickup':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = c.pickup(bid, target[0], target[1])
        
        elif action == 'chop':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                c.chop(bid, target[0], target[1])
                # Check if done
                tile = c.get_tile(c.get_team(), target[0], target[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Food):
                    done = tile.item.chopped
        
        elif action == 'cook':
            cooker = step[1]
            if self._move_to(c, bid, bs, cooker):
                done = c.start_cook(bid, cooker[0], cooker[1]) or c.place(bid, cooker[0], cooker[1])
        
        elif action == 'wait_cook':
            cooker = step[1]
            if self._move_to(c, bid, bs, cooker):
                tile = c.get_tile(c.get_team(), cooker[0], cooker[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                    if tile.item.food and tile.item.food.cooked_stage >= 1:
                        done = c.take_from_pan(bid, cooker[0], cooker[1])
        
        elif action == 'add':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = c.add_food_to_plate(bid, target[0], target[1])
        
        elif action == 'submit':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = c.submit(bid, target[0], target[1])
        
        elif action == 'trash':
            target = step[1]
            if self._move_to(c, bid, bs, target):
                done = c.trash(bid, target[0], target[1])
        
        if done:
            t['step'] += 1
            t['stuck'] = 0
            
            if t['step'] >= len(t['recipe']):
                if t['oid']:
                    self.completed.add(t['oid'])
                    self.assigned.discard(t['oid'])
                t['recipe'] = []
                t['step'] = 0
                t['oid'] = None
        else:
            pos = (bs['x'], bs['y'])
            if pos == t.get('_last'):
                t['stuck'] += 1
            else:
                t['stuck'] = 0
            t['_last'] = pos
            
            if t['stuck'] > 25:
                if t['oid']:
                    self.assigned.discard(t['oid'])
                    self.failed[t['oid']] = turn
                t['recipe'] = []
                t['step'] = 0
                t['oid'] = None
