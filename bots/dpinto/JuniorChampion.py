"""
JuniorChampion Bot - AWAP 2026
Enhanced recipe-based approach with two-bot coordination.
"""
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food


FOOD_INFO = {
    "EGG": {"type": FoodType.EGG, "chop": False, "cook": True, "cost": 20},
    "ONIONS": {"type": FoodType.ONIONS, "chop": True, "cook": False, "cost": 30},
    "MEAT": {"type": FoodType.MEAT, "chop": True, "cook": True, "cost": 80},
    "NOODLES": {"type": FoodType.NOODLES, "chop": False, "cook": False, "cost": 40},
    "SAUCE": {"type": FoodType.SAUCE, "chop": False, "cook": False, "cost": 10},
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
        self._sinks = self.tile_locs.get('SINK', [])
        self._sinktables = self.tile_locs.get('SINKTABLE', [])
        
        # Strategic counter selection
        self.plate_counter = None
        self.work_counter = None
        self._select_strategic_counters()
        
        # Pre-compute travel distances
        self.dist_shop_counter = None
        self.dist_shop_cooker = None
        self.dist_counter_submit = None
        self._compute_travel_dists()
        
        self.tasks = {}
        self.assigned = set()
        self.completed = set()
        self.failed = {}
        self.team = None
        self.map_area = self.w * self.h
    
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
    
    def _facility_dist(self, pos_a, pos_b):
        """Distance between adjacent walkable tiles of two facilities."""
        if pos_a is None or pos_b is None:
            return 9999
        a_adj = self._adj(pos_a[0], pos_a[1])
        b_adj = self._adj(pos_b[0], pos_b[1])
        if not a_adj or not b_adj:
            return 9999
        best = 9999
        for a in a_adj:
            if a not in self._dist:
                continue
            for b in b_adj:
                d = self._dist[a].get(b, 9999)
                if d < best:
                    best = d
        return best
    
    def _select_strategic_counters(self):
        """Select plate counter near submit, work counter near shop/cookers."""
        if not self._counters:
            return
        
        # Find nearest shop and submit
        shop = self._shops[0] if self._shops else None
        submit = self._submits[0] if self._submits else None
        cooker = self._cookers[0] if self._cookers else None
        
        if len(self._counters) >= 2 and submit:
            # Plate counter: closest to submit with good shop access
            best_plate = None
            best_score = 9999
            for c in self._counters:
                d_submit = self._facility_dist(c, submit)
                d_shop = self._facility_dist(c, shop) if shop else 0
                score = d_submit * 1.0 + d_shop * 0.5
                if score < best_score:
                    best_score = score
                    best_plate = c
            self.plate_counter = best_plate
            
            # Work counter: close to shop and cookers, not plate counter
            best_work = None
            best_score = 9999
            for c in self._counters:
                if c == self.plate_counter:
                    continue
                d_shop = self._facility_dist(c, shop) if shop else 0
                d_cook = self._facility_dist(c, cooker) if cooker else 0
                score = d_shop * 1.0 + d_cook * 0.5
                if score < best_score:
                    best_score = score
                    best_work = c
            self.work_counter = best_work
        elif self._counters:
            self.plate_counter = self._counters[0]
            self.work_counter = self._counters[0] if len(self._counters) == 1 else self._counters[1]
    
    def _compute_travel_dists(self):
        """Pre-compute key travel distances for scoring."""
        shop = self._shops[0] if self._shops else None
        submit = self._submits[0] if self._submits else None
        
        if shop and self._counters:
            self.dist_shop_counter = min(self._facility_dist(shop, c) for c in self._counters)
        if shop and self._cookers:
            self.dist_shop_cooker = min(self._facility_dist(shop, c) for c in self._cookers)
        if self._counters and submit:
            self.dist_counter_submit = min(self._facility_dist(c, submit) for c in self._counters)
    
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
    
    def _find_empty_counter(self, c, x, y, exclude=None):
        """Find empty counter, preferring work_counter."""
        exclude = exclude or []
        team = c.get_team()
        
        # Try work counter first
        if self.work_counter and self.work_counter not in exclude:
            tile = c.get_tile(team, self.work_counter[0], self.work_counter[1])
            if tile and getattr(tile, 'item', None) is None:
                return self.work_counter
        
        # Find any empty counter
        for counter in self._counters:
            if counter in exclude:
                continue
            tile = c.get_tile(team, counter[0], counter[1])
            if tile and getattr(tile, 'item', None) is None:
                d = self._dist_to(x, y, counter[0], counter[1])
                if d < 9999:
                    return counter
        return None
    
    def _find_available_cooker(self, c, x, y):
        """Find cooker with empty pan."""
        team = c.get_team()
        best = None
        best_dist = 9999
        
        for cooker in self._cookers:
            tile = c.get_tile(team, cooker[0], cooker[1])
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    d = self._dist_to(x, y, cooker[0], cooker[1])
                    if d < best_dist:
                        best_dist = d
                        best = cooker
        
        # If no empty pan, return nearest cooker
        if best is None and self._cookers:
            best = min(self._cookers, key=lambda ck: self._dist_to(x, y, ck[0], ck[1]))
        return best
    
    def play_turn(self, c: RobotController):
        turn = c.get_turn()
        team = c.get_team()
        if self.team is None:
            self.team = team
            import sys
            print(f'[{team.name}] Init: shops={len(self._shops)}, counters={len(self._counters)}, cookers={len(self._cookers)}', file=sys.stderr, flush=True)
        
        bots = c.get_team_bot_ids(team)
        orders = c.get_orders(team)
        
        if turn <= 15 or turn % 100 == 0:
            import sys
            print(f'[T{turn}] {team.name}: bots={len(bots)}, orders={len(orders)}, assigned={len(self.assigned)}, completed={len(self.completed)}', file=sys.stderr, flush=True)
        
        for bid in bots:
            if bid not in self.tasks:
                self.tasks[bid] = {'recipe': [], 'step': 0, 'oid': None, 'stuck': 0, 'role': 'main'}
        
        # Assign roles: first bot is main, second is helper
        if len(bots) >= 2:
            self.tasks[bots[0]]['role'] = 'main'
            self.tasks[bots[1]]['role'] = 'helper'
        
        # Clean up expired orders
        active = {o['order_id'] for o in orders if o['is_active']}
        for bid, t in self.tasks.items():
            if t['oid'] and t['oid'] not in active:
                self.assigned.discard(t['oid'])
                t['recipe'] = []
                t['step'] = 0
                t['oid'] = None
        
        # Execute bots
        for bid in bots:
            t = self.tasks[bid]
            if t['role'] == 'helper':
                self._run_helper(c, bid, turn, orders)
            else:
                self._run_main(c, bid, turn, orders)
    
    def _run_main(self, c, bid, turn, orders):
        """Main bot: complete orders."""
        t = self.tasks[bid]
        bs = c.get_bot_state(bid)
        if not bs:
            return
        
        # If has recipe, execute it
        if t['recipe'] and t['step'] < len(t['recipe']):
            self._execute(c, bid, turn)
            return
        
        # Find new order
        if bs.get('holding'):
            trash = self._nearest(bs['x'], bs['y'], 'TRASH')
            if trash:
                t['recipe'] = [('trash', trash)]
                t['step'] = 0
                t['oid'] = None
            return
        
        best = None
        best_score = -9999
        active_orders = [o for o in orders if o['is_active']]
        
        if turn <= 15:
            import sys
            print(f'[T{turn}] Active orders: {len(active_orders)}', file=sys.stderr, flush=True)
        
        for o in orders:
            if not o['is_active'] or o['order_id'] in self.assigned or o['order_id'] in self.completed:
                continue
            if o['order_id'] in self.failed and turn - self.failed[o['order_id']] < 20:
                continue
            
            score = self._score(o, bs['x'], bs['y'], turn)
            
            if turn <= 15:
                import sys
                print(f'[T{turn}] Order {o["order_id"]}: {o["required"]}, score={score}', file=sys.stderr, flush=True)
            
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
            else:
                # Recipe creation failed
                if turn <= 30:
                    print(f'[T{turn}] Recipe FAILED for order {best["order_id"]}: {best["required"]}')
    
    def _run_helper(self, c, bid, turn, orders):
        """Helper bot: wash dishes, prepare plates, assist."""
        bs = c.get_bot_state(bid)
        if not bs:
            return
        
        team = c.get_team()
        holding = bs.get('holding')
        holding_type = holding.get('type') if holding else None
        
        # If holding dirty plate, put in sink
        if holding and holding.get('type') == 'Plate' and holding.get('dirty'):
            sink = self._nearest(bs['x'], bs['y'], 'SINK')
            if sink and self._move_to(c, bid, bs, sink):
                c.put_dirty_plate_in_sink(bid, sink[0], sink[1])
            return
        
        # If sink has dirty plates, wash them
        for sink in self._sinks:
            tile = c.get_tile(team, sink[0], sink[1])
            if tile and hasattr(tile, 'num_dirty_plates') and tile.num_dirty_plates > 0:
                if self._move_to(c, bid, bs, sink):
                    c.wash_sink(bid, sink[0], sink[1])
                return
        
        # If plate counter doesn't have plate, prepare one (only on larger maps)
        if self.plate_counter and self.map_area >= 200:
            tile = c.get_tile(team, self.plate_counter[0], self.plate_counter[1])
            plate_item = getattr(tile, 'item', None) if tile else None
            plate_ready = isinstance(plate_item, Plate) and not plate_item.dirty
            
            if not plate_ready:
                # If holding clean plate, place it
                if holding_type == 'Plate' and not holding.get('dirty', True):
                    if self._move_to(c, bid, bs, self.plate_counter):
                        c.place(bid, self.plate_counter[0], self.plate_counter[1])
                    return
                
                # If holding something else, trash it
                if holding:
                    trash = self._nearest(bs['x'], bs['y'], 'TRASH')
                    if trash and self._move_to(c, bid, bs, trash):
                        c.trash(bid, trash[0], trash[1])
                    return
                
                # Get plate from sinktable
                for st in self._sinktables:
                    tile = c.get_tile(team, st[0], st[1])
                    if tile and hasattr(tile, 'num_clean_plates') and tile.num_clean_plates > 0:
                        if self._move_to(c, bid, bs, st):
                            c.take_clean_plate(bid, st[0], st[1])
                        return
                
                # Buy plate if enough money (only if really well-funded)
                money = c.get_team_money(team)
                shop = self._nearest(bs['x'], bs['y'], 'SHOP')
                if shop and money >= ShopCosts.PLATE.buy_cost + 150:
                    if self._move_to(c, bid, bs, shop):
                        c.buy(bid, ShopCosts.PLATE, shop[0], shop[1])
                    return
        
        # Stay near sink
        sink = self._nearest(bs['x'], bs['y'], 'SINK')
        if sink:
            self._move_to(c, bid, bs, sink)
    
    def _score(self, order, x, y, turn):
        req = order['required']
        
        # Calculate cost
        cost = ShopCosts.PLATE.buy_cost
        for fn in req:
            info = FOOD_INFO.get(fn, {})
            cost += info.get('cost', 0)
        
        profit = order['reward'] - cost
        if profit <= 0:
            if turn <= 15:
                import sys
                print(f'  REJECT: profit={profit} <= 0', file=sys.stderr, flush=True)
            return -9999
        
        tleft = order['expires_turn'] - turn
        if tleft < 12:
            if turn <= 15:
                import sys
                print(f'  REJECT: tleft={tleft} < 12', file=sys.stderr, flush=True)
            return -9999
        
        # Count processing needs
        n_cook = sum(1 for fn in req if FOOD_INFO.get(fn, {}).get('cook', False))
        n_chop = sum(1 for fn in req if FOOD_INFO.get(fn, {}).get('chop', False))
        
        # Estimate time with travel distances
        base = 12
        per_item = 3
        cook_time = 22
        chop_time = 4
        
        est = base + len(req) * per_item + n_cook * cook_time + n_chop * chop_time
        
        # Add travel time estimates
        travel = 0
        if self.dist_shop_counter:
            travel += self.dist_shop_counter * len(req)
        if self.dist_shop_cooker and n_cook > 0:
            travel += self.dist_shop_cooker * n_cook
        if self.dist_counter_submit:
            travel += self.dist_counter_submit
        
        # Adjust travel for map size
        if self.map_area >= 250:
            est += travel * 0.8
        else:
            est += travel * 0.4
        
        # Complexity penalty for limited counters
        if len(self._counters) <= 1 and n_chop >= 2:
            if turn <= 15:
                import sys
                print(f'  REJECT: counters={len(self._counters)}, n_chop={n_chop}', file=sys.stderr, flush=True)
            return -9999
        if self.map_area >= 300 and len(self._counters) <= 8:
            if len(req) >= 4:
                if turn <= 15:
                    import sys
                    print(f'  REJECT: large map, 4+ items', file=sys.stderr, flush=True)
                return -9999
            if len(req) == 3 and (n_chop + n_cook) >= 2:
                if turn <= 15:
                    import sys
                    print(f'  REJECT: large map, complex 3-item', file=sys.stderr, flush=True)
                return -9999
        
        # Feasibility check
        if est > tleft * 0.75:
            if turn <= 15:
                import sys
                print(f'  REJECT: est={est} > tleft*0.75={tleft*0.75}', file=sys.stderr, flush=True)
            return -9999
        
        # Prioritize urgent orders
        score = profit / max(est, 1)
        if tleft < 80:
            score += 0.5
        
        return score
    
    def _make_recipe(self, c, bid, order, bs):
        bx, by = bs['x'], bs['y']
        team = c.get_team()
        turn = c.get_turn()
        
        shop = self._nearest(bx, by, 'SHOP')
        submit = self._nearest(bx, by, 'SUBMIT')
        if not shop or not submit:
            if turn <= 30:
                print(f'[T{turn}] Recipe fail: shop={shop}, submit={submit}')
            return None
        
        # Parse ingredients by processing type
        cook_chop = []
        cook_only = []
        chop_only = []
        simple = []
        
        for fn in order['required']:
            info = FOOD_INFO.get(fn, {})
            ftype = info.get('type')
            if not ftype:
                continue
            if info.get('cook') and info.get('chop'):
                cook_chop.append((fn, ftype))
            elif info.get('cook'):
                cook_only.append((fn, ftype))
            elif info.get('chop'):
                chop_only.append((fn, ftype))
            else:
                simple.append((fn, ftype))
        
        all_cook = cook_chop + cook_only
        all_chop = cook_chop + chop_only
        
        # Find resources
        plate_pos = self.plate_counter or self._nearest(bx, by, 'COUNTER')
        counter = self.work_counter or self._find_empty_counter(c, bx, by, exclude=[plate_pos])
        cooker = self._find_available_cooker(c, bx, by) if all_cook else None
        
        if not plate_pos:
            if turn <= 30:
                print(f'[T{turn}] Recipe fail: no plate_pos')
            return None
        if all_chop and not counter:
            counter = plate_pos  # Use plate counter for chopping if needed
        if all_cook and not cooker:
            if turn <= 30:
                print(f'[T{turn}] Recipe fail: need cooker but cooker={cooker}, all_cookers={self._cookers}')
            return None
        
        steps = []
        
        # Check if plate is already on plate_counter
        tile = c.get_tile(team, plate_pos[0], plate_pos[1])
        plate_item = getattr(tile, 'item', None) if tile else None
        plate_ready = isinstance(plate_item, Plate) and not plate_item.dirty
        
        if not plate_ready:
            # Buy plate and place at plate_counter
            steps.append(('buy', ShopCosts.PLATE, shop))
            steps.append(('place', plate_pos))
        
        # Ensure pan is on cooker for cook orders
        if all_cook and cooker:
            tile = c.get_tile(team, cooker[0], cooker[1])
            pan_item = getattr(tile, 'item', None) if tile else None
            if not isinstance(pan_item, Pan):
                steps.append(('buy', ShopCosts.PAN, shop))
                steps.append(('place', cooker))
        
        # Simple items first (fastest)
        for fn, ftype in simple:
            steps.append(('buy', ftype, shop))
            steps.append(('add', plate_pos))
        
        # Chop-only items
        for fn, ftype in chop_only:
            steps.append(('buy', ftype, shop))
            steps.append(('place', counter))
            steps.append(('chop', counter))
            steps.append(('pickup', counter))
            steps.append(('add', plate_pos))
        
        # Cook items (chop+cook and cook-only)
        for fn, ftype in all_cook:
            info = FOOD_INFO.get(fn, {})
            steps.append(('buy', ftype, shop))
            if info.get('chop'):
                steps.append(('place', counter))
                steps.append(('chop', counter))
                steps.append(('pickup', counter))
            steps.append(('cook', cooker))
            steps.append(('wait_cook', cooker))
            steps.append(('add', plate_pos))
        
        # Submit
        steps.append(('pickup', plate_pos))
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
        
        # Debug: show progress periodically
        if turn <= 30 or turn % 50 == 0:
            import sys
            print(f'[T{turn}] Bot{bid}: step={t["step"]}/{len(t["recipe"])}, action={action}, stuck={t.get("stuck", 0)}', file=sys.stderr, flush=True)
        
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
                # Try start_cook first, then place
                done = c.start_cook(bid, cooker[0], cooker[1]) or c.place(bid, cooker[0], cooker[1])
        
        elif action == 'wait_cook':
            cooker = step[1]
            if self._move_to(c, bid, bs, cooker):
                tile = c.get_tile(c.get_team(), cooker[0], cooker[1])
                if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan):
                    pan = tile.item
                    if pan.food and pan.food.cooked_stage >= 1:
                        done = c.take_from_pan(bid, cooker[0], cooker[1])
                    elif pan.food and pan.food.cooked_stage == 2:
                        # Burned! Take and trash
                        if c.take_from_pan(bid, cooker[0], cooker[1]):
                            # Insert trash step
                            trash = self._nearest(bs['x'], bs['y'], 'TRASH')
                            if trash:
                                t['recipe'].insert(t['step'] + 1, ('trash', trash))
                                # Re-add buy and cook steps
                                # This is a simplified recovery - just mark as stuck
                                t['stuck'] = 100
        
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
