"""
Super Rush Bot - AWAP 2026 Tournament Entry
===========================================

Strategy:
- SPEED IS KEY. 
- Only selects orders that can be completed VERY fast (e.g. Salads, Sushi).
- Ignores high-value complex orders.
- Bot 1 and Bot 2 work INDEPENDENTLY on different simple orders if possible, 
  or Bot 2 acts as a dedicated "fetcher" for Bot 1 to minimize travel time.
"""

import numpy as np
import heapq
from collections import deque
from typing import Tuple, Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass

DEBUG = False
def log(msg):
    if DEBUG: print(f"[RUSH] {msg}")

INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True},
}

class FastPathfinder:
    DIRS_8 = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    def __init__(self, map_obj):
        self.width = map_obj.width
        self.height = map_obj.height
        self.walkable = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                self.walkable[x, y] = getattr(map_obj.tiles[x][y], 'is_walkable', False)
        
        self.tile_cache = {}
        for x in range(self.width):
            for y in range(self.height):
                name = map_obj.tiles[x][y].tile_name
                if name not in self.tile_cache: self.tile_cache[name] = []
                self.tile_cache[name].append((x,y))
                
        self.dist_matrices = {}
        for tile in ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE']:
            if tile in self.tile_cache:
                for pos in self.tile_cache[tile]:
                    self.dist_matrices[pos] = self._bfs(pos)

    def _bfs(self, target):
        dist = np.full((self.width, self.height), 9999.0)
        q = deque()
        tx, ty = target
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx, ny = tx+dx, ty+dy
                if 0<=nx<self.width and 0<=ny<self.height and self.walkable[nx,ny]:
                    dist[nx,ny] = 0
                    q.append((nx,ny))
        
        while q:
            x,y = q.popleft()
            for dx,dy in self.DIRS_8:
                nx,ny = x+dx, y+dy
                if 0<=nx<self.width and 0<=ny<self.height and self.walkable[nx,ny]:
                    if dist[nx,ny] > dist[x,y]+1:
                        dist[nx,ny] = dist[x,y]+1
                        q.append((nx,ny))
        return dist

    @staticmethod
    def chebyshev(p1, p2):
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))

    def get_next_step(self, controller, bot_id, target):
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        if self.chebyshev((bx,by), target) <= 1: return None
        
        # Simple avoidance checks
        team = controller.get_team()
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
            if bid != bot_id:
                st = controller.get_bot_state(bid)
                if st: avoid.add((st['x'], st['y']))

        dm = self.dist_matrices.get(target)
        best_dist = 9999
        best_step = None
        
        for dx, dy in self.DIRS_8:
            if not controller.can_move(bot_id, dx, dy): continue
            nx, ny = bx+dx, by+dy
            if (nx,ny) in avoid: continue
            
            d = dm[nx,ny] if dm is not None else self.chebyshev((nx,ny), target)
            if d < best_dist:
                best_dist = d
                best_step = (dx,dy)
        return best_step

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.init = False
        self.pf = None
        self.orders = {} # bot_id -> order
        self.plate_contents = {} # bot_id -> set(ingredients)
        self.cooking = {} # bot_id -> (ingredient, turn_started)
        
        # Cache
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.sinks = []
        self.sink_tables = []
        
        self.my_counter = {} # bot_id -> (x,y)
    
    def _init(self, controller, team):
        m = controller.get_map(team)
        self.pf = FastPathfinder(m)
        for x in range(m.width):
            for y in range(m.height):
                n = m.tiles[x][y].tile_name
                p = (x,y)
                if n == "SHOP": self.shops.append(p)
                elif n == "COOKER": self.cookers.append(p)
                elif n == "COUNTER": self.counters.append(p)
                elif n == "SUBMIT": self.submits.append(p)
                elif n == "SINK": self.sinks.append(p)
                elif n == "SINKTABLE": self.sink_tables.append(p)
        
        # Assign counters to bots
        bots = controller.get_team_bot_ids(team)
        for i, bid in enumerate(bots):
            if i < len(self.counters):
                self.my_counter[bid] = self.counters[i]
            else:
                self.my_counter[bid] = self.counters[0]
        self.init = True

    def _move(self, controller, bot_id, target):
        step = self.pf.get_next_step(controller, bot_id, target)
        if step:
            controller.move(bot_id, step[0], step[1])
            return False # Moving
        return True # Arrived (or stuck)

    def _get_rush_order(self, controller, team):
        # Pick FASTEST order
        orders = controller.get_orders(team)
        best = None
        min_time = 9999
        
        for o in orders:
            if not o['is_active']: continue
            time = 0
            for ing in o['required']:
                info = INGREDIENT_INFO.get(ing, {'cook':False, 'chop':False})
                if info['cook']: time += 30
                elif info['chop']: time += 10
                else: time += 5
            
            # Penalize complex orders heavily
            if time < min_time:
                min_time = time
                best = o
        
        return best

    def play_turn(self, controller):
        if not self.init: self._init(controller, controller.get_team())
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        
        for bid in bots:
            self._run_indep_bot(controller, bid, team)

    def _run_indep_bot(self, controller, bot_id, team):
        # Independent bot logic - strict state machine
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        if bot_id not in self.orders or not self.orders[bot_id]:
            self.orders[bot_id] = self._get_rush_order(controller, team)
            self.plate_contents[bot_id] = set()
            self.cooking[bot_id] = None
            
        order = self.orders[bot_id]
        if not order: return # No orders?
        
        holding = bot.get('holding')
        bx, by = bot['x'], bot['y']
        
        # State Machine Priority
        
        # 0. SUBMIT if done
        if len(self.plate_contents[bot_id]) == len(order['required']):
            # Need to pick up plate?
            if holding and holding['type'] == 'Plate':
                if self._move(controller, bot_id, self.submits[0]): # Just go to first submit
                     controller.submit(bot_id, self.submits[0][0], self.submits[0][1])
                     self.orders[bot_id] = None # Reset
                return
            else:
                # Pick up plate from counter
                ctr = self.my_counter[bot_id]
                if self._move(controller, bot_id, ctr):
                     controller.pickup(bot_id, ctr[0], ctr[1])
                return

        # 1. Start Cooking?
        needed_cook = [i for i in order['required'] if INGREDIENT_INFO[i]['cook'] and i not in self.plate_contents[bot_id]]
        if needed_cook and not self.cooking[bot_id]:
            # Do we have a pan?
            cooker = self.cookers[0]
            tile = controller.get_tile(team, cooker[0], cooker[1])
            has_pan = tile and hasattr(tile, 'item') and isinstance(tile.item, Pan)
            
            if not has_pan:
                # Need pan
                if holding and holding['type'] == 'Pan':
                    if self._move(controller, bot_id, cooker):
                        controller.place(bot_id, cooker[0], cooker[1])
                else:
                    shop = self.shops[0]
                    if self._move(controller, bot_id, shop):
                         controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                return
            
            # Start cooking ingredient
            ing = needed_cook[0]
            if holding and holding.get('food_name') == ing:
                 if self._move(controller, bot_id, cooker):
                     controller.place(bot_id, cooker[0], cooker[1])
                     self.cooking[bot_id] = ing
            else:
                shop = self.shops[0]
                if self._move(controller, bot_id, shop):
                    ft = getattr(FoodType, ing)
                    controller.buy(bot_id, ft, shop[0], shop[1])
            return

        # 2. Collect Cooked Food?
        if self.cooking[bot_id]:
             # Check if done
             cooker = self.cookers[0]
             tile = controller.get_tile(team, cooker[0], cooker[1])
             if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan) and tile.item.food:
                 if tile.item.food.cooked_stage == 1: # Done
                     # Need plate on counter first?
                     ctr = self.my_counter[bot_id]
                     c_tile = controller.get_tile(team, ctr[0], ctr[1])
                     has_plate = c_tile and hasattr(c_tile, 'item') and isinstance(c_tile.item, Plate)
                     
                     if not has_plate:
                         # Prioritize getting plate
                         if holding and holding['type'] == 'Plate':
                             if self._move(controller, bot_id, ctr):
                                 controller.place(bot_id, ctr[0], ctr[1])
                         else:
                             shop = self.shops[0]
                             if self._move(controller, bot_id, shop):
                                 controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
                         return

                     if self._move(controller, bot_id, cooker):
                         controller.take_from_pan(bot_id, cooker[0], cooker[1])
                         self.cooking[bot_id] = None # Reset cooking state, now holding cooked food
                         # Will fall through to "Add to Plate"
                 else:
                     return # Wait for cook
             else:
                 self.cooking[bot_id] = None # Pan empty? Something wrong

        # 3. Add to Plate (Holding food)
        if holding and holding['type'] == 'Food':
            ctr = self.my_counter[bot_id]
            # Ensure plate is there
            c_tile = controller.get_tile(team, ctr[0], ctr[1])
            has_plate = c_tile and hasattr(c_tile, 'item') and isinstance(c_tile.item, Plate)
            
            if not has_plate:
                # Drop food on another counter temporarily? Or just hold?
                # Rush bot: just wait implies bad logic, but for simplicity assuming we bought plate first
                pass 
                
            if self._move(controller, bot_id, ctr):
                if controller.add_food_to_plate(bot_id, ctr[0], ctr[1]):
                    self.plate_contents[bot_id].add(holding['food_name'])
            return

        # 4. Get Plate (if needed and not cooking)
        # Check if plate on counter
        ctr = self.my_counter[bot_id]
        c_tile = controller.get_tile(team, ctr[0], ctr[1])
        has_plate = c_tile and hasattr(c_tile, 'item') and isinstance(c_tile.item, Plate)
        
        if not has_plate and not holding:
             # Buy plate
             shop = self.shops[0]
             if self._move(controller, bot_id, shop):
                 controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
             return
        
        if holding and holding['type'] == 'Plate':
             if self._move(controller, bot_id, ctr):
                 controller.place(bot_id, ctr[0], ctr[1])
             return

        # 5. Get Next Raw Ingredient
        needed = [i for i in order['required'] if i not in self.plate_contents[bot_id] and not INGREDIENT_INFO[i]['cook']]
        if needed:
            ing = needed[0]
            if holding:
                if holding.get('food_name') == ing:
                    # Go to plate
                    if self._move(controller, bot_id, ctr):
                        controller.add_food_to_plate(bot_id, ctr[0], ctr[1])
                        self.plate_contents[bot_id].add(ing)
                else:
                    # Holding wrong thing? Trash it or place it
                    # For RUSH bot, just temporary place on counter if possible, or trash
                    if self._move(controller, bot_id, ctr):
                        if not controller.place(bot_id, ctr[0], ctr[1]):
                            # If place failed (counter full), go to trash
                            tr = self.shops[0] # Temporary fallback if no trash nearby, but likely just stuck
                            # Actually let's just find trash
                            pass 
            else:
                # Buy
                shop = self.shops[0]
                if self._move(controller, bot_id, shop):
                    ft = getattr(FoodType, ing)
                    controller.buy(bot_id, ft, shop[0], shop[1])
            return

        # If nothing matches, idle
