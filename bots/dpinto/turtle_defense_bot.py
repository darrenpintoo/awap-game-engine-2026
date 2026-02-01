"""
Turtle Defense Bot - AWAP 2026 Tournament Entry
===============================================

Strategy:
- DEFENSIVE & SAFE.
- Hoards resources (Pans/Plates) early to deny them to enemy (if shared) or ensure safety.
- NEVER burns food (Highest priority rescue).
- If winning, switches to "Stall Mode" where it just holds resources and blocks key tiles.
- Prioritizes easy, risk-free orders.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List
from enum import Enum, auto

try:
    from game_constants import Team, TileType, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate
except ImportError:
    pass

DEBUG = False
def log(msg):
    if DEBUG: print(f"[TURTLE] {msg}")

class BotState(Enum):
    IDLE = auto()
    HOARD_PAN = auto()      # Buy and hold/stash a pan
    HOARD_PLATE = auto()    # Buy and hold/stash a plate
    SAFE_COOK = auto()      # Cook with extreme caution
    DELIVER = auto()
    STALL = auto()          # Just stand there and block

class TurtleBot:
    def __init__(self, map_copy):
        self.map = map_copy
        self.init = False
        self.pf = None
        
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.boxes = []
        
        self.my_stash = {} # bot_id -> (x,y) for stashing items
        self.tasks = {} # bot_id -> task
        
        self.hoard_count = 0
        self.target_hoard = 2 # Try to hoard 2 extra items
        
    def _init(self, controller, team):
        m = controller.get_map(team)
        self.width = m.width
        self.height = m.height
        
        # Simple BFS Pathfinder
        self.dist_cache = {}
        
        for x in range(m.width):
            for y in range(m.height):
                n = m.tiles[x][y].tile_name
                p = (x,y)
                if n == "SHOP": self.shops.append(p)
                elif n == "COOKER": self.cookers.append(p)
                elif n == "COUNTER": self.counters.append(p)
                elif n == "SUBMIT": self.submits.append(p)
                elif n == "BOX": self.boxes.append(p)
        
        # Assign stash spots (Boxes first, then far counters)
        bots = controller.get_team_bot_ids(team)
        stashes = self.boxes + self.counters[::-1] # Use counters from end
        for i, bid in enumerate(bots):
            if i < len(stashes):
                self.my_stash[bid] = stashes[i]
        
        self.init = True

    def _dist(self, p1, p2):
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))

    def _move(self, controller, bot_id, target):
        # Very basic movement
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        
        if self._dist((bx,by), target) <= 1: return None
        
        # Random walk towards target
        best_d = 9999
        best_m = None
        
        # Simple avoidance checks
        team = controller.get_team()
        avoid = set()
        for bid in controller.get_team_bot_ids(team):
             if bid != bot_id:
                 st = controller.get_bot_state(bid)
                 if st: avoid.add((st['x'],st['y']))

        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = bx+dx, by+dy
                if not controller.can_move(bot_id, dx, dy): continue
                if (nx,ny) in avoid: continue
                
                d = self._dist((nx,ny), target)
                if d < best_d:
                    best_d = d
                    best_m = (dx,dy)
        
        if best_m:
            controller.move(bot_id, best_m[0], best_m[1])
            return True
        return False

    def play_turn(self, controller):
        if not self.init: self._init(controller, controller.get_team())
        team = controller.get_team()
        bots = controller.get_team_bot_ids(team)
        
        # Check score
        my_score = controller.get_team_money(team)
        enemy_score = controller.get_team_money(controller.get_enemy_team())
        is_winning = my_score > enemy_score
        
        for bid in bots:
            self._run_turtle(controller, bid, team, is_winning)

    def _run_turtle(self, controller, bot_id, team, is_winning):
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        # 1. SAFETY UPDATE: Check for burning food!
        for cx, cy in self.cookers:
            tile = controller.get_tile(team, cx, cy)
            if tile and hasattr(tile, 'item') and isinstance(tile.item, Pan) and tile.item.food:
                # If cooked and about to burn (arbitrary threshold, say cooked_progress > 90)
                # API doesn't give exact progress, just stage. 
                # Stage 1 = Cooked. Grab it immediately to be safe!
                if tile.item.food.cooked_stage == 1:
                    # Emergency rescue!
                    if self._move(controller, bot_id, (cx,cy)) is None: # Adjacent
                        if bot['holding']:
                            # Panic drop info
                             stash = self.my_stash.get(bot_id, self.counters[0])
                             if self._move(controller, bot_id, stash) is None:
                                 controller.place(bot_id, stash[0], stash[1])
                        else:
                            controller.take_from_pan(bot_id, cx, cy)
                    return

        # 2. Winning? STALL
        if is_winning and controller.get_turn() > 300: # Only stall late game
             # If holding something valuable, hide
             if bot['holding']:
                 stash = self.my_stash.get(bot_id)
                 if stash and self._move(controller, bot_id, stash) is None:
                      pass # Just stand there guarding it
             else:
                 # Go block a submit station
                 if self.submits:
                     self._move(controller, bot_id, self.submits[0])
             return

        # 3. Hoarding Phase (Early game)
        if controller.get_turn() < 50 and self.hoard_count < self.target_hoard:
            # Buy pans/plates and hide them
            if not bot['holding']:
                 shop = self.shops[0]
                 if self._move(controller, bot_id, shop) is None:
                     controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1]) # Buy pans to deny
            else:
                stash = self.my_stash.get(bot_id, self.counters[-1])
                if self._move(controller, bot_id, stash) is None:
                    if controller.place(bot_id, stash[0], stash[1]):
                        self.hoard_count += 1
            return

        # 4. Normal Play (Slow & Safe)
        # Randomly do an order if nothing else
        # (Simplified for this file - just grab any ingredient and put on counter)
        orders = controller.get_orders(team)
        active = [o for o in orders if o['is_active']]
        if active:
            order = active[0]
            # Just try to fulfill it slowly
            pass 

# Main Entry Point
class BotPlayer(TurtleBot):
    pass
