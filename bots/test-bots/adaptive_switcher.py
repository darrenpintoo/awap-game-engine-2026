"""
Adaptive Switcher Bot - Strategic Score-Based Sabotage
========================================================

STRATEGY:
- Monitors score difference between teams
- Switches ONLY when significantly behind (defensive sabotage)
- Focuses on BLOCKING rather than stealing (time-efficient)
- Returns to own map once ahead or neutral

KEY DECISION LOGIC:
1. Track score every 10 turns
2. If behind by >200 points, switch to sabotage
3. Sabotage = block enemy bots, don't steal (stealing wastes time)
4. If score recovers, return to own production
5. Never switch if winning by >100

WEAKNESSES THIS EXPLOITS:
- Bots that pull ahead early then coast
- Bots without defensive capabilities
- Maps where blocking is effective

PERFORMANCE PROFILE:
- EXCELS ON: Competitive matches, maps with chokepoints
- STRUGGLES ON: One-sided matches, maps where blocking is ineffective
"""

import random
from collections import deque
from typing import Optional, List, Tuple

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass


INGREDIENT_INFO = {
    'SAUCE':   {'cost': 10, 'chop': False, 'cook': False},
    'EGG':     {'cost': 20, 'chop': False, 'cook': True},
    'ONIONS':  {'cost': 30, 'chop': True,  'cook': False},
    'NOODLES': {'cost': 40, 'chop': False, 'cook': False},
    'MEAT':    {'cost': 80, 'chop': True,  'cook': True},
}


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map features
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sink_tables = []
        
        # Score tracking
        self.last_check_turn = 0
        self.score_threshold = 200  # Switch if behind by this much
        self.recovery_threshold = 50  # Return if within this margin
        
        # State
        self.is_sabotaging = False
        self.has_switched = False
        self.primary_state = 0
        self.plate_counter = None
        self.current_order = None
        self.current_ingredient = None
        self.cooking_cooker = None
        
    def _init_map(self, controller, team):
        m = controller.get_map(team)
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                pos = (x, y)
                name = tile.tile_name
                if name == "SHOP": self.shops.append(pos)
                elif name == "COOKER": self.cookers.append(pos)
                elif name == "COUNTER": self.counters.append(pos)
                elif name == "SUBMIT": self.submits.append(pos)
                elif name == "TRASH": self.trashes.append(pos)
                elif name == "SINKTABLE": self.sink_tables.append(pos)
        self.initialized = True
    
    def bfs_move(self, controller, bot_id, target, team):
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        
        if max(abs(bx - target[0]), abs(by - target[1])) <= 1:
            return True
        
        m = controller.get_map(team)
        queue = deque([((bx, by), None)])
        visited = {(bx, by)}
        
        while queue:
            (x, y), first_move = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited or not m.is_tile_walkable(nx, ny):
                        continue
                    
                    visited.add((nx, ny))
                    fm = first_move or (dx, dy)
                    
                    if max(abs(nx - target[0]), abs(ny - target[1])) <= 1:
                        if controller.can_move(bot_id, fm[0], fm[1]):
                            controller.move(bot_id, fm[0], fm[1])
                        return False
                    queue.append(((nx, ny), fm))
        return False
    
    def get_nearest(self, pos, locations):
        if not locations:
            return None
        return min(locations, key=lambda p: max(abs(p[0]-pos[0]), abs(p[1]-pos[1])))
    
    def get_free_counter(self, controller, team, pos, exclude=None):
        free = []
        for c in self.counters:
            if exclude and c in exclude:
                continue
            tile = controller.get_tile(team, c[0], c[1])
            if tile and getattr(tile, 'item', None) is None:
                free.append(c)
        return self.get_nearest(pos, free)
    
    def should_sabotage(self, controller, team):
        """Check if we should switch to sabotage mode"""
        turn = controller.get_turn()
        
        # Don't check too frequently
        if turn - self.last_check_turn < 10:
            return self.is_sabotaging
        
        self.last_check_turn = turn
        
        our_money = controller.get_team_money(team)
        enemy_money = controller.get_team_money(controller.get_enemy_team())
        
        score_diff = our_money - enemy_money
        
        # Currently sabotaging - should we return?
        if self.is_sabotaging:
            if score_diff >= -self.recovery_threshold:
                return False  # Score recovered, go back to production
            return True
        
        # Not sabotaging - should we start?
        if turn >= 150 and turn <= 400:  # Mid-game window
            if score_diff < -self.score_threshold:
                return True  # Behind significantly
        
        return False
    
    def select_simple_order(self, controller, team):
        """Select a simple order (prefer no cooking)"""
        orders = controller.get_orders(team)
        current_turn = controller.get_turn()
        
        best = None
        best_score = -9999
        
        for order in orders:
            if not order.get('is_active'):
                continue
            
            time_left = order['expires_turn'] - current_turn
            if time_left < 25:
                continue
            
            required = order['required']
            has_cooking = any(INGREDIENT_INFO.get(i, {}).get('cook') for i in required)
            
            score = 100 - len(required) * 20
            if not has_cooking:
                score += 30
            if len(required) <= 2:
                score += 20
            
            if score > best_score:
                best_score = score
                best = order
        
        return best
    
    def execute_blocking(self, controller, bot_id, team):
        """Execute BLOCKING sabotage (not stealing - more time efficient)"""
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        enemy_map = controller.get_map(enemy_team)
        
        # If holding something, trash it quickly
        if holding:
            enemy_trashes = []
            for x in range(enemy_map.width):
                for y in range(enemy_map.height):
                    if enemy_map.tiles[x][y].tile_name == "TRASH":
                        enemy_trashes.append((x, y))
            trash = self.get_nearest((bx, by), enemy_trashes)
            if trash and self.bfs_move(controller, bot_id, trash, enemy_team):
                controller.trash(bot_id, trash[0], trash[1])
            return
        
        # Find enemy bot positions
        enemy_bots = controller.get_team_bot_ids(enemy_team)
        enemy_positions = []
        for eb_id in enemy_bots:
            eb = controller.get_bot_state(eb_id)
            if eb:
                enemy_positions.append((eb['x'], eb['y']))
        
        # Find enemy key locations (cookers, counters with items)
        enemy_cookers = []
        enemy_counters_with_items = []
        
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                if tile.tile_name == "COOKER":
                    enemy_cookers.append((x, y))
                elif tile.tile_name == "COUNTER":
                    if getattr(controller.get_tile(enemy_team, x, y), 'item', None):
                        enemy_counters_with_items.append((x, y))
        
        # Strategy: Block access to cookers
        if enemy_cookers:
            # Find cooker nearest to enemy bots (most likely target)
            target_cooker = None
            if enemy_positions:
                target_cooker = min(enemy_cookers, 
                                   key=lambda c: min(max(abs(c[0]-ep[0]), abs(c[1]-ep[1])) 
                                                    for ep in enemy_positions))
            else:
                target_cooker = enemy_cookers[0]
            
            # Move to block path to cooker
            if self.bfs_move(controller, bot_id, target_cooker, enemy_team):
                # We're adjacent - just stay here blocking
                pass
            return
        
        # Alternative: Block counters with items
        if enemy_counters_with_items:
            target = self.get_nearest((bx, by), enemy_counters_with_items)
            if target:
                self.bfs_move(controller, bot_id, target, enemy_team)
            return
        
        # Fallback: Random movement near enemy bots
        if enemy_positions:
            target = random.choice(enemy_positions)
            self.bfs_move(controller, bot_id, target, enemy_team)
    
    def execute_production(self, controller, bot_id, team):
        """Execute normal production pipeline"""
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        shop = self.get_nearest((bx, by), self.shops)
        submit = self.get_nearest((bx, by), self.submits)
        trash = self.get_nearest((bx, by), self.trashes)
        cooker = self.cookers[0] if self.cookers else None
        
        if not self.current_order:
            self.current_order = self.select_simple_order(controller, team)
            self.primary_state = 0
            self.plate_counter = None
            self.current_ingredient = None
            
        if not self.current_order:
            return
        
        order = self.current_order
        
        # Simplified pipeline (similar to other bots)
        # State 0: Get plate
        if self.primary_state == 0:
            if holding and holding.get('type') == 'Plate':
                self.primary_state = 1
            else:
                if self.sink_tables:
                    st = self.get_nearest((bx, by), self.sink_tables)
                    tile = controller.get_tile(team, st[0], st[1])
                    if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                        if self.bfs_move(controller, bot_id, st, team):
                            controller.take_clean_plate(bot_id, st[0], st[1])
                        return
                if shop and self.bfs_move(controller, bot_id, shop, team):
                    if money >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
        
        # State 1: Place plate
        elif self.primary_state == 1:
            counter = self.get_free_counter(controller, team, (bx, by))
            if counter and self.bfs_move(controller, bot_id, counter, team):
                controller.place(bot_id, counter[0], counter[1])
                self.plate_counter = counter
                self.primary_state = 2
        
        # State 2: Process ingredients
        elif self.primary_state == 2:
            plate_contents = []
            if self.plate_counter:
                tile = controller.get_tile(team, self.plate_counter[0], self.plate_counter[1])
                if tile and isinstance(getattr(tile, 'item', None), Plate):
                    plate_contents = [f.food_name for f in tile.item.food]
            
            missing = [i for i in order['required'] if i not in plate_contents]
            if not missing:
                self.primary_state = 5
                return
            
            next_ing = missing[0]
            
            if holding and holding.get('type') == 'Food':
                # Add to plate directly (skip cooking for speed)
                if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                    controller.add_food_to_plate(bot_id, self.plate_counter[0], self.plate_counter[1])
            elif shop and self.bfs_move(controller, bot_id, shop, team):
                food_type = getattr(FoodType, next_ing, None)
                if food_type and money >= food_type.buy_cost:
                    controller.buy(bot_id, food_type, shop[0], shop[1])
        
        # State 5: Pickup plate
        elif self.primary_state == 5:
            if self.plate_counter and self.bfs_move(controller, bot_id, self.plate_counter, team):
                controller.pickup(bot_id, self.plate_counter[0], self.plate_counter[1])
                self.primary_state = 6
        
        # State 6: Submit
        elif self.primary_state == 6:
            if submit and self.bfs_move(controller, bot_id, submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.current_order = None
                    self.primary_state = 0
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        turn = controller.get_turn()
        
        if not self.initialized:
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Adaptive decision: should we sabotage?
        want_sabotage = self.should_sabotage(controller, team)
        
        if want_sabotage and not self.is_sabotaging:
            # Try to switch
            if controller.can_switch_maps():
                if controller.switch_maps():
                    self.is_sabotaging = True
                    self.has_switched = True
        elif not want_sabotage and self.is_sabotaging:
            # Return to production
            self.is_sabotaging = False
        
        # Execute based on mode
        if self.is_sabotaging:
            # Both bots do blocking
            for bot_id in bots:
                self.execute_blocking(controller, bot_id, team)
        else:
            # Normal production
            self.execute_production(controller, bots[0], team)
            
            # Helper bot: wash dishes
            if len(bots) > 1:
                helper = bots[1]
                hbot = controller.get_bot_state(helper)
                hx, hy = hbot['x'], hbot['y']
                
                for x in range(self.map.width):
                    for y in range(self.map.height):
                        tile = controller.get_tile(team, x, y)
                        if tile and tile.tile_name == "SINK":
                            if getattr(tile, 'num_dirty_plates', 0) > 0:
                                if self.bfs_move(controller, helper, (x, y), team):
                                    controller.wash_sink(helper, x, y)
                                break
