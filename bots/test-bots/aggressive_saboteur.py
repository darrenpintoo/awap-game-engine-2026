"""
Aggressive Saboteur Bot - Early Switch Disruption Strategy
=============================================================

STRATEGY:
- Switches to enemy map EARLY (turn 60-100) to disrupt before they build momentum
- Prioritizes stealing/trashing enemy pans and plates
- One bot focuses on sabotage, other handles basic orders
- Sacrifices own production to cripple enemy economy

KEY DECISION LOGIC:
1. Turn 60-100: Switch to enemy map immediately
2. Sabotage priorities: Pan (cripples cooking) > Plates (blocks submissions) > Counter items
3. Primary bot handles simple orders only while waiting to switch
4. Never defends against enemy sabotage - pure offense

WEAKNESSES THIS EXPLOITS:
- Bots that invest heavily in early cooking (lose pan = lose time)
- Bots without recovery logic for stolen items
- Bots that pre-buy expensive ingredients (can be trashed)

PERFORMANCE PROFILE:
- EXCELS ON: Small maps (fast sabotage), maps with limited cookers
- STRUGGLES ON: Large maps (long travel time), maps with many cookers/plates
"""

import random
from collections import deque
from typing import Tuple, Optional, List, Set
from enum import Enum, auto

try:
    from game_constants import Team, FoodType, ShopCosts
    from robot_controller import RobotController
    from item import Pan, Plate, Food
except ImportError:
    pass


class SabotagePhase(Enum):
    WAITING = auto()
    STEAL_PAN = auto()
    STEAL_PLATE = auto()
    STEAL_COUNTER = auto()
    CHAOS = auto()
    TRASH_ITEM = auto()


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map locations (own map)
        self.shops = []
        self.cookers = []
        self.counters = []
        self.submits = []
        self.trashes = []
        self.sink_tables = []
        
        # State
        self.switch_turn = random.randint(60, 100)  # Randomize for unpredictability
        self.has_switched = False
        self.sabotage_phase = SabotagePhase.WAITING
        self.primary_bot_state = 0
        self.items_stolen = 0
        
    def _init_map(self, controller: RobotController, team):
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
    
    def bfs_move(self, controller, bot_id, start, target, team):
        """Simple BFS pathfinding - returns True if adjacent"""
        if max(abs(start[0] - target[0]), abs(start[1] - target[1])) <= 1:
            return True
        
        m = controller.get_map(team)
        queue = deque([(start, None)])
        visited = {start}
        
        while queue:
            (x, y), first_move = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in visited:
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    
                    visited.add((nx, ny))
                    fm = first_move if first_move else (dx, dy)
                    
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
    
    def execute_simple_order(self, controller, bot_id, team):
        """Primary bot does simple single-ingredient orders while waiting"""
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        money = controller.get_team_money(team)
        
        shop = self.get_nearest((bx, by), self.shops)
        submit = self.get_nearest((bx, by), self.submits)
        
        if not shop or not submit:
            return
        
        # State 0: Buy plate
        if self.primary_bot_state == 0:
            if holding and holding.get('type') == 'Plate':
                self.primary_bot_state = 1
            elif self.bfs_move(controller, bot_id, (bx, by), shop, team):
                if money >= ShopCosts.PLATE.buy_cost:
                    controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1])
        
        # State 1: Put plate on counter
        elif self.primary_bot_state == 1:
            counter = self.get_nearest((bx, by), self.counters)
            if counter:
                if self.bfs_move(controller, bot_id, (bx, by), counter, team):
                    controller.place(bot_id, counter[0], counter[1])
                    self.plate_counter = counter
                    self.primary_bot_state = 2
        
        # State 2: Buy simple ingredient (SAUCE or NOODLES - no processing)
        elif self.primary_bot_state == 2:
            if holding:
                self.primary_bot_state = 3
            elif self.bfs_move(controller, bot_id, (bx, by), shop, team):
                # Prioritize SAUCE (cheapest, no processing)
                if money >= FoodType.SAUCE.buy_cost:
                    controller.buy(bot_id, FoodType.SAUCE, shop[0], shop[1])
        
        # State 3: Add to plate
        elif self.primary_bot_state == 3:
            if hasattr(self, 'plate_counter') and self.plate_counter:
                if self.bfs_move(controller, bot_id, (bx, by), self.plate_counter, team):
                    controller.add_food_to_plate(bot_id, self.plate_counter[0], self.plate_counter[1])
                    self.primary_bot_state = 4
        
        # State 4: Pickup plate
        elif self.primary_bot_state == 4:
            if hasattr(self, 'plate_counter') and self.plate_counter:
                if self.bfs_move(controller, bot_id, (bx, by), self.plate_counter, team):
                    controller.pickup(bot_id, self.plate_counter[0], self.plate_counter[1])
                    self.primary_bot_state = 5
        
        # State 5: Submit
        elif self.primary_bot_state == 5:
            if self.bfs_move(controller, bot_id, (bx, by), submit, team):
                if controller.submit(bot_id, submit[0], submit[1]):
                    self.primary_bot_state = 0
    
    def execute_sabotage(self, controller, bot_id, team):
        """Aggressive sabotage on enemy map"""
        bot = controller.get_bot_state(bot_id)
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        enemy_team = controller.get_enemy_team()
        enemy_map = controller.get_map(enemy_team)
        
        # Find enemy locations
        enemy_cookers = []
        enemy_sink_tables = []
        enemy_trashes = []
        enemy_counters = []
        
        for x in range(enemy_map.width):
            for y in range(enemy_map.height):
                tile = enemy_map.tiles[x][y]
                name = tile.tile_name
                if name == "COOKER": enemy_cookers.append((x, y))
                elif name == "SINKTABLE": enemy_sink_tables.append((x, y))
                elif name == "TRASH": enemy_trashes.append((x, y))
                elif name == "COUNTER": enemy_counters.append((x, y))
        
        # If holding something, TRASH IT immediately
        if holding:
            trash = self.get_nearest((bx, by), enemy_trashes)
            if trash:
                if self.bfs_move(controller, bot_id, (bx, by), trash, enemy_team):
                    controller.trash(bot_id, trash[0], trash[1])
                    self.items_stolen += 1
            return
        
        # Priority 1: STEAL PAN (cripples cooking entirely)
        if self.sabotage_phase == SabotagePhase.STEAL_PAN:
            for cooker in enemy_cookers:
                tile = controller.get_tile(enemy_team, cooker[0], cooker[1])
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self.bfs_move(controller, bot_id, (bx, by), cooker, enemy_team):
                        if controller.pickup(bot_id, cooker[0], cooker[1]):
                            return
            self.sabotage_phase = SabotagePhase.STEAL_PLATE
        
        # Priority 2: STEAL CLEAN PLATES (blocks submissions)
        if self.sabotage_phase == SabotagePhase.STEAL_PLATE:
            for st in enemy_sink_tables:
                tile = controller.get_tile(enemy_team, st[0], st[1])
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    if self.bfs_move(controller, bot_id, (bx, by), st, enemy_team):
                        if controller.take_clean_plate(bot_id, st[0], st[1]):
                            return
            self.sabotage_phase = SabotagePhase.STEAL_COUNTER
        
        # Priority 3: STEAL from counters (ingredients, plates)
        if self.sabotage_phase == SabotagePhase.STEAL_COUNTER:
            for counter in enemy_counters:
                tile = controller.get_tile(enemy_team, counter[0], counter[1])
                if tile and getattr(tile, 'item', None):
                    if self.bfs_move(controller, bot_id, (bx, by), counter, enemy_team):
                        if controller.pickup(bot_id, counter[0], counter[1]):
                            return
            self.sabotage_phase = SabotagePhase.CHAOS
        
        # Priority 4: CHAOS - move randomly to block enemy bots
        if self.sabotage_phase == SabotagePhase.CHAOS:
            # Reset and cycle through again
            self.sabotage_phase = SabotagePhase.STEAL_PAN
            # Random movement
            dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    break
    
    def play_turn(self, controller: RobotController):
        team = controller.get_team()
        turn = controller.get_turn()
        
        if not self.initialized:
            self._init_map(controller, team)
        
        bots = controller.get_team_bot_ids(team)
        if not bots:
            return
        
        # Check if it's time to switch (EARLY - turn 60-100)
        if not self.has_switched and turn >= self.switch_turn:
            if controller.can_switch_maps():
                if controller.switch_maps():
                    self.has_switched = True
                    self.sabotage_phase = SabotagePhase.STEAL_PAN
        
        # Bot 0: Simple orders (keep some income)
        self.execute_simple_order(controller, bots[0], team)
        
        # Bot 1: Sabotage if switched, otherwise assist
        if len(bots) > 1:
            if self.has_switched:
                self.execute_sabotage(controller, bots[1], team)
            else:
                # Help with orders before switch
                pass  # Just idle
