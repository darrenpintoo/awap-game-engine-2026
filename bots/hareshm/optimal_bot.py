"""
Optimal AWAP 2026 Bot - Streamlined Version
============================================

Simplified single-bot approach inspired by duo_noodle_bot.
Each bot independently tries to complete orders.

Key Changes:
- Linear state machine per bot (not Chef/Runner)
- 4-directional A* pathfinding
- Manhattan distance adjacency
- Proper turn management
"""

import heapq
from typing import Tuple, Optional, List, Dict, Set, Any

try:
    from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
    from robot_controller import RobotController
    from item import Pan, Plate, Food, Item
except ImportError:
    pass

# --- Configuration ---
DEBUG = False

def log(msg):
    if DEBUG:
        print(f"[OptimalBot] {msg}")


# =============================================================================
# PATHFINDING - 4-directional A* with Manhattan distance
# =============================================================================

class Pathfinding:
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    @staticmethod
    def dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    @staticmethod
    def get_path(controller: RobotController, start: Tuple[int, int], 
                 target: Tuple[int, int], stop_dist: int,
                 avoid: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        m = controller.get_map(controller.get_team())
        w, h = m.width, m.height
        
        start_h = Pathfinding.dist(start, target)
        queue = [(start_h, 0, start, [])]
        visited = {start: 0}
        
        while queue:
            f, g, curr, path = heapq.heappop(queue)
            
            if Pathfinding.dist(curr, target) <= stop_dist:
                return path
            
            for dx, dy in Pathfinding.DIRECTIONS:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not m.is_tile_walkable(nx, ny):
                    continue
                if neighbor in avoid:
                    continue
                
                new_g = g + 1
                if neighbor not in visited or new_g < visited[neighbor]:
                    visited[neighbor] = new_g
                    h_score = Pathfinding.dist(neighbor, target)
                    heapq.heappush(queue, (new_g + h_score, new_g, neighbor, path + [(dx, dy)]))
        
        return None


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class BotPlayer:
    """
    Streamlined bot with state machine approach per bot.
    Bot 0 (primary): Full order pipeline
    Bot 1 (helper): Assists with buying/placing
    """
    
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Cached locations
        self.counters = []
        self.cookers = []
        self.submit_locs = []
        self.shops = []
        self.trash_locs = []
        self.sinks = []
        self.sink_tables = []
        
        # Pre-assigned locations per bot
        self.assembly_counter = None
        self.cooker_loc = None
        
        # State machine for primary bot
        self.state = 0
    
    def _init(self, controller: RobotController):
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                name = getattr(tile, 'tile_name', '')
                pos = (x, y)
                
                if name == "COUNTER": self.counters.append(pos)
                elif name == "COOKER": self.cookers.append(pos)
                elif name == "SUBMIT": self.submit_locs.append(pos)
                elif name == "SHOP": self.shops.append(pos)
                elif name == "TRASH": self.trash_locs.append(pos)
                elif name == "SINK": self.sinks.append(pos)
                elif name == "SINKTABLE": self.sink_tables.append(pos)
        
        if self.cookers:
            self.cooker_loc = self.cookers[0]
        if self.counters:
            self.assembly_counter = self.counters[0]
        
        self.initialized = True
    
    def get_closest(self, pos, locs):
        if not locs: return None
        return min(locs, key=lambda p: Pathfinding.dist(pos, p))
    
    def get_avoid_set(self, controller, exclude_bot_id):
        avoid = set()
        for bid in controller.get_team_bot_ids(controller.get_team()):
            if bid != exclude_bot_id:
                st = controller.get_bot_state(bid)
                if st:
                    avoid.add((st['x'], st['y']))
        return avoid
    
    def move_towards(self, controller, bot_id, target, stop_dist=1):
        """Move towards target. Returns True if adjacent."""
        state = controller.get_bot_state(bot_id)
        pos = (state['x'], state['y'])
        
        if Pathfinding.dist(pos, target) <= stop_dist:
            return True
        
        avoid = self.get_avoid_set(controller, bot_id)
        path = Pathfinding.get_path(controller, pos, target, stop_dist, avoid)
        
        if path:
            dx, dy = path[0]
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        
        # No path - try wiggle
        for dx, dy in Pathfinding.DIRECTIONS:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return False
        return False
    
    def find_nearest_tile(self, controller, bx, by, tile_name):
        m = controller.get_map(controller.get_team())
        best_dist = float('inf')
        best_pos = None
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].tile_name == tile_name:
                    d = Pathfinding.dist((bx, by), (x, y))
                    if d < best_dist:
                        best_dist = d
                        best_pos = (x, y)
        return best_pos
    
    def play_turn(self, controller: RobotController):
        if not self.initialized:
            self._init(controller)
        
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return
        
        # Execute primary bot with state machine
        self.execute_primary_bot(controller, my_bots[0])
        
        # Second bot helps or wiggles
        if len(my_bots) > 1:
            self.execute_helper_bot(controller, my_bots[1])
    
    def execute_primary_bot(self, controller, bot_id):
        """
        State machine for primary bot - follows duo_noodle_bot pattern.
        
        States:
        0: Check cooker for pan
        1: Buy pan
        2: Buy meat
        3: Place meat on counter
        4: Chop meat
        5: Pick up meat
        6: Place meat on pan (starts cooking)
        7: (auto-skip)
        8: Buy plate
        9: Place plate on counter
        10: Buy noodles
        11: Add noodles to plate
        12: Wait for cook, take meat
        13: Add meat to plate
        14: Pick up plate
        15: Submit
        16: Trash (recovery)
        """
        team = controller.get_team()
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']
        holding = bot_info.get('holding')
        
        kx, ky = self.cooker_loc if self.cooker_loc else (0, 0)
        cx, cy = self.assembly_counter if self.assembly_counter else (0, 0)
        
        # Recovery: if stuck holding something in certain states
        if self.state in [2, 8, 10] and holding:
            self.state = 16
        
        # State 0: Check cooker for pan
        if self.state == 0:
            tile = controller.get_tile(team, kx, ky)
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                self.state = 2  # Pan exists
            else:
                self.state = 1  # Need pan
        
        # State 1: Buy pan
        elif self.state == 1:
            if holding and holding.get('type') == 'Pan':
                if self.move_towards(controller, bot_id, (kx, ky)):
                    if controller.place(bot_id, kx, ky):
                        self.state = 2
            else:
                shop = self.find_nearest_tile(controller, bx, by, "SHOP")
                if shop:
                    if self.move_towards(controller, bot_id, shop):
                        if controller.get_team_money(team) >= ShopCosts.PAN.buy_cost:
                            controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
        
        # State 2: Buy meat
        elif self.state == 2:
            shop = self.find_nearest_tile(controller, bx, by, "SHOP")
            if shop:
                if self.move_towards(controller, bot_id, shop):
                    if controller.get_team_money(team) >= FoodType.MEAT.buy_cost:
                        if controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1]):
                            self.state = 3
        
        # State 3: Place meat on counter
        elif self.state == 3:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.place(bot_id, cx, cy):
                    self.state = 4
        
        # State 4: Chop meat
        elif self.state == 4:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.chop(bot_id, cx, cy):
                    self.state = 5
        
        # State 5: Pick up chopped meat
        elif self.state == 5:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.pickup(bot_id, cx, cy):
                    self.state = 6
        
        # State 6: Place meat on pan (starts cooking)
        elif self.state == 6:
            if self.move_towards(controller, bot_id, (kx, ky)):
                if controller.place(bot_id, kx, ky):
                    self.state = 8  # Skip state 7
        
        # State 7: (skipped - cooking starts automatically)
        elif self.state == 7:
            self.state = 8
        
        # State 8: Buy plate
        elif self.state == 8:
            shop = self.find_nearest_tile(controller, bx, by, "SHOP")
            if shop:
                if self.move_towards(controller, bot_id, shop):
                    if controller.get_team_money(team) >= ShopCosts.PLATE.buy_cost:
                        if controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1]):
                            self.state = 9
        
        # State 9: Place plate on counter
        elif self.state == 9:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.place(bot_id, cx, cy):
                    self.state = 10
        
        # State 10: Buy noodles
        elif self.state == 10:
            shop = self.find_nearest_tile(controller, bx, by, "SHOP")
            if shop:
                if self.move_towards(controller, bot_id, shop):
                    if controller.get_team_money(team) >= FoodType.NOODLES.buy_cost:
                        if controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1]):
                            self.state = 11
        
        # State 11: Add noodles to plate
        elif self.state == 11:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = 12
        
        # State 12: Wait for cook, take meat from pan
        elif self.state == 12:
            if self.move_towards(controller, bot_id, (kx, ky)):
                tile = controller.get_tile(team, kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food:
                        if pan.food.cooked_stage == 1:  # Cooked
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.state = 13
                        elif pan.food.cooked_stage == 2:  # Burnt!
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.state = 16  # Trash it
                else:
                    # Pan missing or empty - restart
                    if holding:
                        self.state = 16
                    else:
                        self.state = 2
        
        # State 13: Add cooked meat to plate
        elif self.state == 13:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = 14
        
        # State 14: Pick up complete plate
        elif self.state == 14:
            if self.move_towards(controller, bot_id, (cx, cy)):
                if controller.pickup(bot_id, cx, cy):
                    self.state = 15
        
        # State 15: Submit order
        elif self.state == 15:
            submit = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            if submit:
                if self.move_towards(controller, bot_id, submit):
                    if controller.submit(bot_id, submit[0], submit[1]):
                        self.state = 0  # Reset for next order
        
        # State 16: Trash held item (recovery)
        elif self.state == 16:
            trash = self.find_nearest_tile(controller, bx, by, "TRASH")
            if trash:
                if self.move_towards(controller, bot_id, trash):
                    if controller.trash(bot_id, trash[0], trash[1]):
                        self.state = 2  # Restart from buying meat
    
    def execute_helper_bot(self, controller, bot_id):
        """Helper bot - just wiggle randomly like duo_noodle_bot does."""
        import random
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']
        
        # Try random direction
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            dx = 1
        
        # Only move in 4 directions
        if dx != 0 and dy != 0:
            dy = 0
        
        if controller.can_move(bot_id, dx, dy):
            controller.move(bot_id, dx, dy)
