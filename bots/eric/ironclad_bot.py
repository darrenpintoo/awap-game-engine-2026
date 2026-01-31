"""
IRONCLAD BOT - Final Implementation v3.1
Based on GPT's IRONCLAD spec with iron_chef_bot's proven state machine

CORE PRINCIPLES:
- Zero wasted turns (no idle unless impossible to act)
- Zero burn tolerance (burn prevention > all other tasks)
- Deterministic behavior (same state = same action)

KEY FEATURES:
- COOK_GUARD Active Monitoring Mode (State 11)
- Global Cooking Override at turn start
- 16-state machine (0-15) like iron_chef_bot
- No "check" or wait states
- Burn prevention as top priority
"""

import time
from collections import deque
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass

from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from tiles import Tile, Counter, Cooker, Sink, SinkTable, Submit, Shop, Box, Trash
from item import Pan, Plate, Food


# ============================================
# COOKING TRACKER
# ============================================

@dataclass
class CookingTracker:
    """Track cooking status for burn prevention"""
    location: Tuple[int, int]
    food_name: str
    cook_progress: int
    cooked_stage: int
    
    @property
    def turns_to_cooked(self) -> int:
        if self.cooked_stage >= 1:
            return 0
        return max(0, GameConstants.COOK_PROGRESS - self.cook_progress)
    
    @property
    def turns_to_burned(self) -> int:
        if self.cooked_stage >= 2:
            return 0
        return max(0, GameConstants.BURN_PROGRESS - self.cook_progress)
    
    @property
    def needs_immediate_pickup(self) -> bool:
        return self.cooked_stage == 1
    
    @property
    def is_burning_soon(self) -> bool:
        return self.cooked_stage == 1 and self.turns_to_burned <= 3


# ============================================
# MAIN BOT CLASS
# ============================================

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Map data
        self.shops: List[Tuple[int, int]] = []
        self.cookers: List[Tuple[int, int]] = []
        self.sinks: List[Tuple[int, int]] = []
        self.sink_tables: List[Tuple[int, int]] = []
        self.counters: List[Tuple[int, int]] = []
        self.submits: List[Tuple[int, int]] = []
        self.trashes: List[Tuple[int, int]] = []
        self.boxes: List[Tuple[int, int]] = []
        self.walkable: Set[Tuple[int, int]] = set()
        
        # Per-bot pipeline state (like iron_chef_bot)
        # state, counter, plate_counter, cooker
        self.pipeline_state: Dict[int, Dict[str, Any]] = {}
        
        # Cooking trackers
        self.cooking_trackers: Dict[Tuple[int, int], CookingTracker] = {}
        
        # Reservation system
        self.reserved_nodes: Set[Tuple[int, int, int]] = set()  # (x, y, turn)

    # ============================================
    # INITIALIZATION
    # ============================================
    
    def initialize(self, controller: RobotController):
        if self.initialized:
            return
        
        m = controller.get_map()
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                tile_name = getattr(tile, 'tile_name', '')
                
                if getattr(tile, 'is_walkable', False):
                    self.walkable.add((x, y))
                
                if tile_name == 'SHOP':
                    self.shops.append((x, y))
                elif tile_name == 'COOKER':
                    self.cookers.append((x, y))
                elif tile_name == 'SINK':
                    self.sinks.append((x, y))
                elif tile_name == 'SINKTABLE':
                    self.sink_tables.append((x, y))
                elif tile_name == 'COUNTER':
                    self.counters.append((x, y))
                elif tile_name == 'SUBMIT':
                    self.submits.append((x, y))
                    self.walkable.add((x, y))
                elif tile_name == 'TRASH':
                    self.trashes.append((x, y))
                elif tile_name == 'BOX':
                    self.boxes.append((x, y))
        
        self.initialized = True
        print(f"[IRONCLAD] Initialized: {len(self.cookers)} cookers, {len(self.counters)} counters")

    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    def is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])) <= 1
    
    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    
    def get_distance_to_adjacent(self, start: Tuple[int, int], target: Tuple[int, int]) -> int:
        """Distance to get adjacent to a non-walkable target"""
        min_dist = float('inf')
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                adj = (target[0] + dx, target[1] + dy)
                if adj in self.walkable:
                    d = self.get_distance(start, adj)
                    min_dist = min(min_dist, d)
        return min_dist if min_dist != float('inf') else 999
    
    def get_move_toward(self, controller: RobotController, bot_id: int, 
                        target: Tuple[int, int], turn: int) -> Optional[Tuple[int, int]]:
        """BFS pathfinding with reservation system"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        
        start = (bot['x'], bot['y'])
        
        if self.is_adjacent(start, target):
            return (0, 0)
        
        queue = deque([(start, [])])
        visited = {start}
        m = controller.get_map()
        
        while queue:
            (cx, cy), path = queue.popleft()
            
            if self.is_adjacent((cx, cy), target):
                if not path:
                    return (0, 0)
                step = path[0]
                next_pos = (start[0] + step[0], start[1] + step[1])
                self.reserved_nodes.add((next_pos[0], next_pos[1], turn + 1))
                return step
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    
                    if (nx, ny) in visited:
                        continue
                    if not (0 <= nx < m.width and 0 <= ny < m.height):
                        continue
                    if not m.is_tile_walkable(nx, ny):
                        continue
                    
                    step_turn = turn + len(path) + 1
                    if (nx, ny, step_turn) in self.reserved_nodes:
                        continue
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(dx, dy)]))
        
        return None
    
    def find_nearest(self, pos: Tuple[int, int], 
                     locations: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not locations:
            return None
        return min(locations, key=lambda loc: self.get_distance_to_adjacent(pos, loc))
    
    def find_empty_counter(self, controller: RobotController, 
                           near: Tuple[int, int], 
                           exclude: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        if exclude is None:
            exclude = set()
        
        best = None
        best_dist = float('inf')
        
        for cx, cy in self.counters:
            if (cx, cy) in exclude:
                continue
            tile = controller.get_tile(controller.get_team(), cx, cy)
            if tile and getattr(tile, 'item', None) is None:
                dist = self.get_distance_to_adjacent(near, (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best = (cx, cy)
        
        return best
    
    def find_cooker_with_empty_pan(self, controller: RobotController, 
                                    near: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        best = None
        best_dist = float('inf')
        
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile:
                pan = getattr(tile, 'item', None)
                if isinstance(pan, Pan) and pan.food is None:
                    dist = self.get_distance_to_adjacent(near, (kx, ky))
                    if dist < best_dist:
                        best_dist = dist
                        best = (kx, ky)
        
        return best

    # ============================================
    # COOKING STATE UPDATE
    # ============================================
    
    def update_cooking_trackers(self, controller: RobotController):
        self.cooking_trackers.clear()
        
        for kx, ky in self.cookers:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if pan.food is not None:
                    self.cooking_trackers[(kx, ky)] = CookingTracker(
                        location=(kx, ky),
                        food_name=pan.food.food_name,
                        cook_progress=getattr(tile, 'cook_progress', 0),
                        cooked_stage=pan.food.cooked_stage
                    )

    # ============================================
    # GLOBAL COOKING OVERRIDE (RULE 4)
    # ============================================
    
    def check_cooking_emergency(self, controller: RobotController) -> Optional[Tuple[int, int]]:
        """
        Check if there's a cooking emergency that needs immediate attention.
        Returns the cooker location if emergency exists.
        """
        for loc, tracker in self.cooking_trackers.items():
            if tracker.needs_immediate_pickup or tracker.is_burning_soon:
                return loc
        return None

    # ============================================
    # CHEF PIPELINE (16 states, 0-15)
    # Copied from iron_chef_bot's proven state machine
    # ============================================
    
    def run_cooking_pipeline(self, controller: RobotController, bot_id: int):
        """
        State machine for order completion.
        Uses per-bot state dictionary like iron_chef_bot.
        """
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        # Initialize pipeline state for this bot
        if bot_id not in self.pipeline_state:
            self.pipeline_state[bot_id] = {
                'state': 0,
                'counter': None,
                'plate_counter': None,
                'cooker': None
            }
        
        state = self.pipeline_state[bot_id]
        current_state = state['state']
        
        # ====== GLOBAL COOKING OVERRIDE ======
        # If food is about to burn, ALL bots must help
        emergency_cooker = self.check_cooking_emergency(controller)
        if emergency_cooker and not holding:
            # Override current state - go grab the cooked food
            step = self.get_move_toward(controller, bot_id, emergency_cooker, turn)
            if self.is_adjacent((bx, by), emergency_cooker):
                if controller.take_from_pan(bot_id, emergency_cooker[0], emergency_cooker[1]):
                    # Successfully grabbed - now need to add to plate or trash
                    state['state'] = 12  # Add to plate state
            elif step:
                controller.move(bot_id, step[0], step[1])
            return
        
        # ====== STATE MACHINE ======
        
        if current_state == 0:
            # Check if pan exists
            has_pan = any(
                isinstance(getattr(controller.get_tile(controller.get_team(), kx, ky), 'item', None), Pan)
                for kx, ky in self.cookers
            )
            state['state'] = 2 if has_pan else 1
            
        elif current_state == 1:
            # Buy and place pan
            if holding and holding.get('type') == 'Pan':
                cooker = self.find_nearest((bx, by), self.cookers)
                if cooker:
                    step = self.get_move_toward(controller, bot_id, cooker, turn)
                    if self.is_adjacent((bx, by), cooker):
                        controller.place(bot_id, cooker[0], cooker[1])
                        state['state'] = 2
                    elif step:
                        controller.move(bot_id, step[0], step[1])
            else:
                shop = self.find_nearest((bx, by), self.shops)
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        controller.buy(bot_id, ShopCosts.PAN, shop[0], shop[1])
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 2:
            # Buy meat
            if holding:
                state['state'] = 3
            else:
                shop = self.find_nearest((bx, by), self.shops)
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, FoodType.MEAT, shop[0], shop[1]):
                            state['state'] = 3
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 3:
            # Place on counter
            exclude = {state.get('plate_counter')} if state.get('plate_counter') else set()
            counter = self.find_empty_counter(controller, (bx, by), exclude)
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.place(bot_id, counter[0], counter[1]):
                        state['counter'] = counter
                        state['state'] = 4
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 4:
            # Chop
            counter = state.get('counter')
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.chop(bot_id, counter[0], counter[1]):
                        state['state'] = 5
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 5:
            # Pickup chopped
            counter = state.get('counter')
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.pickup(bot_id, counter[0], counter[1]):
                        state['state'] = 6
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 6:
            # Start cooking
            cooker = self.find_cooker_with_empty_pan(controller, (bx, by))
            if cooker:
                step = self.get_move_toward(controller, bot_id, cooker, turn)
                if self.is_adjacent((bx, by), cooker):
                    if controller.place(bot_id, cooker[0], cooker[1]):
                        state['cooker'] = cooker
                        state['state'] = 7
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 7:
            # Buy plate
            if holding:
                state['state'] = 8
            else:
                shop = self.find_nearest((bx, by), self.shops)
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, ShopCosts.PLATE, shop[0], shop[1]):
                            state['state'] = 8
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 8:
            # Place plate
            exclude = {state.get('counter')} if state.get('counter') else set()
            counter = self.find_empty_counter(controller, (bx, by), exclude)
            if counter:
                step = self.get_move_toward(controller, bot_id, counter, turn)
                if self.is_adjacent((bx, by), counter):
                    if controller.place(bot_id, counter[0], counter[1]):
                        state['plate_counter'] = counter
                        state['state'] = 9
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 9:
            # Buy noodles
            if holding:
                state['state'] = 10
            else:
                shop = self.find_nearest((bx, by), self.shops)
                if shop:
                    step = self.get_move_toward(controller, bot_id, shop, turn)
                    if self.is_adjacent((bx, by), shop):
                        if controller.buy(bot_id, FoodType.NOODLES, shop[0], shop[1]):
                            state['state'] = 10
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        
        elif current_state == 10:
            # Add noodles to plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 11
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 11:
            # COOK_GUARD: Wait for meat and take from pan
            cooker = state.get('cooker')
            if cooker:
                step = self.get_move_toward(controller, bot_id, cooker, turn)
                if self.is_adjacent((bx, by), cooker):
                    tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
                    if tile and isinstance(getattr(tile, 'item', None), Pan):
                        pan = tile.item
                        if pan.food and pan.food.cooked_stage == 1:
                            # COOKED - grab it NOW
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                state['state'] = 12
                        elif pan.food and pan.food.cooked_stage == 2:
                            # BURNED - trash and restart
                            if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                state['state'] = 15
                    # If not cooked yet, stay adjacent (this IS cook guard)
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 12:
            # Add cooked food to plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 13
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 13:
            # Pickup plate
            plate_counter = state.get('plate_counter')
            if plate_counter:
                step = self.get_move_toward(controller, bot_id, plate_counter, turn)
                if self.is_adjacent((bx, by), plate_counter):
                    if controller.pickup(bot_id, plate_counter[0], plate_counter[1]):
                        state['state'] = 14
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 14:
            # Submit
            submit = self.find_nearest((bx, by), self.submits)
            if submit:
                step = self.get_move_toward(controller, bot_id, submit, turn)
                if self.is_adjacent((bx, by), submit):
                    if controller.submit(bot_id, submit[0], submit[1]):
                        # SUCCESS! Reset for next order
                        state['state'] = 0
                        state['counter'] = None
                        state['plate_counter'] = None
                        state['cooker'] = None
                elif step:
                    controller.move(bot_id, step[0], step[1])
                    
        elif current_state == 15:
            # Trash burned food and restart
            if holding:
                trash = self.find_nearest((bx, by), self.trashes)
                if trash:
                    step = self.get_move_toward(controller, bot_id, trash, turn)
                    if self.is_adjacent((bx, by), trash):
                        controller.trash(bot_id, trash[0], trash[1])
                        state['state'] = 2  # Restart from buying meat
                    elif step:
                        controller.move(bot_id, step[0], step[1])
            else:
                state['state'] = 2

    # ============================================
    # SUPPORT BOT LOGIC
    # ============================================
    
    def run_support_pipeline(self, controller: RobotController, bot_id: int):
        """Support bot: washes dishes, stays out of way"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        turn = controller.get_turn()
        
        # Priority: Wash dishes
        for sx, sy in self.sinks:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                step = self.get_move_toward(controller, bot_id, (sx, sy), turn)
                if self.is_adjacent((bx, by), (sx, sy)):
                    controller.wash_sink(bot_id, sx, sy)
                elif step:
                    controller.move(bot_id, step[0], step[1])
                return
        
        # Otherwise move randomly (don't block)
        import random
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        random.shuffle(directions)
        for dx, dy in directions:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return

    # ============================================
    # SABOTAGE MODE (WHEN SWITCHED)
    # ============================================
    
    def run_sabotage(self, controller: RobotController, bot_id: int):
        """Sabotage when on enemy map"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        turn = controller.get_turn()
        
        # If holding pan, move away
        if holding and holding.get('type') == 'Pan':
            import random
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            random.shuffle(directions)
            for dx, dy in directions:
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    return
            return
        
        # Steal pan from cooker
        if not holding:
            for kx, ky in self.cookers:
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    step = self.get_move_toward(controller, bot_id, (kx, ky), turn)
                    if self.is_adjacent((bx, by), (kx, ky)):
                        controller.pickup(bot_id, kx, ky)
                        return
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        return
        
        # Steal plate
        if not holding:
            for sx, sy in self.sink_tables:
                tile = controller.get_tile(controller.get_team(), sx, sy)
                if tile and getattr(tile, 'num_clean_plates', 0) > 0:
                    step = self.get_move_toward(controller, bot_id, (sx, sy), turn)
                    if self.is_adjacent((bx, by), (sx, sy)):
                        controller.take_clean_plate(bot_id, sx, sy)
                        return
                    elif step:
                        controller.move(bot_id, step[0], step[1])
                        return
        
        # Trash held plate
        if holding and holding.get('type') == 'Plate':
            trash = self.find_nearest((bx, by), self.trashes)
            if trash:
                step = self.get_move_toward(controller, bot_id, trash, turn)
                if self.is_adjacent((bx, by), trash):
                    controller.trash(bot_id, trash[0], trash[1])
                elif step:
                    controller.move(bot_id, step[0], step[1])

    # ============================================
    # MAIN ENTRY POINT
    # ============================================
    
    def play_turn(self, controller: RobotController):
        """Main entry point called each turn"""
        start_time = time.time()
        
        try:
            # Initialize on first turn
            self.initialize(controller)
            
            # Update cooking trackers
            self.update_cooking_trackers(controller)
            
            # Clear per-turn reservations
            self.reserved_nodes.clear()
            
            # Get bot IDs
            bot_ids = controller.get_team_bot_ids()
            if not bot_ids:
                return
            
            # Check switch window status
            switch_info = controller.get_switch_info()
            turn = controller.get_turn()
            
            # Sabotage decision - DISABLED for stability
            # The original iron_chef_bot disables sabotage for maximum cooking efficiency
            is_sabotaging = switch_info.get('my_team_switched', False)
            
            # Run each bot
            for i, bot_id in enumerate(bot_ids):
                if is_sabotaging:
                    self.run_sabotage(controller, bot_id)
                elif i == 0:
                    # Main bot runs cooking pipeline
                    self.run_cooking_pipeline(controller, bot_id)
                else:
                    # Support bots wash dishes
                    self.run_support_pipeline(controller, bot_id)
                    
        except Exception as e:
            print(f"[IRONCLAD ERROR] Turn failed: {e}")
            
        elapsed = time.time() - start_time
        if elapsed > 0.4:
            print(f"[IRONCLAD WARN] Turn took {elapsed:.3f}s")
