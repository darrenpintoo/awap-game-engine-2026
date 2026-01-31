"""
Optimal Competition Bot for Carnegie Cookoff (AWAP 2026)

Features:
- Pre-computed pathfinding with dynamic obstacle avoidance
- Robust state machine with proper error recovery
- Handles counter occupancy correctly
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List, Dict, Any

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from tiles import Cooker, Counter
from item import Pan, Plate, Food


class BotPlayer:
    """Optimized bot with pre-computed paths and robust state handling"""

    def __init__(self, map_copy):
        self.map = map_copy
        self.width = map_copy.width
        self.height = map_copy.height
        
        # Pre-compute walkability matrix
        self.walkable = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                self.walkable[x, y] = getattr(map_copy.tiles[x][y], 'is_walkable', False)
        
        # Cache tile locations by type
        self.tile_cache: Dict[str, List[Tuple[int, int]]] = {}
        for x in range(self.width):
            for y in range(self.height):
                tile_name = map_copy.tiles[x][y].tile_name
                if tile_name not in self.tile_cache:
                    self.tile_cache[tile_name] = []
                self.tile_cache[tile_name].append((x, y))
        
        # Pre-compute distance matrices to key tile positions
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        for tile_name in ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE']:
            if tile_name in self.tile_cache:
                for pos in self.tile_cache[tile_name]:
                    self.dist_matrices[pos] = self._compute_distance_matrix(pos)
        
        # Bot states
        self.bot_states: Dict[int, int] = {}
        self.initialized = False

    def _compute_distance_matrix(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS to compute distance from every tile to adjacent-to-target"""
        dist = np.full((self.width, self.height), 9999.0)
        tx, ty = target
        
        queue = deque()
        # Start from tiles adjacent to target
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.walkable[nx, ny]:
                        dist[nx, ny] = 0
                        queue.append((nx, ny))
        
        while queue:
            x, y = queue.popleft()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.walkable[nx, ny] and dist[nx, ny] > dist[x, y] + 1:
                            dist[nx, ny] = dist[x, y] + 1
                            queue.append((nx, ny))
        return dist

    def _get_tile_pos(self, tile_name: str, bot_x: int = 0, bot_y: int = 0) -> Optional[Tuple[int, int]]:
        """Get nearest tile of given type"""
        if tile_name not in self.tile_cache or not self.tile_cache[tile_name]:
            return None
        positions = self.tile_cache[tile_name]
        return min(positions, key=lambda p: max(abs(p[0] - bot_x), abs(p[1] - bot_y)))

    def _get_next_step(self, controller: RobotController, bot_id: int,
                       target_x: int, target_y: int) -> Optional[Tuple[int, int]]:
        """Get next step toward target with dynamic obstacle avoidance"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        bx, by = bot['x'], bot['y']
        
        # Already adjacent?
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return None  # No movement needed
        
        # Get pre-computed distances if available
        target_key = (target_x, target_y)
        dist_matrix = self.dist_matrices.get(target_key)
        
        best_step = None
        best_dist = 9999.0
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if not controller.can_move(bot_id, dx, dy):
                continue
            
            nx, ny = bx + dx, by + dy
            if dist_matrix is not None:
                step_dist = dist_matrix[nx, ny]
            else:
                step_dist = max(abs(nx - target_x), abs(ny - target_y))
            
            if step_dist < best_dist:
                best_dist = step_dist
                best_step = (dx, dy)
        
        return best_step

    def _move_toward(self, controller: RobotController, bot_id: int,
                     target_x: int, target_y: int) -> bool:
        """Move toward target. Returns True if adjacent."""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return False
        bx, by = bot['x'], bot['y']
        
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return True
        
        step = self._get_next_step(controller, bot_id, target_x, target_y)
        if step:
            controller.move(bot_id, step[0], step[1])
        return False

    def _is_counter_free(self, controller: RobotController, cx: int, cy: int) -> bool:
        """Check if counter is empty"""
        tile = controller.get_tile(controller.get_team(), cx, cy)
        return tile is not None and getattr(tile, 'item', None) is None

    def _play_bot(self, controller: RobotController, bot_id: int):
        """Main state machine for a single bot"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        state = self.bot_states.get(bot_id, 0)
        
        # Get key positions
        cooker_pos = self._get_tile_pos('COOKER', bx, by)
        counter_pos = self._get_tile_pos('COUNTER', bx, by)
        shop_pos = self._get_tile_pos('SHOP', bx, by)
        submit_pos = self._get_tile_pos('SUBMIT', bx, by)
        trash_pos = self._get_tile_pos('TRASH', bx, by)
        
        if not all([cooker_pos, counter_pos, shop_pos, submit_pos]):
            return
        
        kx, ky = cooker_pos
        cx, cy = counter_pos
        sx, sy = shop_pos
        ux, uy = submit_pos

        # Error recovery: if in buying states but already holding, skip ahead
        if state in [2, 8, 10] and holding:
            if holding.get('type') == 'Plate':
                self.bot_states[bot_id] = 16  # Trash it
            else:
                self.bot_states[bot_id] = 3  # Place what we have
            return

        # State 0: Initialize - check cooker for pan
        if state == 0:
            tile = controller.get_tile(controller.get_team(), kx, ky)
            has_pan = tile and isinstance(getattr(tile, 'item', None), Pan)
            self.bot_states[bot_id] = 2 if has_pan else 1

        # State 1: Buy pan
        elif state == 1:
            if holding and holding.get('type') == 'Pan':
                if self._move_toward(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        self.bot_states[bot_id] = 2
            else:
                if self._move_toward(controller, bot_id, sx, sy):
                    if controller.get_team_money() >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        # State 2: Buy meat
        elif state == 2:
            if self._move_toward(controller, bot_id, sx, sy):
                if controller.get_team_money() >= FoodType.MEAT.buy_cost:
                    if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                        self.bot_states[bot_id] = 3

        # State 3: Place meat on counter (check if free first)
        elif state == 3:
            if not self._is_counter_free(controller, cx, cy):
                # Counter occupied - wait or find another counter
                # For now, just wait
                return
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.bot_states[bot_id] = 4

        # State 4: Chop meat
        elif state == 4:
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.chop(bot_id, cx, cy):
                    self.bot_states[bot_id] = 5

        # State 5: Pick up chopped meat
        elif state == 5:
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.bot_states[bot_id] = 6

        # State 6: Put meat in cooker
        elif state == 6:
            if self._move_toward(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    self.bot_states[bot_id] = 8  # Skip to buying plate

        # State 8: Buy plate
        elif state == 8:
            if self._move_toward(controller, bot_id, sx, sy):
                if controller.get_team_money() >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        self.bot_states[bot_id] = 9

        # State 9: Place plate on counter
        elif state == 9:
            if not self._is_counter_free(controller, cx, cy):
                return  # Wait for counter
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.bot_states[bot_id] = 10

        # State 10: Buy noodles
        elif state == 10:
            if self._move_toward(controller, bot_id, sx, sy):
                if controller.get_team_money() >= FoodType.NOODLES.buy_cost:
                    if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                        self.bot_states[bot_id] = 11

        # State 11: Add noodles to plate (holding noodles, plate on counter)
        elif state == 11:
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.bot_states[bot_id] = 12

        # State 12: Wait for meat to cook
        elif state == 12:
            if self._move_toward(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    pan = tile.item
                    if pan.food:
                        if pan.food.cooked_stage == 1:  # Cooked
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.bot_states[bot_id] = 13
                        elif pan.food.cooked_stage == 2:  # Burnt
                            if controller.take_from_pan(bot_id, kx, ky):
                                self.bot_states[bot_id] = 16  # Trash
                        # else still cooking, wait
                    else:
                        # No food in pan - something went wrong, restart
                        self.bot_states[bot_id] = 2
                else:
                    # No pan - restart
                    self.bot_states[bot_id] = 0

        # State 13: Add meat to plate
        elif state == 13:
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.bot_states[bot_id] = 14

        # State 14: Pick up plate
        elif state == 14:
            if self._move_toward(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.bot_states[bot_id] = 15

        # State 15: Submit
        elif state == 15:
            if self._move_toward(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    self.bot_states[bot_id] = 0  # Restart

        # State 16: Trash held item and restart
        elif state == 16:
            if trash_pos:
                tx, ty = trash_pos
                if self._move_toward(controller, bot_id, tx, ty):
                    if controller.trash(bot_id, tx, ty):
                        self.bot_states[bot_id] = 2

    def play_turn(self, controller: RobotController):
        """Main entry point"""
        my_bots = controller.get_team_bot_ids()
        if not my_bots:
            return
        
        # Only use first bot (simpler, avoids coordination bugs)
        # Additional bots just idle to avoid interference
        main_bot = my_bots[0]
        
        if main_bot not in self.bot_states:
            self.bot_states[main_bot] = 0
        
        self._play_bot(controller, main_bot)
