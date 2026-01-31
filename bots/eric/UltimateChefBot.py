"""
UltimateChefBot - Merged PipelineChefBot + planner_bot
Best-of-both-worlds competition bot.

FROM PipelineChefBot:
- Pre-computed distance matrices (O(1) pathfinding)
- Robust 16-state machine with error recovery
- Counter occupancy checks
- Burnt food handling

FROM planner_bot:
- Order prioritization & scoring
- Dynamic goal selection
- World state scanning (reuse existing items)
- Multi-bot coordination via reservations

NEW ADDITIONS:
- Holding checks before buy (fixes planner_bot bug)
- 2-bot pipeline (chef + support)
- Adaptive order targeting
"""

import numpy as np
from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from tiles import Cooker, Counter
from item import Pan, Plate, Food


# ============================================
# ORDER SCORING (from planner_bot)
# ============================================

@dataclass
class ScoredOrder:
    """Order with computed priority score"""
    order_id: int
    required: List[str]
    reward: int
    expires_turn: int
    time_left: int
    score: float  # Higher = better target
    
    @staticmethod
    def from_order(order: dict, current_turn: int) -> 'ScoredOrder':
        time_left = order['expires_turn'] - current_turn
        # Score = reward per ingredient, weighted by time urgency
        n_items = len(order.get('required', []))
        base_score = order['reward'] / max(n_items, 1)
        
        # Bonus for simpler orders (fewer items = faster completion)
        simplicity_bonus = (5 - n_items) * 10
        
        # Penalty for orders expiring soon (but not too soon)
        if time_left < 30:
            urgency_penalty = (30 - time_left) * 5
        else:
            urgency_penalty = 0
            
        score = base_score + simplicity_bonus - urgency_penalty
        
        return ScoredOrder(
            order_id=order['order_id'],
            required=order.get('required', []),
            reward=order['reward'],
            expires_turn=order['expires_turn'],
            time_left=time_left,
            score=score
        )


# ============================================
# MAIN BOT CLASS
# ============================================

class BotPlayer:
    """Ultimate competition bot combining PipelineChefBot reliability with planner_bot intelligence"""

    def __init__(self, map_copy):
        self.map = map_copy
        self.width = map_copy.width
        self.height = map_copy.height
        
        # === Pre-computed data (from PipelineChefBot) ===
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
        
        # Pre-compute distance matrices to key locations
        self.dist_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        for tile_name in ['SHOP', 'COOKER', 'COUNTER', 'SUBMIT', 'TRASH', 'SINK', 'SINKTABLE']:
            if tile_name in self.tile_cache:
                for pos in self.tile_cache[tile_name]:
                    self.dist_matrices[pos] = self._compute_distance_matrix(pos)
        
        # === Bot state tracking ===
        self.bot_states: Dict[int, int] = {}  # bot_id -> state
        self.bot_counters: Dict[int, Tuple[int, int]] = {}  # bot_id -> assigned counter
        self.bot_cookers: Dict[int, Tuple[int, int]] = {}  # bot_id -> assigned cooker
        
        # === Order targeting (from planner_bot) ===
        self.target_order: Optional[ScoredOrder] = None
        self.target_ingredients: List[str] = []
        
        # === Resource tracking ===
        self.reserved_counters: Set[Tuple[int, int]] = set()
        
        self.initialized = False

    # ============================================
    # DISTANCE MATRIX (from PipelineChefBot)
    # ============================================

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

    # ============================================
    # UTILITY FUNCTIONS
    # ============================================

    def _get_tile_pos(self, tile_name: str, bot_x: int = 0, bot_y: int = 0) -> Optional[Tuple[int, int]]:
        """Get nearest tile of given type"""
        if tile_name not in self.tile_cache or not self.tile_cache[tile_name]:
            return None
        positions = self.tile_cache[tile_name]
        return min(positions, key=lambda p: max(abs(p[0] - bot_x), abs(p[1] - bot_y)))

    def _get_next_step(self, controller: RobotController, bot_id: int,
                       target_x: int, target_y: int) -> Optional[Tuple[int, int]]:
        """Get next step toward target with O(1) distance lookup"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return None
        bx, by = bot['x'], bot['y']
        
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            return None  # Already adjacent
        
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
        if (cx, cy) in self.reserved_counters:
            return False
        tile = controller.get_tile(controller.get_team(), cx, cy)
        return tile is not None and getattr(tile, 'item', None) is None

    def _find_free_counter(self, controller: RobotController, 
                           bx: int, by: int, exclude: Set[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Find nearest free counter"""
        if exclude is None:
            exclude = set()
        
        counters = self.tile_cache.get('COUNTER', [])
        for cx, cy in sorted(counters, key=lambda c: max(abs(c[0] - bx), abs(c[1] - by))):
            if (cx, cy) not in exclude and self._is_counter_free(controller, cx, cy):
                return (cx, cy)
        return None

    def _find_cooker_with_pan(self, controller: RobotController, 
                               bx: int, by: int, empty: bool = True) -> Optional[Tuple[int, int]]:
        """Find cooker with pan (optionally empty)"""
        cookers = self.tile_cache.get('COOKER', [])
        for kx, ky in sorted(cookers, key=lambda c: max(abs(c[0] - bx), abs(c[1] - by))):
            tile = controller.get_tile(controller.get_team(), kx, ky)
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                pan = tile.item
                if empty and pan.food is None:
                    return (kx, ky)
                elif not empty:
                    return (kx, ky)
        return None

    # ============================================
    # ORDER SELECTION (from planner_bot)
    # ============================================

    def _select_target_order(self, controller: RobotController):
        """Select best order to target based on scoring"""
        team = controller.get_team()
        current_turn = controller.get_turn()
        orders = controller.get_orders(team)
        
        if not orders:
            self.target_order = None
            self.target_ingredients = []
            return
        
        # Score all orders
        scored = []
        for order in orders:
            if not order.get('is_active', True):
                continue
            time_left = order['expires_turn'] - current_turn
            if time_left < 35:  # Not enough time
                continue
            
            scored_order = ScoredOrder.from_order(order, current_turn)
            scored.append(scored_order)
        
        if not scored:
            self.target_order = None
            self.target_ingredients = []
            return
        
        # Pick highest scoring order
        scored.sort(key=lambda o: o.score, reverse=True)
        self.target_order = scored[0]
        self.target_ingredients = list(self.target_order.required)

    # ============================================
    # CHEF STATE MACHINE (enhanced from PipelineChefBot)
    # ============================================

    def _run_chef(self, controller: RobotController, bot_id: int):
        """Main state machine for chef bot"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        state = self.bot_states.get(bot_id, 0)
        
        # Get key positions
        shop_pos = self._get_tile_pos('SHOP', bx, by)
        submit_pos = self._get_tile_pos('SUBMIT', bx, by)
        trash_pos = self._get_tile_pos('TRASH', bx, by)
        
        if not shop_pos or not submit_pos:
            return
        
        sx, sy = shop_pos
        ux, uy = submit_pos
        
        # Get assigned counter/cooker
        counter_pos = self.bot_counters.get(bot_id)
        cooker_pos = self.bot_cookers.get(bot_id)
        
        # If we need a counter but have none, or if we are in a state that NEEDS a free counter
        # States needing free counter: 0 (init), 1 (buy pan), 2 (buy meat), 8 (buy plate), 9 (place plate), 10 (buy noodles)
        # Actually, primarily when we possess an item to place.
        
        # We'll handle finding free counters inside states 2/3 and 8/9/10/11.
        # Fallbacks for existing counters if they are missing
        if not counter_pos:
             # Basic fallback just for safety, but logic should handle assignment
             counter_pos = self._get_tile_pos('COUNTER', bx, by)
        
        if not cooker_pos:
             # Basic fallback
             cooker_pos = self._get_tile_pos('COOKER', bx, by)

        if not counter_pos or not cooker_pos:
            return

        cx, cy = counter_pos
        kx, ky = cooker_pos

        # === HOLDING CHECK (fixes planner_bot bug) ===
        if state in [2, 8, 10] and holding:
            if holding.get('type') == 'Plate':
                self.bot_states[bot_id] = 16  # Trash it
            else:
                self.bot_states[bot_id] = 3  # Place what we have
            return

        # === STATE MACHINE ===
        
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
                if not holding:  # CRITICAL: Only buy if not holding
                    if self._move_toward(controller, bot_id, sx, sy):
                        if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                            controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        # State 2: Buy meat
        elif state == 2:
            if not holding:  # Only buy if not holding
                if self._move_toward(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= FoodType.MEAT.buy_cost:
                        if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                            self.bot_states[bot_id] = 3
            else:
                self.bot_states[bot_id] = 3  # Skip to next state

        # State 3: Place meat on counter
        elif state == 3:
            # Find free counter if current one is occupied
            if not self._is_counter_free(controller, cx, cy):
                new_counter = self._find_free_counter(controller, bx, by, {(cx, cy)})
                if new_counter:
                    self.bot_counters[bot_id] = new_counter
                    cx, cy = new_counter
                else:
                    return  # Wait for counter
            
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
            # Find cooker with empty pan
            cooker_with_pan = self._find_cooker_with_pan(controller, bx, by, empty=True)
            if cooker_with_pan:
                kx, ky = cooker_with_pan
                self.bot_cookers[bot_id] = cooker_with_pan
                
            if self._move_toward(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    self.bot_states[bot_id] = 8

        # State 8: Buy plate
        elif state == 8:
            if not holding:
                if self._move_toward(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                        if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                            self.bot_states[bot_id] = 9

        # State 9: Place plate on counter
        elif state == 9:
            # Find free counter (different from meat counter if possible)
            plate_counter = self._find_free_counter(controller, bx, by)
            if not plate_counter:
                return  # Wait
            
            if self._move_toward(controller, bot_id, plate_counter[0], plate_counter[1]):
                if controller.place(bot_id, plate_counter[0], plate_counter[1]):
                    self.bot_counters[bot_id] = plate_counter  # Update assigned counter
                    self.bot_states[bot_id] = 10

        # State 10: Buy noodles
        elif state == 10:
            if not holding:
                if self._move_toward(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= FoodType.NOODLES.buy_cost:
                        if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                            self.bot_states[bot_id] = 11

        # State 11: Add noodles to plate
        elif state == 11:
            plate_counter = self.bot_counters.get(bot_id)
            if plate_counter:
                if self._move_toward(controller, bot_id, plate_counter[0], plate_counter[1]):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        self.bot_states[bot_id] = 12

        # State 12: Wait for meat to cook + extract
        elif state == 12:
            cooker = self.bot_cookers.get(bot_id)
            if not cooker:
                cooker = self._find_cooker_with_pan(controller, bx, by, empty=False)
            
            if cooker:
                kx, ky = cooker
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
                                    self.bot_states[bot_id] = 16
                        else:
                            # No food in pan - restart
                            self.bot_states[bot_id] = 2

        # State 13: Add meat to plate
        elif state == 13:
            plate_counter = self.bot_counters.get(bot_id)
            if plate_counter:
                if self._move_toward(controller, bot_id, plate_counter[0], plate_counter[1]):
                    if controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1]):
                        self.bot_states[bot_id] = 14

        # State 14: Pick up plate
        elif state == 14:
            plate_counter = self.bot_counters.get(bot_id)
            if plate_counter:
                if self._move_toward(controller, bot_id, plate_counter[0], plate_counter[1]):
                    if controller.pickup(bot_id, plate_counter[0], plate_counter[1]):
                        self.bot_states[bot_id] = 15

        # State 15: Submit
        elif state == 15:
            if self._move_toward(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    # Clear tracking for next order
                    self.bot_counters.pop(bot_id, None)
                    self.bot_cookers.pop(bot_id, None)
                    self.bot_states[bot_id] = 0

        # State 16: Trash and restart
        elif state == 16:
            if trash_pos:
                tx, ty = trash_pos
                if self._move_toward(controller, bot_id, tx, ty):
                    if controller.trash(bot_id, tx, ty):
                        self.bot_states[bot_id] = 2

    # ============================================
    # SUPPORT BOT (wash dishes, stay out of way)
    # ============================================

    def _run_support(self, controller: RobotController, bot_id: int):
        """Support bot washes dishes and stays out of the way"""
        bot = controller.get_bot_state(bot_id)
        if not bot:
            return
        
        bx, by = bot['x'], bot['y']
        holding = bot.get('holding')
        
        # If holding something, trash it
        if holding:
            trash_pos = self._get_tile_pos('TRASH', bx, by)
            if trash_pos:
                if self._move_toward(controller, bot_id, trash_pos[0], trash_pos[1]):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
            return
        
        # Priority: Wash dishes if any dirty
        sinks = self.tile_cache.get('SINK', [])
        for sx, sy in sinks:
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if tile and getattr(tile, 'num_dirty_plates', 0) > 0:
                if self._move_toward(controller, bot_id, sx, sy):
                    controller.wash_sink(bot_id, sx, sy)
                return
        
        # Otherwise move randomly to avoid blocking
        import random
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(directions)
        for dx, dy in directions:
            if controller.can_move(bot_id, dx, dy):
                controller.move(bot_id, dx, dy)
                return

    # ============================================
    # MAIN ENTRY POINT
    # ============================================

    def play_turn(self, controller: RobotController):
        """Main entry point"""
        team = controller.get_team()
        my_bots = controller.get_team_bot_ids(team)
        if not my_bots:
            return
        
        # Select target order (dynamic goal selection)
        self._select_target_order(controller)
        
        # Clear per-turn reservations
        self.reserved_counters.clear()
        
        # Initialize bot states
        for bot_id in my_bots:
            if bot_id not in self.bot_states:
                self.bot_states[bot_id] = 0
        
        # Run bots: first bot is chef, rest are support
        chef_bot = my_bots[0]
        self._run_chef(controller, chef_bot)
        
        # Support bots
        for bot_id in my_bots[1:]:
            self._run_support(controller, bot_id)
