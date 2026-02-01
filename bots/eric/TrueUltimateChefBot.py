"""
TrueUltimateChefBot.py
The Apex of Eric's Kitchen Engineering.
Merges UltimateChefBot (Stability) with IronChef (Traffic Control).
Features:
- 16-State Cooking Pipeline (Bug-Free)
- Reservation-Based A* Pathfinding (prevents bot collisions)
- Pre-computed Distance Heuristics (O(1) guidance for A*)
- Dynamic Support Role (Washing/Box Utilization)
"""
import heapq
import random
import numpy as np
from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict, Any, Set
from dataclasses import dataclass

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food
from tiles import Counter, Box

@dataclass
class ScoredOrder:
    order_id: int
    score: float
    required_items: List[str]
    reward: int
    
    @classmethod
    def from_order(cls, order: dict, current_money: int, dist_matrix: dict, bot_pos: Tuple[int, int]) -> 'ScoredOrder':
        # Simple scoring: Reward / Size
        # Can be enhanced with distance metrics
        return cls(
            order_id=order['id'],
            score=order['reward'] / len(order['required']),
            required_items=order['required'],
            reward=order['reward']
        )

class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.initialized = False
        
        # Navigation
        self.width = map_copy.width
        self.height = map_copy.height
        self.walkable = set()
        self.tile_cache = defaultdict(list)
        self.dist_matrices = {} # Pre-computed BFS matrices for Heuristics
        self.reserved_nodes = set() # (x, y, turn)
        
        # State
        self.bot_states = {} # State machine index per bot
        self.bot_counters = {} # Assigned counter per bot
        self.bot_cookers = {} # Assigned cooker per bot
        self.reserved_counters = set() # Per-turn reservation for assignment
        self.target_ingredients = {} # Planned ingredients
        self.state_turns = defaultdict(int) # Track how long each bot stays in a state

    def initialize(self, controller: RobotController):
        if self.initialized:
            return
            
        m = controller.get_map(controller.get_team())
        
        # Parse Map
        for x in range(self.width):
            for y in range(self.height):
                tile = m.tiles[x][y]
                if tile.is_walkable:
                    self.walkable.add((x, y))
                
                tn = getattr(tile, 'tile_name', '')
                if tn:
                    self.tile_cache[tn].append((x, y))
                    
        # Compute Heuristics for Key POIs
        # We compute distance FROM every tile TO each POI type
        # This serves as the H(n) for A*
        pois = []
        for cat in ['SHOP', 'COOKER', 'SINK', 'SINKTABLE', 'submit', 'TRASH', 'COUNTER']: # submit is usually 'SUBMIT'
             pois.extend(self.tile_cache.get(cat, []))
        if self.width * self.height <= 300:
             # Add Boxes to Counters only on small maps where space is tight
             self.tile_cache['COUNTER'].extend(self.tile_cache.get('BOX', []))
             pois.extend(self.tile_cache.get('BOX', []))
        pois.extend(self.tile_cache.get('SUBMIT', []))

        # Unique POIs
        unique_pois = set(pois)
        for target in unique_pois:
            self.dist_matrices[target] = self._compute_distance_matrix(target)
            
        self.initialized = True
        print(f"[TRUE_ULTIMATE] Initialized. {len(self.walkable)} walkable nodes.")

    def _compute_distance_matrix(self, target: Tuple[int, int]) -> np.ndarray:
        """BFS from Target to All Nodes (Reverse Dijkstra)"""
        dist = np.full((self.width, self.height), 9999.0)
        queue = deque([(target, 0)])
        visited = {target}
        dist[target] = 0
        
        # Start from adjacent walkable tiles (since we can't walk ON the target usually)
        # Actually, we compute distance TO the adjacent tiles of the target?
        # No, simpler: Compute distance from ALL tiles to the Target's interaction spots.
        # But 'target' is a Coordinate.
        # Our pathfinder targets 'Adjacent to Target'.
        # So H(n) should be 'Dist to Target'.
        # BFS on walkable tiles from Target's neighbors.
        
        # Reset queue
        queue = deque()
        visited = set()
        
        tx, ty = target
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = tx + dx, ty + dy
                if (nx, ny) in self.walkable:
                    dist[nx, ny] = 1 # Distance 1 to interact
                    queue.append(((nx, ny), 1))
                    visited.add((nx, ny))
                    
        while queue:
            (cx, cy), d = queue.popleft()
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if (nx, ny) in self.walkable and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            dist[nx, ny] = d + 1
                            queue.append(((nx, ny), d + 1))
        return dist

    def play_turn(self, controller: RobotController):
        try:
            self.initialize(controller)
            team = controller.get_team()
            bot_ids = controller.get_team_bot_ids(team)
            if not bot_ids: return

            self.reserved_counters.clear()
            self.reserved_nodes.clear() # Reset traffic reservations
            
            # Roles: Bot 0 = Chef, Others = Support
            for i, bot_id in enumerate(bot_ids):
                # Stuck detection: If in same state > 60 turns, force reset
                old_state = self.bot_states.get(bot_id, 0)
                if i == 0:
                    self._run_chef(controller, bot_id)
                else:
                    self._run_support(controller, bot_id)
                
                new_state = self.bot_states.get(bot_id, 0)
                if old_state == new_state and new_state != 0:
                    self.state_turns[bot_id] += 1
                else:
                    self.state_turns[bot_id] = 0
                
                if self.state_turns[bot_id] > 60:
                    print(f"[TRUE_ULTIMATE] Bot {bot_id} STUCK in state {new_state} for 60 turns. Resetting.")
                    self.bot_states[bot_id] = 0
                    self.state_turns[bot_id] = 0
                    
        except Exception as e:
            print(f"[TRUE_ULTIMATE] Error: {e}")

    # ============================================
    # PATHFINDING (A* + Reservations)
    # ============================================
    def _navigate(self, controller: RobotController, bot_id: int, target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Returns the next (dx, dy) step to reach target, respecting reservations."""
        bot = controller.get_bot_state(bot_id)
        if not bot: return None
        start = (bot['x'], bot['y'])
        
        if self._is_adjacent(start, target):
            return None # Arrived

        # Obstacles (Other bots block step 0)
        other_bots = set()
        for bid in controller.get_team_bot_ids(controller.get_team()):
            if bid != bot_id:
                b = controller.get_bot_state(bid)
                if b: other_bots.add((b['x'], b['y']))
            
        # Heuristic
        h_matrix = self.dist_matrices.get(target)
        def heuristic(pos):
            if h_matrix is not None:
                return h_matrix[pos]
            return max(abs(pos[0]-target[0]), abs(pos[1]-target[1])) # Fallback
            
        # A* Search
        # Node: (f_score, g_score, x, y, path)
        open_set = []
        heapq.heappush(open_set, (heuristic(start), 0, start[0], start[1], []))
        visited = {(start[0], start[1], 0)} # x, y, time_offset/turn_offset
        
        current_turn = controller.get_turn()
        
        # Depth Limit (50 is enough for map2)
        while open_set:
            f, g, cx, cy, path = heapq.heappop(open_set)
            
            # Use 'path' length as time offset
            time_offset = len(path)
            
            # Check adjacency to target
            if self._is_adjacent((cx, cy), target):
                if not path: return None
                return path[0]
            
            if len(path) > 50: continue
            
            # Expand
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx, ny = cx + dx, cy + dy
                    
                    if (nx, ny) in self.walkable:
                        # Check Reservation
                        future_turn = current_turn + time_offset + 1
                        if (nx, ny, future_turn) in self.reserved_nodes:
                            continue

                        # Check Dynamic Obstacles (Immediate collisions)
                        if time_offset == 0 and (nx, ny) in other_bots:
                            continue
                            
                        state = (nx, ny, time_offset + 1)
                        if state not in visited:
                            visited.add(state)
                            new_path = path + [(dx, dy)]
                            new_g = g + 1
                            new_f = new_g + heuristic((nx, ny))
                            heapq.heappush(open_set, (new_f, new_g, nx, ny, new_path))
                            
        return None # No path found

    def _move_toward(self, controller: RobotController, bot_id: int, target: Tuple[int, int]) -> bool:
        """Helper to Execute Navigation Step"""
        step = self._navigate(controller, bot_id, target)
        if step:
            # Reserve the node we are moving TO
            bot = controller.get_bot_state(bot_id)
            bx, by = bot['x'], bot['y']
            nx, ny = bx + step[0], by + step[1]
            turn = controller.get_turn()
            self.reserved_nodes.add((nx, ny, turn + 1))
            
            controller.move(bot_id, step[0], step[1])
            return False # Moved
        else:
            # Check if adjacent
            bot = controller.get_bot_state(bot_id)
            if self._is_adjacent((bot['x'], bot['y']), target):
                return True # Arrived
            return False # Stuck

    def _is_adjacent(self, p1, p2):
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])) <= 1

    # ============================================
    # CHEF LOGIC (State Machine)
    # ============================================
    def _run_chef(self, controller: RobotController, bot_id: int):
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        state = self.bot_states.get(bot_id, 0)
        holding = bot.get('holding')
        
        # State 0: Check Pan Availability
        if state == 0:
            assigned_cooker = self.bot_cookers.get(bot_id)
            if not assigned_cooker:
                assigned_cooker = self._find_cooker_with_pan(controller)
                if assigned_cooker:
                    self.bot_cookers[bot_id] = assigned_cooker
                    self.bot_states[bot_id] = 2 # Have pan, get meat
                else:
                    self.bot_states[bot_id] = 1 # Need pan
            else:
                 self.bot_states[bot_id] = 2 # Have pan

        # State 1: Buy Pan
        elif state == 1:
            if holding:
                if holding.get('type') == 'Pan':
                     # Find cooker to place
                     target = self._get_tile_pos('COOKER', (bot['x'], bot['y']))
                     if target and self._move_toward(controller, bot_id, target):
                         if controller.place(bot_id, target[0], target[1]):
                             self.bot_cookers[bot_id] = target
                             self.bot_states[bot_id] = 2
                else:
                    # Holding something else? Trash it.
                    target = self._get_tile_pos('TRASH', (bot['x'], bot['y']))
                    if target and self._move_toward(controller, bot_id, target):
                        controller.trash(bot_id, target[0], target[1])
            else:
                 # Buy Pan
                target = self._get_tile_pos('SHOP', (bot['x'], bot['y']))
                if target and self._move_toward(controller, bot_id, target):
                    controller.buy(bot_id, ShopCosts.PAN, target[0], target[1])

        # State 2: Buy Ingredients (Meat/Noodles)
        elif state == 2:
            if holding:
                self.bot_states[bot_id] = 3
            else:
                target = self._get_tile_pos('SHOP', (bot['x'], bot['y']))
                if target and self._move_toward(controller, bot_id, target):
                    # Hardcoded Meat + Noodles Strategy (Extendable later)
                    # Check what order we want? For now, Meat -> Noodles
                    # Default to Meat for stability
                    controller.buy(bot_id, FoodType.MEAT, target[0], target[1])

        # State 3: Place Ingredient on Counter
        elif state == 3:
            counter = self.bot_counters.get(bot_id)
            if not counter or not self._is_counter_free(controller, counter):
                counter = self._find_free_counter(controller, (bot['x'], bot['y']))
                self.bot_counters[bot_id] = counter
                
            if counter:
                self.reserved_counters.add(counter) # Claim for this turn
                if self._move_toward(controller, bot_id, counter):
                    if controller.place(bot_id, counter[0], counter[1]):
                        self.bot_states[bot_id] = 4
            else:
                # No counter? Wait.
                pass

        # State 4: Chop
        elif state == 4:
            counter = self.bot_counters.get(bot_id)
            if counter:
                self.reserved_counters.add(counter)
                if self._move_toward(controller, bot_id, counter):
                    # Check progress
                    tile = controller.get_tile(controller.get_team(), counter[0], counter[1])
                    item = getattr(tile, 'item', None)
                    if not item:
                         print(f"[TRUE_ULTIMATE] Bot {bot_id} item missing from counter {counter} in State 4. Resetting.")
                         self.bot_states[bot_id] = 0
                    elif hasattr(item, 'chopped') and item.chopped:
                         # Pick up
                         if controller.pickup(bot_id, counter[0], counter[1]):
                             self.bot_states[bot_id] = 5
                    else:
                         controller.chop(bot_id, counter[0], counter[1])
            else:
                self.bot_states[bot_id] = 0

        # State 5: Cook (Place in Pan)
        elif state == 5:
            cooker = self.bot_cookers.get(bot_id)
            if not cooker:
                self.bot_states[bot_id] = 0
            else:
                if self._move_toward(controller, bot_id, cooker):
                     if controller.place(bot_id, cooker[0], cooker[1]):
                         self.bot_states[bot_id] = 6

        # State 6: Buy Noodles (Second Ingredient)
        elif state == 6:
            if holding:
                 self.bot_states[bot_id] = 7
            else:
                target = self._get_tile_pos('SHOP', (bot['x'], bot['y']))
                if target and self._move_toward(controller, bot_id, target):
                    controller.buy(bot_id, FoodType.NOODLES, target[0], target[1])

        # State 7: Place Noodles on Counter
        elif state == 7:
            # Note: We can reuse the same counter if we cleared it
            counter = self.bot_counters.get(bot_id)
            if not counter or not self._is_counter_free(controller, counter):
                counter = self._find_free_counter(controller, (bot['x'], bot['y']))
                self.bot_counters[bot_id] = counter
            
            if counter:
                 self.reserved_counters.add(counter)
                 if self._move_toward(controller, bot_id, counter):
                     if controller.place(bot_id, counter[0], counter[1]):
                         self.bot_states[bot_id] = 8

        # State 8: Chop Noodles (Wait, Noodles usually don't need chop? Assuming recipe needs it)
        # Actually in this game, Noodles don't need chop. We can skip to Cook.
        # But 'PipelineChefBot' had chop states. Let's assume we just pickup.
        # Wait, if we placed it, we need to pick it up.
        elif state == 8:
             counter = self.bot_counters.get(bot_id)
             if self._move_toward(controller, bot_id, counter):
                 # Item-presence check
                 tile = controller.get_tile(controller.get_team(), counter[0], counter[1])
                 if not getattr(tile, 'item', None):
                      print(f"[TRUE_ULTIMATE] Bot {bot_id} item missing from counter {counter} in State 8. Resetting.")
                      self.bot_states[bot_id] = 0
                 else:
                      if controller.pickup(bot_id, counter[0], counter[1]):
                          self.bot_states[bot_id] = 9

        # State 9: Add Noodles to Pan
        elif state == 9:
            cooker = self.bot_cookers.get(bot_id)
            if self._move_toward(controller, bot_id, cooker):
                 # Add ingredient
                 controller.place(bot_id, cooker[0], cooker[1]) # 'place' adds to pan
                 self.bot_states[bot_id] = 10

        # State 10: Prepare Plate (Get & Place on Counter)
        elif state == 10:
             if holding:
                 if holding.get('type') == 'Plate':
                      # Place on counter to free hands
                      counter = self.bot_counters.get(bot_id)
                      if not counter or not self._is_counter_free(controller, counter):
                           counter = self._find_free_counter(controller, (bot['x'], bot['y']))
                           self.bot_counters[bot_id] = counter
                      
                      if counter:
                           self.reserved_counters.add(counter)
                           if self._move_toward(controller, bot_id, counter):
                                if controller.place(bot_id, counter[0], counter[1]):
                                     self.bot_states[bot_id] = 12
                 else:
                     # Holding junk? Trash it
                     target = self._get_tile_pos('TRASH', (bot['x'], bot['y']))
                     if target and self._move_toward(controller, bot_id, target):
                         controller.trash(bot_id, target[0], target[1])
             else:
                  # Get Plate normally
                  # Check piles first
                  target = self._get_tile_pos('SINKTABLE', (bot['x'], bot['y']))
                  found_plate = False
                  for sx, sy in self.tile_cache.get('SINKTABLE', []):
                      t = controller.get_tile(controller.get_team(), sx, sy)
                      if getattr(t, 'num_clean_plates', 0) > 0:
                          target = (sx, sy)
                          found_plate = True
                          break
                  if found_plate:
                      if self._move_toward(controller, bot_id, target):
                          controller.take_clean_plate(bot_id, target[0], target[1])
                  else:
                      # Buy plate
                      target = self._get_tile_pos('SHOP', (bot['x'], bot['y']))
                      if target and self._move_toward(controller, bot_id, target):
                          controller.buy(bot_id, ShopCosts.PLATE, target[0], target[1])

        # State 12: Wait for Cook & Take Food (Empty Hands)
        elif state == 12:
             if holding:
                  # Error state: We shouldn't hold anything here
                  self.bot_states[bot_id] = 10
             else:
                 cooker = self.bot_cookers.get(bot_id)
                 if not cooker:
                     self.bot_states[bot_id] = 0
                 else:
                     if self._move_toward(controller, bot_id, cooker):
                         # Check if cooked
                         tile = controller.get_tile(controller.get_team(), cooker[0], cooker[1])
                         pan = getattr(tile, 'item', None)
                         if not pan or not hasattr(pan, 'food'):
                              print(f"[TRUE_ULTIMATE] Bot {bot_id} pan/food missing from {cooker} in State 12. Resetting.")
                              self.bot_states[bot_id] = 0
                         elif pan.food and hasattr(pan.food, 'cooked_stage') and pan.food.cooked_stage == 1: # Cooked stage is 1
                             if controller.take_from_pan(bot_id, cooker[0], cooker[1]):
                                 self.bot_states[bot_id] = 13

        # State 13: Add Food to Plate & Pickup
        elif state == 13:
             counter = self.bot_counters.get(bot_id)
             if not counter:
                  self.bot_states[bot_id] = 10
             else:
                  if self._move_toward(controller, bot_id, counter):
                       # Verify plate still exists
                       tile = controller.get_tile(controller.get_team(), counter[0], counter[1])
                       if not isinstance(getattr(tile, 'item', None), Plate):
                            print(f"[TRUE_ULTIMATE] Bot {bot_id} plate missing from counter {counter} in State 13. Resetting.")
                            self.bot_states[bot_id] = 0
                       else:
                            # We are holding food. Target has Plate.
                            if controller.add_food_to_plate(bot_id, counter[0], counter[1]):
                                 self.bot_states[bot_id] = 14

        # State 14: Pickup Plated Food
        elif state == 14:
             counter = self.bot_counters.get(bot_id)
             if not counter:
                 # Lost plate?
                 self.bot_states[bot_id] = 10
             else:
                 if holding and holding.get('type') == 'Plate':
                     # Already holding it (maybe pickup worked somehow?)
                     self.bot_states[bot_id] = 15
                 else:
                     if self._move_toward(controller, bot_id, counter):
                         if controller.pickup(bot_id, counter[0], counter[1]):
                             self.bot_states[bot_id] = 15

        # State 15: Submit
        elif state == 15:
             target = self._get_tile_pos('SUBMIT', (bot['x'], bot['y']))
             if target and self._move_toward(controller, bot_id, target):
                 # Valid order check
                 if holding and holding.get('type') == 'Plate':
                     plate_foods = sorted([f.get('food_name') for f in holding.get('food', [])])
                     orders = controller.get_orders(controller.get_team())
                     any_match = False
                     for o in orders:
                         if not o.get('is_active'): continue
                         req_foods = sorted(o.get('required', []))
                         if plate_foods == req_foods:
                             any_match = True
                             break
                     
                     if any_match:
                         if controller.submit(bot_id, target[0], target[1]):
                             self.bot_states[bot_id] = 0
                     else:
                         # Invalid plate! Go trash it.
                         print(f"[TRUE_ULTIMATE] Bot {bot_id} invalid plate {plate_foods}. Trashing.")
                         self.bot_states[bot_id] = 16
                 else:
                     self.bot_states[bot_id] = 0
                     
        # State 16: Trash (Error Recovery)
        elif state == 16:
             target = self._get_tile_pos('TRASH', (bot['x'], bot['y']))
             if target and self._move_toward(controller, bot_id, target):
                 controller.trash(bot_id, target[0], target[1])
                 self.bot_states[bot_id] = 0

    # ============================================
    # SUPPORT LOGIC
    # ============================================
    def _run_support(self, controller: RobotController, bot_id: int):
        """Support moves unpredictably but reserves nodes to avoid crashes."""
        bot = controller.get_bot_state(bot_id)
        if not bot: return
        
        # 1. Wash Dishes if needed
        for sx, sy in self.tile_cache.get('SINK', []):
            tile = controller.get_tile(controller.get_team(), sx, sy)
            if getattr(tile, 'num_dirty_plates', 0) > 0:
                r_count = len(self.reserved_nodes)
                if self._move_toward(controller, bot_id, (sx, sy)):
                    controller.wash_sink(bot_id, sx, sy)
                    return
                if len(self.reserved_nodes) > r_count:
                    return # We moved towards sink

        # 2. Else Random Move (Patrol)
        # We pick a random walkable neighbor
        bx, by = bot['x'], bot['y']
        valid_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = bx + dx, by + dy
                if (nx, ny) in self.walkable:
                     valid_moves.append((nx, ny))
        
        if valid_moves:
            target = random.choice(valid_moves)
            # Just try to move there
            self._move_toward(controller, bot_id, target)

    # ============================================
    # HELPERS
    # ============================================
    def _get_tile_pos(self, name: str, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        candidates = self.tile_cache.get(name, [])
        if not candidates: return None
        # Use dist matrix for accurate nearest
        # But we need DISTANCE from current_pos TO candidates
        # dist_matrices are stored by TARGET.
        # So we check self.dist_matrices[c][current_pos]
        best = None
        min_dist = 9999.0
        
        for c in candidates:
             if c in self.dist_matrices:
                 d = self.dist_matrices[c][current_pos]
                 if d < min_dist:
                     min_dist = d
                     best = c
        if best: return best
        # Fallback
        return min(candidates, key=lambda p: max(abs(p[0]-current_pos[0]), abs(p[1]-current_pos[1])))

    def _find_cooker_with_pan(self, controller) -> Optional[Tuple[int, int]]:
        team = controller.get_team()
        candidates = []
        for loc in self.tile_cache.get('COOKER', []):
            tile = controller.get_tile(team, loc[0], loc[1])
            if tile and isinstance(getattr(tile, 'item', None), Pan):
                candidates.append(loc)
        return candidates[0] if candidates else None

    def _find_free_counter(self, controller, pos) -> Optional[Tuple[int, int]]:
        candidates = []
        team = controller.get_team()
        for loc in self.tile_cache.get('COUNTER', []): # Box included
            # Check availability
            if loc in self.reserved_counters: continue
            
            tile = controller.get_tile(team, loc[0], loc[1])
            if not getattr(tile, 'item', None):
                candidates.append(loc)
        
        if not candidates: return None
        return min(candidates, key=lambda p: max(abs(p[0]-pos[0]), abs(p[1]-pos[1])))

    def _is_counter_free(self, controller, loc):
        if loc in self.reserved_counters: return False
        tile = controller.get_tile(controller.get_team(), loc[0], loc[1])
        return not getattr(tile, 'item', None)
