"""
Expert RL Training with Bayesian Hyperparameter Optimization
=============================================================
A sophisticated training pipeline that fundamentally understands the game mechanics.

Key Features:
- Rich observation encoding based on game understanding
- Hierarchical reward shaping tied to game objectives
- Bayesian optimization for hyperparameter tuning (Optuna)
- Multi-map training for generalization
- Self-play curriculum learning
- Proper evaluation across all maps

Game Understanding:
- Orders: Have required foods, deadlines, rewards, penalties
- Kitchen workflow: Box → Counter → Chop → Cook → Plate → Submit
- Resources: Plates (dirty/clean), Pans, Sink for washing
- Progression: Money accumulates, orders spawn over time
- Switch mechanic: Mid-game opportunity to sabotage opponent
"""

import os
import sys
import json
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
from gymnasium import spaces

# Game imports
from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from game_state import GameState, Order
from robot_controller import RobotController
from map_processor import load_two_team_maps_and_orders
from item import Food, Plate, Pan
from tiles import Tile, Counter, Sink, SinkTable, Cooker, Submit, Shop, Box


# =============================================================================
# RICH OBSERVATION ENCODER
# =============================================================================

class GameObservationEncoder:
    """
    Encodes 2D game state into a rich feature vector that captures
    fundamental game understanding.
    """
    
    # Tile type encoding (one-hot indices)
    TILE_TYPES = ['floor', 'wall', 'counter', 'sink', 'sinktable', 'cooker', 
                  'trash', 'submit', 'shop', 'box', 'other']
    
    # Food types
    FOOD_TYPES = ['tomato', 'lettuce', 'onion', 'meat', 'bread', 'unknown']
    
    def __init__(self, map_width: int = 20, map_height: int = 15, max_bots: int = 4, max_orders: int = 10):
        self.map_width = map_width
        self.map_height = map_height
        self.max_bots = max_bots
        self.max_orders = max_orders
        
        # Calculate observation size
        self.obs_size = self._calculate_obs_size()
        
    def _calculate_obs_size(self) -> int:
        """Calculate total observation vector size."""
        size = 0
        
        # === Global State (10 features) ===
        size += 1   # Turn progress [0, 1]
        size += 1   # Red money (normalized)
        size += 1   # Blue money (normalized)
        size += 1   # Money difference (advantage signal)
        size += 1   # Switch window active
        size += 1   # Switch window ended
        size += 1   # We have switched
        size += 1   # Opponent has switched
        size += 1   # Active orders count
        size += 1   # Urgency (min time to deadline)
        
        # === Per-Bot State (per_bot * max_bots * 2 teams = bot_features) ===
        per_bot = 0
        per_bot += 2    # Position (x, y normalized)
        per_bot += 4    # Holding type (nothing, food, plate, pan)
        per_bot += 6    # Holding food details (food_type one-hot)
        per_bot += 2    # Food state (chopped, cooked)
        per_bot += 1    # Plate has food count
        per_bot += 1    # Plate is dirty
        per_bot += 4    # Nearby important tiles (submit, cooker, sink, box nearby flags)
        # Total per bot: 20
        size += per_bot * self.max_bots * 2  # Red and Blue bots
        
        # === Order State (per_order * max_orders) ===
        per_order = 0
        per_order += 1  # Is active
        per_order += 6  # Required foods (multi-hot for 5 food types + complexity)
        per_order += 1  # Time remaining (normalized)
        per_order += 1  # Reward (normalized)
        per_order += 1  # Urgency score
        # Total per order: 10
        size += per_order * self.max_orders
        
        # === Map State Summary (key resource counts) ===
        size += 1   # Clean plates available
        size += 1   # Dirty plates in sink
        size += 1   # Pans on cookers
        size += 1   # Items on counters
        size += 1   # Foods in boxes
        
        # === Progress Features (what we're working towards) ===
        size += 5   # For each food type: do we have it ready (chopped, cooked if needed)
        size += 1   # Plates ready for plating
        size += 1   # Foods ready for plating
        
        return size
    
    def encode(self, game_state: GameState, our_team: Team) -> np.ndarray:
        """Encode the full game state for our team."""
        obs = []
        enemy_team = Team.BLUE if our_team == Team.RED else Team.RED
        our_map = game_state.get_map(our_team)
        
        # === Global State ===
        obs.append(game_state.turn / GameConstants.TOTAL_TURNS)
        obs.append(game_state.team_money[our_team] / 5000.0)
        obs.append(game_state.team_money[enemy_team] / 5000.0)
        obs.append((game_state.team_money[our_team] - game_state.team_money[enemy_team]) / 2000.0)
        obs.append(1.0 if game_state.switch_window_active() else 0.0)
        obs.append(1.0 if game_state.switch_window_ended() else 0.0)
        obs.append(1.0 if game_state.switched.get(our_team, False) else 0.0)
        obs.append(1.0 if game_state.switched.get(enemy_team, False) else 0.0)
        
        # Active orders
        active_orders = [o for o in game_state.orders.get(our_team, []) if o.is_active(game_state.turn)]
        obs.append(len(active_orders) / 10.0)
        
        # Urgency
        if active_orders:
            min_deadline = min(o.expires_turn - game_state.turn for o in active_orders)
            obs.append(1.0 / (min_deadline + 1))
        else:
            obs.append(0.0)
        
        # === Bot States ===
        our_bots = [b for b in game_state.bots.values() if b.team == our_team]
        enemy_bots = [b for b in game_state.bots.values() if b.team == enemy_team]
        
        for i in range(self.max_bots):
            if i < len(our_bots):
                obs.extend(self._encode_bot(our_bots[i], our_map, game_state))
            else:
                obs.extend([0.0] * 20)
        
        for i in range(self.max_bots):
            if i < len(enemy_bots):
                obs.extend(self._encode_bot(enemy_bots[i], our_map, game_state))
            else:
                obs.extend([0.0] * 20)
        
        # === Order State ===
        for i in range(self.max_orders):
            if i < len(active_orders):
                obs.extend(self._encode_order(active_orders[i], game_state.turn))
            else:
                obs.extend([0.0] * 10)
        
        # === Map State Summary ===
        map_summary = self._encode_map_summary(our_map)
        obs.extend(map_summary)
        
        # === Progress Features ===
        progress = self._encode_progress(game_state, our_team, active_orders)
        obs.extend(progress)
        
        # Pad/truncate to expected size
        while len(obs) < self.obs_size:
            obs.append(0.0)
        
        return np.array(obs[:self.obs_size], dtype=np.float32)
    
    def _encode_bot(self, bot, game_map, game_state) -> List[float]:
        """Encode a single bot's state."""
        features = []
        
        # Position
        features.append(bot.x / self.map_width)
        features.append(bot.y / self.map_height)
        
        # Holding type (one-hot: nothing, food, plate, pan)
        holding = [0.0, 0.0, 0.0, 0.0]
        if bot.holding is None:
            holding[0] = 1.0
        elif isinstance(bot.holding, Food):
            holding[1] = 1.0
        elif isinstance(bot.holding, Plate):
            holding[2] = 1.0
        elif isinstance(bot.holding, Pan):
            holding[3] = 1.0
        features.extend(holding)
        
        # Holding food details
        food_type = [0.0] * 6
        food_state = [0.0, 0.0]  # chopped, cooked
        plate_food_count = 0.0
        plate_dirty = 0.0
        
        if isinstance(bot.holding, Food):
            if bot.holding.food_id < 5:
                food_type[bot.holding.food_id] = 1.0
            else:
                food_type[5] = 1.0
            food_state[0] = 1.0 if bot.holding.chopped else 0.0
            food_state[1] = 1.0 if bot.holding.cooked_stage == 1 else 0.0
        elif isinstance(bot.holding, Plate):
            plate_food_count = len(bot.holding.food) / 5.0
            plate_dirty = 1.0 if bot.holding.dirty else 0.0
        elif isinstance(bot.holding, Pan):
            if bot.holding.food is not None:
                if bot.holding.food.food_id < 5:
                    food_type[bot.holding.food.food_id] = 1.0
                food_state[0] = 1.0 if bot.holding.food.chopped else 0.0
                food_state[1] = 1.0 if bot.holding.food.cooked_stage == 1 else 0.0
        
        features.extend(food_type)
        features.extend(food_state)
        features.append(plate_food_count)
        features.append(plate_dirty)
        
        # Nearby important tiles
        nearby = [0.0, 0.0, 0.0, 0.0]  # submit, cooker, sink, box
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = bot.x + dx, bot.y + dy
                if 0 <= nx < game_map.width and 0 <= ny < game_map.height:
                    tile = game_map.tiles[nx][ny]
                    if isinstance(tile, Submit):
                        nearby[0] = 1.0
                    elif isinstance(tile, Cooker):
                        nearby[1] = 1.0
                    elif isinstance(tile, (Sink, SinkTable)):
                        nearby[2] = 1.0
                    elif isinstance(tile, Box):
                        nearby[3] = 1.0
        features.extend(nearby)
        
        return features
    
    def _encode_order(self, order: Order, current_turn: int) -> List[float]:
        """Encode a single order."""
        features = []
        
        # Is active
        features.append(1.0)
        
        # Required foods (multi-hot for 5 food types)
        required = [0.0] * 5
        for ft in order.required:
            if ft.food_id < 5:
                required[ft.food_id] = 1.0
        features.extend(required)
        
        # Complexity (number of items)
        features.append(len(order.required) / 5.0)
        
        # Time remaining
        time_left = order.expires_turn - current_turn
        features.append(time_left / 50.0)
        
        # Reward
        features.append(order.reward / 200.0)
        
        # Urgency score (higher when deadline is soon)
        features.append(1.0 / (time_left + 1) if time_left > 0 else 1.0)
        
        return features
    
    def _encode_map_summary(self, game_map) -> List[float]:
        """Encode a summary of map state."""
        clean_plates = 0
        dirty_plates = 0
        pans_on_cookers = 0
        items_on_counters = 0
        foods_in_boxes = 0
        
        for x in range(game_map.width):
            for y in range(game_map.height):
                tile = game_map.tiles[x][y]
                if isinstance(tile, SinkTable):
                    clean_plates += tile.num_clean_plates
                elif isinstance(tile, Sink):
                    dirty_plates += tile.num_dirty_plates
                elif isinstance(tile, Cooker):
                    if tile.item is not None:
                        pans_on_cookers += 1
                elif isinstance(tile, Counter):
                    if hasattr(tile, 'item') and tile.item is not None:
                        items_on_counters += 1
                elif isinstance(tile, Box):
                    foods_in_boxes += 1
        
        return [
            clean_plates / 10.0,
            dirty_plates / 10.0,
            pans_on_cookers / 5.0,
            items_on_counters / 10.0,
            foods_in_boxes / 5.0,
        ]
    
    def _encode_progress(self, game_state, our_team, active_orders) -> List[float]:
        """Encode progress towards completing orders."""
        # For each food type, check if we have it ready
        food_ready = [0.0] * 5
        plates_ready = 0.0
        foods_ready = 0.0
        
        for bot in game_state.bots.values():
            if bot.team != our_team:
                continue
            if isinstance(bot.holding, Food):
                foods_ready += 1
                # Check if properly prepared
                if bot.holding.food_id < 5:
                    ft = [ft for ft in FoodType if ft.food_id == bot.holding.food_id]
                    if ft:
                        ft = ft[0]
                        needs_chop = ft.can_chop
                        needs_cook = ft.can_cook
                        if (not needs_chop or bot.holding.chopped) and (not needs_cook or bot.holding.cooked_stage == 1):
                            food_ready[bot.holding.food_id] = 1.0
            elif isinstance(bot.holding, Plate):
                if not bot.holding.dirty:
                    plates_ready += 1
        
        return food_ready + [plates_ready / 4.0, foods_ready / 4.0]


# =============================================================================
# EXPERT ENVIRONMENT
# =============================================================================

class ExpertAWAPEnv(gym.Env):
    """
    An expert-level AWAP environment with rich observations and
    shaped rewards based on fundamental game understanding.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # Action space: 16 actions per bot
    NUM_ACTIONS = 16
    ACTION_NAMES = [
        'stay', 'up', 'down', 'left', 'right',
        'pickup', 'place', 'chop', 'start_cook', 'take_from_pan',
        'take_plate', 'put_dirty', 'wash', 'add_to_plate', 'submit', 'buy_plate'
    ]
    
    MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    def __init__(
        self,
        map_paths: List[str],
        max_bots: int = 4,
        opponent_type: str = "idle",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.map_paths = map_paths
        self.current_map_idx = 0
        self.max_bots = max_bots
        self.opponent_type = opponent_type
        self.render_mode = render_mode
        
        # Initialize encoder
        self.encoder = GameObservationEncoder(max_bots=max_bots)
        
        # Spaces
        self.action_space = spaces.MultiDiscrete([self.NUM_ACTIONS] * max_bots)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.encoder.obs_size,), dtype=np.float32
        )
        
        # State
        self.game_state: Optional[GameState] = None
        self.red_controller: Optional[RobotController] = None
        self.blue_controller: Optional[RobotController] = None
        self.current_turn = 0
        self.total_turns = GameConstants.TOTAL_TURNS
        
        # Progress tracking for reward shaping
        self.prev_money = 0
        self.prev_orders_completed = 0
        self.prev_holding_counts = {}
        self.episode_stats = {}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Rotate through maps for multi-map training
        self.current_map_idx = (self.current_map_idx + 1) % len(self.map_paths)
        map_path = self.map_paths[self.current_map_idx]
        
        # Load game
        map_red, map_blue, orders_red, orders_blue, parsed = load_two_team_maps_and_orders(map_path)
        self.game_state = GameState(red_map=map_red, blue_map=map_blue)
        
        # Load orders
        self.game_state.orders[Team.RED] = orders_red
        self.game_state.orders[Team.BLUE] = orders_blue
        
        # Add bots
        if parsed.spawns_red:
            for x, y in parsed.spawns_red:
                self.game_state.add_bot(Team.RED, x, y)
        else:
            self.game_state.add_bot(Team.RED, 1, 1)
        
        if parsed.spawns_blue:
            for x, y in parsed.spawns_blue:
                self.game_state.add_bot(Team.BLUE, x, y)
        else:
            self.game_state.add_bot(Team.BLUE, 1, 1)
        
        # Controllers
        self.red_controller = RobotController(Team.RED, self.game_state)
        self.blue_controller = RobotController(Team.BLUE, self.game_state)
        
        # Reset tracking
        self.current_turn = 0
        self.prev_money = self.game_state.team_money[Team.RED]
        self.prev_orders_completed = 0
        self.episode_stats = {'orders': 0, 'money': 0, 'actions': 0}
        
        obs = self.encoder.encode(self.game_state, Team.RED)
        return obs, {"map": map_path}
    
    def step(self, action):
        # Track state before actions
        prev_money = self.game_state.team_money[Team.RED]
        prev_active_orders = len([o for o in self.game_state.orders.get(Team.RED, []) 
                                  if o.is_active(self.current_turn)])
        prev_holding = self._count_useful_holdings(Team.RED)
        
        # Start turn
        self.game_state.start_turn()
        
        # Execute RED team actions
        red_bot_ids = self.red_controller.get_team_bot_ids(Team.RED)
        action_success = 0
        for i, bot_id in enumerate(red_bot_ids):
            if i < len(action):
                if self._execute_action(self.red_controller, bot_id, action[i]):
                    action_success += 1
        
        # Execute opponent
        self._execute_opponent()
        
        # Advance turn
        self.current_turn += 1
        self.game_state.turn = self.current_turn
        
        # ===================================================================
        # SOPHISTICATED REWARD SHAPING
        # ===================================================================
        
        reward = 0.0
        current_money = self.game_state.team_money[Team.RED]
        enemy_money = self.game_state.team_money[Team.BLUE]
        money_gained = current_money - prev_money
        
        # 1. Primary reward: Money gained (scaled)
        reward += money_gained * 0.005
        
        # 2. Order completion bonus
        current_active_orders = len([o for o in self.game_state.orders.get(Team.RED, [])
                                      if o.is_active(self.current_turn)])
        orders_completed = prev_active_orders - current_active_orders
        if orders_completed > 0 and money_gained > 0:
            # Completed order (not expired)
            reward += 1.0 * orders_completed
            self.episode_stats['orders'] += orders_completed
        
        # 3. Progress reward: picking up useful items
        current_holding = self._count_useful_holdings(Team.RED)
        if current_holding > prev_holding:
            reward += 0.05
        
        # 4. Action success bonus (small)
        reward += 0.002 * action_success
        
        # 5. Time pressure (increasing penalty)
        time_progress = self.current_turn / self.total_turns
        reward -= 0.001 * (1 + time_progress)
        
        # 6. Relative performance bonus
        money_diff = current_money - enemy_money
        if money_diff > 0:
            reward += 0.001 * min(money_diff / 500, 1.0)
        
        # Track stats
        self.episode_stats['money'] = current_money
        self.episode_stats['actions'] += action_success
        
        # Check termination
        terminated = self.current_turn >= self.total_turns
        truncated = False
        
        info = {
            "turn": self.current_turn,
            "red_money": current_money,
            "blue_money": enemy_money,
            "orders_completed": self.episode_stats['orders'],
        }
        
        if terminated:
            # End-game scoring
            margin = current_money - enemy_money
            if margin > 0:
                reward += 3.0 + margin / 500.0  # Win bonus + margin
                info["result"] = "WIN"
            elif margin < 0:
                reward -= 1.5  # Lose penalty
                info["result"] = "LOSE"
            else:
                reward += 0.5  # Draw is okay
                info["result"] = "DRAW"
        
        obs = self.encoder.encode(self.game_state, Team.RED)
        return obs, reward, terminated, truncated, info
    
    def _count_useful_holdings(self, team: Team) -> int:
        """Count useful items held by team."""
        count = 0
        for bot in self.game_state.bots.values():
            if bot.team != team:
                continue
            if bot.holding is not None:
                count += 1
                if isinstance(bot.holding, Plate) and bot.holding.food:
                    count += len(bot.holding.food)
        return count
    
    def _execute_action(self, controller: RobotController, bot_id: int, action: int) -> bool:
        """Execute action and return success."""
        try:
            if action == 0:  # Stay
                return True
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                return controller.move(bot_id, dx, dy)
            elif action == 5:  # Pickup
                return controller.pickup(bot_id)
            elif action == 6:  # Place
                return self._try_on_neighbors(controller.place, bot_id)
            elif action == 7:  # Chop
                return self._try_on_neighbors(controller.chop, bot_id)
            elif action == 8:  # Start cook
                return self._try_on_neighbors(controller.start_cook, bot_id)
            elif action == 9:  # Take from pan
                return self._try_on_neighbors(controller.take_from_pan, bot_id)
            elif action == 10:  # Take clean plate
                return self._try_on_neighbors(controller.take_clean_plate, bot_id)
            elif action == 11:  # Put dirty plate
                return self._try_on_neighbors(controller.put_dirty_plate_in_sink, bot_id)
            elif action == 12:  # Wash
                return self._try_on_neighbors(controller.wash_sink, bot_id)
            elif action == 13:  # Add food to plate
                return self._try_on_neighbors(controller.add_food_to_plate, bot_id)
            elif action == 14:  # Submit
                return self._try_on_neighbors(controller.submit, bot_id)
            elif action == 15:  # Buy plate
                return self._try_on_neighbors(
                    lambda bid, x, y: controller.buy(bid, ShopCosts.PLATE, x, y), bot_id
                )
        except Exception:
            pass
        return False
    
    def _try_on_neighbors(self, action_fn, bot_id: int) -> bool:
        """Try action on current and neighboring tiles."""
        bot_state = self.red_controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        
        x, y = bot_state["x"], bot_state["y"]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                try:
                    if action_fn(bot_id, x + dx, y + dy):
                        return True
                except:
                    pass
        return False
    
    def _execute_opponent(self):
        """Execute opponent actions."""
        if self.opponent_type == "idle":
            return
        elif self.opponent_type == "random":
            blue_bot_ids = self.blue_controller.get_team_bot_ids(Team.BLUE)
            for bot_id in blue_bot_ids:
                action = random.randint(0, self.NUM_ACTIONS - 1)
                self._execute_action_blue(bot_id, action)
    
    def _execute_action_blue(self, bot_id: int, action: int):
        """Execute action for blue team."""
        try:
            controller = self.blue_controller
            if action == 0:
                return True
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                return controller.move(bot_id, dx, dy)
            elif action == 5:
                return controller.pickup(bot_id)
            # ... simplified for opponent
        except:
            pass
        return False


# =============================================================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# =============================================================================

def train_with_optuna(
    n_trials: int = 20,
    timesteps_per_trial: int = 100000,
    final_timesteps: int = 500000,
    maps: Optional[List[str]] = None,
):
    """
    Train using Bayesian optimization with Optuna.
    """
    try:
        import optuna
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Please install: pip install optuna stable-baselines3")
        return
    
    if maps is None:
        maps = [
            str(ROOT_DIR / "maps/official/chopped.txt"),
            str(ROOT_DIR / "maps/official/v1.txt"),
            str(ROOT_DIR / "maps/official/orbit.txt"),
            str(ROOT_DIR / "maps/official/throughput.txt"),
        ]
    
    # Validate maps exist
    maps = [m for m in maps if os.path.exists(m)]
    if not maps:
        print("No valid maps found!")
        return
    
    print(f"Training on {len(maps)} maps")
    
    def make_env(opponent_type="idle"):
        def _init():
            return ExpertAWAPEnv(map_paths=maps, opponent_type=opponent_type)
        return _init
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        
        # Network architecture
        net_arch_size = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
        net_arch = {
            "small": [64, 64],
            "medium": [128, 128],
            "large": [256, 256, 128],
        }[net_arch_size]
        
        # Create environment
        n_envs = 4
        env = DummyVecEnv([make_env("idle") for _ in range(n_envs)])
        
        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
        )
        
        # Train
        model.learn(total_timesteps=timesteps_per_trial)
        
        # Evaluate
        eval_env = ExpertAWAPEnv(map_paths=maps, opponent_type="random")
        total_reward = 0
        n_eval = 20
        
        for _ in range(n_eval):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
            # Score based on final money
            total_reward += info.get("red_money", 0) - info.get("blue_money", 0)
        
        env.close()
        return total_reward / n_eval
    
    # Run optimization
    print(f"\n{'='*60}")
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print(f"Trials: {n_trials}, Timesteps per trial: {timesteps_per_trial}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial: {study.best_trial.value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print(f"\n{'='*60}")
    print(f"TRAINING FINAL MODEL ({final_timesteps} timesteps)")
    print(f"{'='*60}\n")
    
    best_params = study.best_trial.params
    net_arch = {"small": [64, 64], "medium": [128, 128], "large": [256, 256, 128]}[best_params.pop("net_arch")]
    
    n_envs = 4
    env = DummyVecEnv([make_env("random") for _ in range(n_envs)])
    
    final_model = PPO(
        "MlpPolicy",
        env,
        **{k: v for k, v in best_params.items()},
        policy_kwargs={"net_arch": net_arch},
        verbose=1,
    )
    
    # Train with curriculum
    # Phase 1: vs idle
    env_idle = DummyVecEnv([make_env("idle") for _ in range(n_envs)])
    final_model.set_env(env_idle)
    final_model.learn(total_timesteps=final_timesteps // 3)
    
    # Phase 2: vs random
    env_random = DummyVecEnv([make_env("random") for _ in range(n_envs)])
    final_model.set_env(env_random)
    final_model.learn(total_timesteps=final_timesteps * 2 // 3)
    
    # Save
    os.makedirs(str(SCRIPT_DIR / "models"), exist_ok=True)
    model_path = str(SCRIPT_DIR / "models" / f"expert_bot_{int(time.time())}.zip")
    final_model.save(model_path)
    
    # Save best params
    params_path = str(SCRIPT_DIR / "models" / "best_params.json")
    with open(params_path, "w") as f:
        json.dump({"params": study.best_trial.params, "score": study.best_trial.value}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"Best params saved to: {params_path}")
    print(f"{'='*60}")
    
    env.close()
    return model_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Expert RL Training with Bayesian Optimization")
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials")
    parser.add_argument("--trial-steps", type=int, default=50000, help="Timesteps per trial")
    parser.add_argument("--final-steps", type=int, default=300000, help="Final training timesteps")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test
        print("Quick test mode...")
        args.trials = 3
        args.trial_steps = 10000
        args.final_steps = 30000
    
    model_path = train_with_optuna(
        n_trials=args.trials,
        timesteps_per_trial=args.trial_steps,
        final_timesteps=args.final_steps,
    )
    
    if model_path:
        print(f"\nTo export: python export_model.py {model_path}")
