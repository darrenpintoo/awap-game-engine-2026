"""
Enhanced RL Training - Training Against Strong Opponents
==========================================================
Key improvements:
1. Train against actual strong bots (ultimate_champion, Hydra)
2. Larger 256x256 network
3. Better reward shaping with order progress tracking
4. 1M+ timesteps for deeper learning
"""

import os
import sys
import time
import random
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple

# Add paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Game imports
from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from game_state import GameState, Order
from robot_controller import RobotController
from map_processor import load_two_team_maps_and_orders
from item import Food, Plate, Pan
from tiles import Tile, Counter, Sink, SinkTable, Cooker, Submit, Shop, Box


# =============================================================================
# OPPONENT BOT WRAPPER
# =============================================================================

class OpponentBot:
    """Wrapper to run an opponent bot as a subprocess-like action generator."""
    
    def __init__(self, bot_path: str):
        self.bot_path = bot_path
        self.bot_module = None
        self._load_bot()
    
    def _load_bot(self):
        """Load the bot module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("opponent_bot", self.bot_path)
        self.bot_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.bot_module)
        except Exception as e:
            print(f"Warning: Could not load bot {self.bot_path}: {e}")
            self.bot_module = None
    
    def get_actions(self, game_state: GameState, controller: RobotController) -> None:
        """Execute the opponent bot's turn."""
        if self.bot_module is None:
            return
        
        try:
            # Call the bot's turn function
            if hasattr(self.bot_module, 'turn'):
                self.bot_module.turn(game_state, controller)
        except Exception as e:
            pass  # Silently fail - opponent errors shouldn't crash training


# =============================================================================
# ENHANCED ENVIRONMENT WITH OPPONENT TRAINING
# =============================================================================

class EnhancedAWAPEnv(gym.Env):
    """
    Enhanced AWAP environment with:
    - Training against actual opponent bots
    - Better reward shaping with order progress
    - Rich observations
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    NUM_ACTIONS = 16
    MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    def __init__(
        self,
        map_paths: List[str],
        opponent_bots: Optional[List[str]] = None,
        max_bots: int = 4,
    ):
        super().__init__()
        
        self.map_paths = map_paths
        self.opponent_bots = opponent_bots or []
        self.max_bots = max_bots
        self.current_map_idx = 0
        
        # Load opponent bots
        self.opponents = []
        for bot_path in self.opponent_bots:
            if os.path.exists(bot_path):
                self.opponents.append(OpponentBot(bot_path))
        
        # Observation: 300 features (rich encoding)
        self.obs_size = 300
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.NUM_ACTIONS] * max_bots)
        
        # State
        self.game_state = None
        self.red_controller = None
        self.blue_controller = None
        self.current_turn = 0
        self.total_turns = GameConstants.TOTAL_TURNS
        
        # Progress tracking for reward shaping
        self.prev_money = 0
        self.prev_order_progress = {}
        self.orders_completed_this_episode = 0
        
        # Current opponent
        self.current_opponent = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Rotate maps
        self.current_map_idx = (self.current_map_idx + 1) % len(self.map_paths)
        map_path = self.map_paths[self.current_map_idx]
        
        # Select random opponent (or None for random/idle)
        if self.opponents and random.random() < 0.7:  # 70% trained vs bot
            self.current_opponent = random.choice(self.opponents)
        else:
            self.current_opponent = None
        
        # Load game
        map_red, map_blue, orders_red, orders_blue, parsed = load_two_team_maps_and_orders(map_path)
        self.game_state = GameState(red_map=map_red, blue_map=map_blue)
        self.game_state.orders[Team.RED] = orders_red
        self.game_state.orders[Team.BLUE] = orders_blue
        
        # Add bots
        if parsed.spawns_red:
            for x, y in parsed.spawns_red:
                self.game_state.add_bot(Team.RED, x, y)
        else:
            self.game_state.add_bot(Team.RED, 1, 1)
            self.game_state.add_bot(Team.RED, 2, 1)
        
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
        self.prev_order_progress = {}
        self.orders_completed_this_episode = 0
        
        obs = self._get_obs()
        return obs, {"map": map_path, "opponent": type(self.current_opponent).__name__ if self.current_opponent else "random"}
    
    def step(self, action):
        # Track pre-step state
        prev_money = self.game_state.team_money[Team.RED]
        prev_active = len([o for o in self.game_state.orders.get(Team.RED, []) if o.is_active(self.current_turn)])
        pre_order_progress = self._calculate_order_progress()
        
        # Start turn
        self.game_state.start_turn()
        self.current_turn = self.game_state.turn
        
        # Execute RED actions (our RL agent)
        red_bot_ids = self.red_controller.get_team_bot_ids(Team.RED)
        action_success = 0
        for i, bot_id in enumerate(red_bot_ids):
            if i < len(action):
                if self._execute_action(self.red_controller, bot_id, action[i]):
                    action_success += 1
        
        # Execute opponent (BLUE)
        if self.current_opponent:
            self.current_opponent.get_actions(self.game_state, self.blue_controller)
        else:
            # Random opponent
            blue_bot_ids = self.blue_controller.get_team_bot_ids(Team.BLUE)
            for bot_id in blue_bot_ids:
                act = random.randint(0, self.NUM_ACTIONS - 1)
                self._execute_action(self.blue_controller, bot_id, act)
        
        # ===================================================================
        # ENHANCED REWARD SHAPING
        # ===================================================================
        
        reward = 0.0
        current_money = self.game_state.team_money[Team.RED]
        enemy_money = self.game_state.team_money[Team.BLUE]
        
        # 1. Primary: Money gained (scaled well)
        money_gained = current_money - prev_money
        reward += money_gained * 0.01  # Strong signal
        
        # 2. Order completion bonus
        current_active = len([o for o in self.game_state.orders.get(Team.RED, []) if o.is_active(self.current_turn)])
        orders_done = prev_active - current_active
        if orders_done > 0 and money_gained > 0:
            reward += 2.0 * orders_done  # Big bonus for completing orders
            self.orders_completed_this_episode += orders_done
        
        # 3. Order PROGRESS reward (partial credit!)
        post_order_progress = self._calculate_order_progress()
        progress_delta = post_order_progress - pre_order_progress
        if progress_delta > 0:
            reward += 0.3 * progress_delta  # Reward for making progress
        
        # 4. Action success (small)
        reward += 0.005 * action_success
        
        # 5. Time pressure (encourage efficiency)
        time_progress = self.current_turn / self.total_turns
        reward -= 0.002 * time_progress
        
        # 6. Competitive pressure
        if current_money > enemy_money:
            reward += 0.01 * min((current_money - enemy_money) / 200, 1.0)
        elif current_money < enemy_money:
            reward -= 0.005  # Small penalty for being behind
        
        # Check termination
        terminated = self.current_turn >= self.total_turns
        truncated = False
        
        info = {
            "turn": self.current_turn,
            "red_money": current_money,
            "blue_money": enemy_money,
            "orders_completed": self.orders_completed_this_episode,
        }
        
        if terminated:
            margin = current_money - enemy_money
            if margin > 0:
                reward += 5.0 + margin / 100.0  # Win bonus
                info["result"] = "WIN"
            elif margin < 0:
                reward -= 2.0
                info["result"] = "LOSE"
            else:
                reward += 1.0
                info["result"] = "DRAW"
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _calculate_order_progress(self) -> float:
        """Calculate progress towards completing current orders."""
        progress = 0.0
        
        active_orders = [o for o in self.game_state.orders.get(Team.RED, []) if o.is_active(self.current_turn)]
        if not active_orders:
            return 0.0
        
        # Count items we have that match order requirements
        for order in active_orders:
            required_ids = set(ft.food_id for ft in order.required)
            
            for bot in self.game_state.bots.values():
                if bot.team != Team.RED:
                    continue
                
                if isinstance(bot.holding, Food):
                    if bot.holding.food_id in required_ids:
                        progress += 0.2
                        if bot.holding.chopped:
                            progress += 0.1
                        if bot.holding.cooked_stage == 1:
                            progress += 0.1
                elif isinstance(bot.holding, Plate):
                    if not bot.holding.dirty:
                        progress += 0.1
                        for f in bot.holding.food:
                            if isinstance(f, Food) and f.food_id in required_ids:
                                progress += 0.3
        
        return progress
    
    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        obs = np.zeros(self.obs_size, dtype=np.float32)
        idx = 0
        
        # Global state (20 features)
        obs[idx] = self.current_turn / self.total_turns; idx += 1
        obs[idx] = self.game_state.team_money[Team.RED] / 3000; idx += 1
        obs[idx] = self.game_state.team_money[Team.BLUE] / 3000; idx += 1
        obs[idx] = (self.game_state.team_money[Team.RED] - self.game_state.team_money[Team.BLUE]) / 1000; idx += 1
        
        obs[idx] = 1.0 if self.game_state.switch_window_active() else 0.0; idx += 1
        obs[idx] = 1.0 if self.game_state.switch_window_ended() else 0.0; idx += 1
        
        active_orders = [o for o in self.game_state.orders.get(Team.RED, []) if o.is_active(self.current_turn)]
        obs[idx] = len(active_orders) / 10.0; idx += 1
        
        if active_orders:
            min_deadline = min(o.expires_turn - self.current_turn for o in active_orders)
            obs[idx] = 1.0 / (min_deadline + 1); idx += 1
        else:
            idx += 1
        
        # Pad to 20
        idx = 20
        
        # Per-bot state (25 features x 4 bots x 2 teams = 200)
        our_map = self.game_state.get_map(Team.RED)
        
        for team in [Team.RED, Team.BLUE]:
            bots = [b for b in self.game_state.bots.values() if b.team == team]
            for i in range(self.max_bots):
                if i < len(bots):
                    bot = bots[i]
                    obs[idx] = bot.x / 20; idx += 1
                    obs[idx] = bot.y / 15; idx += 1
                    
                    # Holding type (4)
                    if bot.holding is None:
                        obs[idx] = 1.0
                    idx += 1
                    if isinstance(bot.holding, Food):
                        obs[idx] = 1.0
                    idx += 1
                    if isinstance(bot.holding, Plate):
                        obs[idx] = 1.0
                    idx += 1
                    if isinstance(bot.holding, Pan):
                        obs[idx] = 1.0
                    idx += 1
                    
                    # Food details (10)
                    if isinstance(bot.holding, Food):
                        if bot.holding.food_id < 5:
                            obs[idx + bot.holding.food_id] = 1.0
                        obs[idx + 5] = 1.0 if bot.holding.chopped else 0.0
                        obs[idx + 6] = 1.0 if bot.holding.cooked_stage == 1 else 0.0
                    idx += 10
                    
                    # Plate state (4)
                    if isinstance(bot.holding, Plate):
                        obs[idx] = 1.0 if bot.holding.dirty else 0.0
                        obs[idx + 1] = len(bot.holding.food) / 5.0
                    idx += 4
                    
                    # Nearby features (5)
                    idx += 5
                else:
                    idx += 25
        
        # Orders (10 features x 5 orders = 50)
        for i in range(5):
            if i < len(active_orders):
                o = active_orders[i]
                obs[idx] = 1.0; idx += 1
                obs[idx] = len(o.required) / 5.0; idx += 1
                obs[idx] = (o.expires_turn - self.current_turn) / 50.0; idx += 1
                obs[idx] = o.reward / 100.0; idx += 1
                
                # Required foods
                for ft in o.required[:5]:
                    if ft.food_id < 5:
                        obs[idx + ft.food_id] = 1.0
                idx += 6
            else:
                idx += 10
        
        return obs
    
    def _execute_action(self, controller: RobotController, bot_id: int, action: int) -> bool:
        """Execute action."""
        try:
            if action == 0:
                return True
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                return controller.move(bot_id, dx, dy)
            elif action == 5:
                return controller.pickup(bot_id)
            elif action in range(6, 16):
                return self._try_neighbors(controller, bot_id, action)
        except:
            pass
        return False
    
    def _try_neighbors(self, controller, bot_id, action) -> bool:
        """Try action on neighbors."""
        bot_state = controller.get_bot_state(bot_id)
        if not bot_state:
            return False
        x, y = bot_state["x"], bot_state["y"]
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                try:
                    if action == 6:
                        if controller.place(bot_id, x+dx, y+dy): return True
                    elif action == 7:
                        if controller.chop(bot_id, x+dx, y+dy): return True
                    elif action == 8:
                        if controller.start_cook(bot_id, x+dx, y+dy): return True
                    elif action == 9:
                        if controller.take_from_pan(bot_id, x+dx, y+dy): return True
                    elif action == 10:
                        if controller.take_clean_plate(bot_id, x+dx, y+dy): return True
                    elif action == 11:
                        if controller.put_dirty_plate_in_sink(bot_id, x+dx, y+dy): return True
                    elif action == 12:
                        if controller.wash_sink(bot_id, x+dx, y+dy): return True
                    elif action == 13:
                        if controller.add_food_to_plate(bot_id, x+dx, y+dy): return True
                    elif action == 14:
                        if controller.submit(bot_id, x+dx, y+dy): return True
                    elif action == 15:
                        if controller.buy(bot_id, ShopCosts.PLATE, x+dx, y+dy): return True
                except:
                    pass
        return False


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

class TrainingCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.best_reward = -float('inf')
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}_{self.n_calls}_steps.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[{self.n_calls}] Checkpoint saved: {path}")
        return True


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ENHANCED RL TRAINING - ADVERSARIAL MODE")
    print("="*60)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel Envs: {args.envs}")
    print("="*60 + "\n")
    
    # Maps for training
    maps = [
        str(ROOT_DIR / "maps/official/chopped.txt"),
        str(ROOT_DIR / "maps/official/v1.txt"),
        str(ROOT_DIR / "maps/official/orbit.txt"),
        str(ROOT_DIR / "maps/official/throughput.txt"),
    ]
    maps = [m for m in maps if os.path.exists(m)]
    
    # Strong opponent bots
    opponents = [
        str(ROOT_DIR / "bots/dpinto/ultimate_champion_bot.py"),
        str(ROOT_DIR / "bots/hareshm/optimal_bot.py"),
        str(ROOT_DIR / "bots/eric/iron_chef_bot.py"),
    ]
    opponents = [o for o in opponents if os.path.exists(o)]
    
    print(f"Maps: {len(maps)}")
    print(f"Opponents: {len(opponents)}")
    for o in opponents:
        print(f"  - {Path(o).name}")
    print()
    
    # Create environments
    def make_env():
        def _init():
            return EnhancedAWAPEnv(map_paths=maps, opponent_bots=opponents)
        return _init
    
    env = DummyVecEnv([make_env() for _ in range(args.envs)])
    
    # Create model with larger network
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"net_arch": [256, 256, 128]},  # Bigger network!
        verbose=1,
    )
    
    # Save path
    os.makedirs(str(SCRIPT_DIR / "models"), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = str(SCRIPT_DIR / f"models/enhanced_{timestamp}")
    
    # Callback
    callback = TrainingCallback(save_path, save_freq=100000)
    
    # Train!
    print("Starting training...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    
    # Save final
    final_path = f"{save_path}_final.zip"
    model.save(final_path)
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final model: {final_path}")
    print("="*60)
    
    env.close()
    return final_path


if __name__ == "__main__":
    main()
