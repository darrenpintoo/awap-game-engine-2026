#!/usr/bin/env python3
"""
Fast RL Training for Colab/Local
=================================
Optimized for speed by:
1. Suppressing RC WARN print statements
2. Using efficient settings
3. Better batch processing

Usage:
  python train_fast.py --timesteps 500000 --envs 4
"""

import os
import sys
import time
import random
import io
from pathlib import Path
from typing import List, Optional
from contextlib import redirect_stdout

# Add paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Suppress RC warnings by patching print in robot_controller
original_print = print
def quiet_print(*args, **kwargs):
    msg = str(args[0]) if args else ""
    if "RC for" not in msg and "WARN" not in msg:
        original_print(*args, **kwargs)

# Apply patch
import builtins
builtins.print = quiet_print

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Game imports
from game_constants import Team, TileType, FoodType, ShopCosts, GameConstants
from game_state import GameState
from robot_controller import RobotController
from map_processor import load_two_team_maps_and_orders
from item import Food, Plate, Pan


class FastAWAPEnv(gym.Env):
    """Fast AWAP environment for training."""
    
    metadata = {"render_modes": []}
    NUM_ACTIONS = 16
    MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    def __init__(self, map_paths: List[str], max_bots: int = 4):
        super().__init__()
        self.map_paths = map_paths
        self.max_bots = max_bots
        self.current_map_idx = 0
        
        # Observation: compact encoding
        self.obs_size = 200
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([self.NUM_ACTIONS] * max_bots)
        
        self.game_state = None
        self.red_controller = None
        self.blue_controller = None
        self.total_turns = GameConstants.TOTAL_TURNS
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_map_idx = (self.current_map_idx + 1) % len(self.map_paths)
        map_path = self.map_paths[self.current_map_idx]
        
        map_red, map_blue, orders_red, orders_blue, parsed = load_two_team_maps_and_orders(map_path)
        self.game_state = GameState(red_map=map_red, blue_map=map_blue)
        self.game_state.orders[Team.RED] = orders_red
        self.game_state.orders[Team.BLUE] = orders_blue
        
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
        
        self.red_controller = RobotController(Team.RED, self.game_state)
        self.blue_controller = RobotController(Team.BLUE, self.game_state)
        self.current_turn = 0
        self.prev_money = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        prev_money = self.game_state.team_money[Team.RED]
        
        self.game_state.start_turn()
        self.current_turn = self.game_state.turn
        
        # Execute actions (silently - no print warnings)
        red_bots = self.red_controller.get_team_bot_ids(Team.RED)
        for i, bot_id in enumerate(red_bots):
            if i < len(action):
                self._execute_action(self.red_controller, bot_id, action[i])
        
        # Random opponent
        blue_bots = self.blue_controller.get_team_bot_ids(Team.BLUE)
        for bot_id in blue_bots:
            act = random.randint(0, self.NUM_ACTIONS - 1)
            self._execute_action(self.blue_controller, bot_id, act)
        
        # Reward
        current_money = self.game_state.team_money[Team.RED]
        enemy_money = self.game_state.team_money[Team.BLUE]
        
        reward = (current_money - prev_money) * 0.01
        reward += 0.001 if current_money > enemy_money else -0.001
        
        terminated = self.current_turn >= self.total_turns
        if terminated:
            if current_money > enemy_money:
                reward += 5.0
            elif current_money < enemy_money:
                reward -= 2.0
        
        return self._get_obs(), reward, terminated, False, {"money": current_money}
    
    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[0] = self.current_turn / self.total_turns
        obs[1] = self.game_state.team_money[Team.RED] / 2000
        obs[2] = self.game_state.team_money[Team.BLUE] / 2000
        
        idx = 10
        for team in [Team.RED, Team.BLUE]:
            for bot in list(self.game_state.bots.values()):
                if bot.team == team and idx < self.obs_size - 10:
                    obs[idx] = bot.x / 20
                    obs[idx+1] = bot.y / 15
                    obs[idx+2] = 1.0 if bot.holding else 0.0
                    idx += 10
        
        return obs
    
    def _execute_action(self, controller, bot_id, action):
        try:
            if action == 0:
                return True
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                return controller.move(bot_id, dx, dy)
            elif action == 5:
                return controller.pickup(bot_id)
            else:
                bot_state = controller.get_bot_state(bot_id)
                if bot_state:
                    x, y = bot_state["x"], bot_state["y"]
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            try:
                                if action == 6 and controller.place(bot_id, x+dx, y+dy): return True
                                if action == 7 and controller.chop(bot_id, x+dx, y+dy): return True
                                if action == 8 and controller.start_cook(bot_id, x+dx, y+dy): return True
                                if action == 9 and controller.take_from_pan(bot_id, x+dx, y+dy): return True
                                if action == 10 and controller.take_clean_plate(bot_id, x+dx, y+dy): return True
                                if action == 11 and controller.put_dirty_plate_in_sink(bot_id, x+dx, y+dy): return True
                                if action == 12 and controller.wash_sink(bot_id, x+dx, y+dy): return True
                                if action == 13 and controller.add_food_to_plate(bot_id, x+dx, y+dy): return True
                                if action == 14 and controller.submit(bot_id, x+dx, y+dy): return True
                                if action == 15 and controller.buy(bot_id, ShopCosts.PLATE, x+dx, y+dy): return True
                            except:
                                pass
        except:
            pass
        return False


class ProgressCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 100000):
        super().__init__(verbose=1)
        self.save_path = save_path
        self.save_freq = save_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}_{self.n_calls}_steps.zip"
            self.model.save(path)
            original_print(f"💾 Checkpoint: {self.n_calls:,} steps")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    
    original_print("\n" + "="*50)
    original_print("🚀 FAST RL TRAINING (warnings suppressed)")
    original_print("="*50)
    original_print(f"Timesteps: {args.timesteps:,}")
    original_print(f"Parallel Envs: {args.envs}")
    original_print("="*50 + "\n")
    
    # Maps
    maps = [
        str(ROOT_DIR / "maps/official/chopped.txt"),
        str(ROOT_DIR / "maps/official/v1.txt"),
        str(ROOT_DIR / "maps/official/orbit.txt"),
        str(ROOT_DIR / "maps/official/throughput.txt"),
    ]
    maps = [m for m in maps if os.path.exists(m)]
    
    # Create env
    def make_env():
        def _init():
            return FastAWAPEnv(map_paths=maps)
        return _init
    
    env = DummyVecEnv([make_env() for _ in range(args.envs)])
    
    # Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=1,
    )
    
    # Paths
    os.makedirs(str(SCRIPT_DIR / "models"), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = str(SCRIPT_DIR / f"models/fast_{timestamp}")
    
    callback = ProgressCallback(save_path, save_freq=100000)
    
    original_print("Training started...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    
    final_path = f"{save_path}_final.zip"
    model.save(final_path)
    
    original_print("\n" + "="*50)
    original_print(f"✅ Done! Model: {final_path}")
    original_print("="*50)
    
    env.close()


if __name__ == "__main__":
    main()
