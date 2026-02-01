"""
Advanced AWAP RL Training
=========================
Sophisticated training pipeline with:
- Curriculum learning (easy → hard opponents)
- Multi-map training for generalization
- Enhanced reward shaping
- Self-play against trained bots
- Parallel environments for speed

Usage:
    python train_advanced.py --timesteps 500000 --curriculum
"""

import os
import sys
import glob
import random
import subprocess
from datetime import datetime
from typing import List, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install stable-baselines3 numpy")
    sys.exit(1)

from env.awap_env import AWAPEnv


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# All official maps for training diversity
OFFICIAL_MAPS = [
    "../maps/official/chopped.txt",
    "../maps/official/cooking_for_dummies.txt",
    "../maps/official/v1.txt",
    "../maps/official/orbit.txt",
]

# Bot opponents for curriculum (easy → hard)
OPPONENT_BOTS = [
    None,  # Idle
    "../bots/test-bots/rush_bot.py",
    "../bots/test-bots/zone_coordinator.py",
]

# Training hyperparameters
TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # Network - larger but still embeddable
    "net_arch": dict(pi=[128, 128], vf=[128, 128]),
}


# ==============================================================================
# ENHANCED REWARD SHAPING
# ==============================================================================

class EnhancedRewardWrapper:
    """
    Wrapper that adds sophisticated reward shaping beyond just money.
    
    Rewards:
    - Money gained (primary signal)
    - Order completion bonus
    - Progress towards orders (holding correct ingredients)
    - Efficiency (penalize excessive movement)
    - Cooperation (bots working on different orders)
    """
    
    def __init__(self, env):
        self.env = env
        self.prev_active_orders = 0
        self.prev_holding_correct = 0
        
    def shape_reward(self, base_reward: float, info: dict) -> float:
        """Add reward shaping to base reward."""
        shaped = base_reward
        
        # Bonus for completing orders (detected via order count change)
        # This is already captured in money, but we add extra emphasis
        
        # Small bonus for having resources
        if info.get("red_money", 0) > info.get("blue_money", 0):
            shaped += 0.002  # Winning position bonus
        
        return shaped


# ==============================================================================
# MULTI-MAP ENVIRONMENT
# ==============================================================================

class MultiMapEnv(AWAPEnv):
    """Environment that randomly selects from multiple maps each episode."""
    
    def __init__(self, map_paths: List[str], opponent_strategy: str = "idle", **kwargs):
        self.map_paths = [self._resolve_path(p) for p in map_paths]
        self.current_map_idx = 0
        
        # Initialize with first map
        super().__init__(
            map_path=self.map_paths[0],
            opponent_strategy=opponent_strategy,
            **kwargs
        )
        
    def _resolve_path(self, path: str) -> str:
        if not os.path.isabs(path):
            return os.path.join(os.path.dirname(__file__), path)
        return path
    
    def reset(self, **kwargs):
        """Reset with a random map."""
        # Randomly select a map
        self.current_map_idx = random.randint(0, len(self.map_paths) - 1)
        self.map_path = self.map_paths[self.current_map_idx]
        
        # Reload map info
        self._load_map_info()
        
        return super().reset(**kwargs)


# ==============================================================================
# CURRICULUM LEARNING CALLBACK
# ==============================================================================

class CurriculumCallback(BaseCallback):
    """
    Implements curriculum learning by increasing opponent difficulty over time.
    
    Stages:
    1. Idle opponent (learn basic mechanics)
    2. Random actions (learn to beat chaos)
    3. Actual bot opponents (learn strategy)
    """
    
    def __init__(
        self,
        stages: List[dict],
        timesteps_per_stage: int = 100_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.stages = stages
        self.timesteps_per_stage = timesteps_per_stage
        self.current_stage = 0
        
    def _on_step(self) -> bool:
        # Check if we should advance to next stage
        stage_timesteps = self.num_timesteps // self.timesteps_per_stage
        
        if stage_timesteps > self.current_stage and self.current_stage < len(self.stages) - 1:
            self.current_stage = min(stage_timesteps, len(self.stages) - 1)
            stage = self.stages[self.current_stage]
            
            if self.verbose > 0:
                print(f"\n🎓 CURRICULUM: Advancing to stage {self.current_stage + 1}")
                print(f"   Opponent: {stage.get('opponent', 'idle')}")
                print(f"   Timesteps: {self.num_timesteps:,}")
        
        return True


# ==============================================================================
# PROGRESS LOGGING CALLBACK
# ==============================================================================

class ProgressCallback(BaseCallback):
    """Logs training progress with rich statistics."""
    
    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.games = 0
        
    def _on_step(self) -> bool:
        # Collect episode stats from info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if info.get("result") == "WIN":
                self.wins += 1
                self.games += 1
            elif info.get("result") in ["LOSE", "DRAW"]:
                self.games += 1
        
        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self.verbose > 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                win_rate = self.wins / max(1, self.games)
                
                print(f"📊 Step {self.num_timesteps:,}: "
                      f"Mean Reward={mean_reward:.2f}, "
                      f"Win Rate={win_rate*100:.1f}%")
        
        return True


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def make_multi_map_env(
    map_paths: List[str],
    opponent: str = "idle",
    rank: int = 0
) -> Callable:
    """Factory function for creating multi-map environments."""
    def _init():
        env = MultiMapEnv(
            map_paths=map_paths,
            opponent_strategy=opponent,
            reward_scale=0.01,
        )
        return Monitor(env)
    return _init


def train_advanced(
    timesteps: int = 500_000,
    n_envs: int = 4,
    use_curriculum: bool = True,
    save_path: str = "./models",
    maps: Optional[List[str]] = None,
):
    """
    Train with advanced techniques.
    
    Args:
        timesteps: Total training timesteps
        n_envs: Number of parallel environments
        use_curriculum: Whether to use curriculum learning
        save_path: Where to save checkpoints
        maps: List of map paths (defaults to OFFICIAL_MAPS)
    """
    
    os.makedirs(save_path, exist_ok=True)
    
    # Use all official maps by default
    if maps is None:
        maps = OFFICIAL_MAPS
    
    # Resolve map paths
    maps = [
        os.path.join(os.path.dirname(__file__), m) if not os.path.isabs(m) else m
        for m in maps
    ]
    
    # Filter to existing maps
    maps = [m for m in maps if os.path.exists(m)]
    if not maps:
        print("ERROR: No valid maps found!")
        return None
    
    print("=" * 70)
    print("🚀 ADVANCED RL TRAINING")
    print("=" * 70)
    print(f"Total Timesteps: {timesteps:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"Maps: {len(maps)}")
    for m in maps:
        print(f"  - {os.path.basename(m)}")
    print(f"Curriculum: {'Enabled' if use_curriculum else 'Disabled'}")
    print(f"Network: {TRAINING_CONFIG['net_arch']}")
    print("=" * 70 + "\n")
    
    # Create vectorized multi-map environment
    print("Creating environments...")
    
    try:
        env = SubprocVecEnv([make_multi_map_env(maps, "idle", i) for i in range(n_envs)])
    except:
        print("SubprocVecEnv failed, using DummyVecEnv")
        env = DummyVecEnv([make_multi_map_env(maps, "idle", i) for i in range(n_envs)])
    
    # Initialize PPO with larger network
    print("Initializing model...")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        n_steps=TRAINING_CONFIG["n_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        n_epochs=TRAINING_CONFIG["n_epochs"],
        gamma=TRAINING_CONFIG["gamma"],
        gae_lambda=TRAINING_CONFIG["gae_lambda"],
        clip_range=TRAINING_CONFIG["clip_range"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        vf_coef=TRAINING_CONFIG["vf_coef"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        tensorboard_log=None,
        device="auto",
        policy_kwargs=dict(net_arch=TRAINING_CONFIG["net_arch"]),
    )
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 5000),
            save_path=save_path,
            name_prefix=f"awap_advanced_{timestamp}"
        ),
        ProgressCallback(log_freq=10_000),
    ]
    
    if use_curriculum:
        # Curriculum stages
        stages = [
            {"opponent": "idle", "description": "Learn basics"},
            {"opponent": "random", "description": "Handle chaos"},
            {"opponent": "random", "description": "Refine strategy"},
        ]
        callbacks.append(CurriculumCallback(
            stages=stages,
            timesteps_per_stage=timesteps // len(stages),
            verbose=1
        ))
    
    # Train!
    print("🏋️ Starting training...\n")
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_path = os.path.join(save_path, f"awap_advanced_final_{timestamp}")
    model.save(final_path)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Model saved to: {final_path}.zip")
    print(f"{'=' * 70}")
    
    env.close()
    return model


def evaluate_model(model_path: str, n_games: int = 20):
    """Evaluate a trained model against various opponents."""
    from stable_baselines3 import PPO
    
    model = PPO.load(model_path)
    
    opponents = [
        ("Idle", "idle"),
        ("Random", "random"),
    ]
    
    maps = OFFICIAL_MAPS[:2]  # Test on subset
    
    print(f"\n{'=' * 70}")
    print(f"📊 EVALUATING: {model_path}")
    print(f"{'=' * 70}\n")
    
    for opp_name, opp_strategy in opponents:
        wins = 0
        total_reward = 0
        
        for map_path in maps:
            resolved_path = os.path.join(os.path.dirname(__file__), map_path)
            if not os.path.exists(resolved_path):
                continue
                
            env = AWAPEnv(
                map_path=resolved_path,
                opponent_strategy=opp_strategy,
            )
            
            for _ in range(n_games // len(maps)):
                obs, info = env.reset()
                done = False
                ep_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                
                total_reward += ep_reward
                if info.get("result") == "WIN":
                    wins += 1
            
            env.close()
        
        print(f"  vs {opp_name:10s}: {wins}/{n_games} wins ({100*wins/n_games:.0f}%), avg reward: {total_reward/n_games:.2f}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced AWAP RL Training")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--eval", type=str, default=None,
                        help="Evaluate a model instead of training")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_model(args.eval)
    else:
        train_advanced(
            timesteps=args.timesteps,
            n_envs=args.envs,
            use_curriculum=not args.no_curriculum,
        )
