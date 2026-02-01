"""
PPO Training Script for AWAP Bot
================================
Train an RL agent using Stable Baselines3 PPO on the AWAP environment.

Usage:
    python train.py --map ../maps/official/chopped.txt --timesteps 100000
    python train.py --eval models/awap_ppo_final.zip
"""

import os
import sys
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for required dependencies
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    print("Run: pip install stable-baselines3")
    sys.exit(1)

from env.awap_env import AWAPEnv


def make_env(map_path: str, opponent: str = "idle", rank: int = 0):
    """Create a wrapped environment instance."""
    def _init():
        env = AWAPEnv(
            map_path=map_path,
            opponent_strategy=opponent,
            reward_scale=0.01,
        )
        return Monitor(env)
    return _init


def train(
    map_path: str = "../maps/official/chopped.txt",
    total_timesteps: int = 100_000,
    save_path: str = "./models",
    log_path: str = "./logs",
    n_envs: int = 4,
    opponent: str = "idle",
):
    """
    Train a PPO agent on the AWAP environment.
    
    Args:
        map_path: Path to the map file for training
        total_timesteps: Total training steps
        save_path: Directory to save model checkpoints
        log_path: Directory for TensorBoard logs
        n_envs: Number of parallel environments
        opponent: Opponent strategy ("idle" or "random")
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Resolve map path
    if not os.path.isabs(map_path):
        map_path = os.path.join(os.path.dirname(__file__), map_path)
    
    print(f"{'='*60}")
    print(f"AWAP RL Training")
    print(f"{'='*60}")
    print(f"Map: {map_path}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Envs: {n_envs}")
    print(f"Opponent: {opponent}")
    print(f"{'='*60}\n")
    
    # Create vectorized environment
    if n_envs > 1:
        try:
            env = SubprocVecEnv([make_env(map_path, opponent, i) for i in range(n_envs)])
        except:
            print("SubprocVecEnv failed, falling back to DummyVecEnv")
            env = DummyVecEnv([make_env(map_path, opponent, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(map_path, opponent, 0)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(map_path, opponent, 0)])
    
    # Initialize PPO agent with SMALL network for competition embedding
    # Smaller network = fewer parameters = smaller embedded weights
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None,  # Disabled - tensorboard not always available
        device="auto",
        policy_kwargs=dict(
            # SMALL network for embedding (~15K params instead of ~100K)
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
    )


    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1000),
        save_path=save_path,
        name_prefix=f"awap_ppo_{timestamp}"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=log_path,
        eval_freq=max(5_000 // n_envs, 500),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # Train!
    print("Starting training...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Save final model
    final_path = os.path.join(save_path, f"awap_ppo_final_{timestamp}")
    model.save(final_path)
    print(f"\n{'='*60}")
    print(f"Model saved to: {final_path}.zip")
    print(f"{'='*60}")
    
    env.close()
    eval_env.close()
    
    return model


def evaluate(model_path: str, map_path: str, n_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        map_path: Path to the map file
        n_episodes: Number of evaluation episodes
        render: Whether to render during evaluation
    """
    
    # Resolve paths
    if not os.path.isabs(map_path):
        map_path = os.path.join(os.path.dirname(__file__), map_path)
    
    print(f"{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Map: {map_path}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = AWAPEnv(
        map_path=map_path,
        opponent_strategy="random",  # Harder opponent for eval
        render_mode="human" if render else None,
    )
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluate
    wins = 0
    total_reward = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        result = info.get("result", "UNKNOWN")
        if result == "WIN":
            wins += 1
        
        print(f"Episode {ep+1:3d}: Reward={ep_reward:7.2f}, RED=${info['red_money']:5d}, BLUE=${info['blue_money']:5d} -> {result}")
        total_reward += ep_reward
    
    print(f"\n{'='*60}")
    print(f"Results: {wins}/{n_episodes} wins ({100*wins/n_episodes:.1f}%)")
    print(f"Average Reward: {total_reward/n_episodes:.2f}")
    print(f"{'='*60}")
    
    env.close()


def test_env(map_path: str):
    """Quick sanity check of the environment."""
    
    if not os.path.isabs(map_path):
        map_path = os.path.join(os.path.dirname(__file__), map_path)
    
    print(f"Testing environment with map: {map_path}")
    
    env = AWAPEnv(map_path=map_path, opponent_strategy="idle")
    
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, money=${info['red_money']}")
        if done:
            break
    
    print(f"✓ Environment test passed!")
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train/Evaluate RL agent for AWAP")
    parser.add_argument("--map", type=str, default="../maps/official/chopped.txt",
                        help="Path to training map")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--opponent", type=str, default="idle",
                        choices=["idle", "random"],
                        help="Opponent strategy")
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to model for evaluation (skip training)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--test", action="store_true",
                        help="Quick environment test")
    
    args = parser.parse_args()
    
    if args.test:
        test_env(args.map)
    elif args.eval:
        evaluate(args.eval, args.map, n_episodes=args.episodes)
    else:
        train(
            map_path=args.map,
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            opponent=args.opponent,
        )
