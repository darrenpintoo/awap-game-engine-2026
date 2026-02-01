"""
RL Bot Player
=============
Wraps a trained RL model as a BotPlayer for integration with the game engine.
"""

import numpy as np
import os
from stable_baselines3 import PPO

class BotPlayer:
    """
    BotPlayer wrapper for a trained RL model.
    Can be used directly in the game engine.
    """
    
    def __init__(self, map_state):
        """
        Initialize the RL bot.
        
        Args:
            map_state: The game map state (provided by game engine)
        """
        self.map_state = map_state
        
        # Load the trained model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "awap_ppo_final.zip"  # Update with actual model path
        )
        
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            self.use_model = True
        else:
            print(f"[RL Bot] No trained model found at {model_path}")
            print("[RL Bot] Falling back to random actions")
            self.use_model = False
        
        self.num_bots = len(map_state.get("bots", []))
        
    def _extract_observation(self, game_state) -> np.ndarray:
        """
        Convert game state to observation vector.
        Must match the format used during training.
        """
        obs = []
        
        # Extract bot positions
        for bot in game_state.get("my_bots", []):
            pos = bot.get("pos", (0, 0))
            obs.extend([pos[0] / 50.0, pos[1] / 50.0])
        
        # Pad if needed
        while len(obs) < self.num_bots * 2:
            obs.extend([0.0, 0.0])
        
        # Enemy bot positions
        for bot in game_state.get("enemy_bots", []):
            pos = bot.get("pos", (0, 0))
            obs.extend([pos[0] / 50.0, pos[1] / 50.0])
            
        while len(obs) < self.num_bots * 4:
            obs.extend([0.0, 0.0])
        
        # Money
        obs.append(game_state.get("my_money", 0) / 10000.0)
        obs.append(game_state.get("enemy_money", 0) / 10000.0)
        
        # Orders (simplified)
        for order in game_state.get("orders", [])[:10]:
            obs.extend([
                order.get("deadline", 0) / 1000.0,
                order.get("value", 0) / 500.0,
                1.0 if order.get("active", False) else 0.0
            ])
        
        # Pad to expected size
        expected_size = self.num_bots * 4 + 2 + 30
        while len(obs) < expected_size:
            obs.append(0.0)
            
        return np.array(obs[:expected_size], dtype=np.float32)
    
    def _action_to_command(self, action_idx: int, bot_id: int) -> dict:
        """
        Convert discrete action index to game command.
        """
        action_map = {
            0: {"type": "MOVE", "dir": "UP"},
            1: {"type": "MOVE", "dir": "DOWN"},
            2: {"type": "MOVE", "dir": "LEFT"},
            3: {"type": "MOVE", "dir": "RIGHT"},
            4: {"type": "PICKUP"},
            5: {"type": "DROP"},
            6: {"type": "COOK"},
            7: {"type": "PLATE"},
            8: {"type": "SERVE"},
            9: {"type": "WASH"},
            10: {"type": "SABOTAGE"},
        }
        
        cmd = action_map.get(action_idx, {"type": "MOVE", "dir": "STAY"})
        cmd["bot_id"] = bot_id
        return cmd
    
    def get_actions(self, game_state) -> list:
        """
        Main entry point called by the game engine each turn.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            List of action commands for each bot
        """
        if not self.use_model:
            # Random fallback
            return [
                self._action_to_command(np.random.randint(0, 11), i)
                for i in range(self.num_bots)
            ]
        
        # Get observation
        obs = self._extract_observation(game_state)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert to game commands
        commands = []
        for i, a in enumerate(action):
            commands.append(self._action_to_command(int(a), i))
        
        return commands
