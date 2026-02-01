"""
Model Export Script
===================
Extracts weights from a trained Stable Baselines3 PPO model and exports them
as a compressed, embeddable format for NumPy-only inference.

Usage:
    python export_model.py models/best/best_model.zip --output ../bots/dpinto/numpy_rl_bot.py
"""

import os
import sys
import gzip
import base64
import pickle
import argparse
import numpy as np

def extract_weights_from_sb3(model_path: str) -> dict:
    """
    Extract policy network weights from a Stable Baselines3 PPO model.
    
    Returns dict with:
        - 'layers': list of (weight, bias) tuples for each layer
        - 'obs_size': input observation size
        - 'action_size': output action size
    """
    from stable_baselines3 import PPO
    import torch
    
    model = PPO.load(model_path)
    policy = model.policy
    
    # Extract MLP extractor weights
    layers = []
    
    # Policy network (actor)
    # SB3 structure: mlp_extractor.policy_net -> action_net
    
    # Get the policy MLP layers
    for i, layer in enumerate(policy.mlp_extractor.policy_net):
        if hasattr(layer, 'weight'):
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            layers.append((w, b))
            print(f"  Layer {len(layers)}: {w.shape[1]} -> {w.shape[0]}")
    
    # Get the action net (final layer)
    w = policy.action_net.weight.detach().cpu().numpy()
    b = policy.action_net.bias.detach().cpu().numpy()
    layers.append((w, b))
    print(f"  Action Layer: {w.shape[1]} -> {w.shape[0]}")
    
    # Get observation size from first layer
    obs_size = layers[0][0].shape[1]
    action_size = layers[-1][0].shape[0]
    
    return {
        'layers': layers,
        'obs_size': obs_size,
        'action_size': action_size,
    }


def compress_weights(weights_dict: dict) -> str:
    """Compress weights dict to base64 string."""
    # Pickle the weights
    pickled = pickle.dumps(weights_dict)
    
    # Compress with gzip
    compressed = gzip.compress(pickled, compresslevel=9)
    
    # Encode as base64
    encoded = base64.b64encode(compressed).decode('ascii')
    
    print(f"  Original size: {len(pickled):,} bytes")
    print(f"  Compressed: {len(compressed):,} bytes")
    print(f"  Base64: {len(encoded):,} chars")
    
    return encoded


def generate_bot_code(weights_b64: str, obs_size: int, action_size: int) -> str:
    """Generate the complete NumPy bot Python code."""
    
    template = '''"""
NumPy RL Bot - Competition Legal
================================
Trained with PPO, runs with NumPy only.
No PyTorch/TensorFlow dependencies at runtime.

Allowed packages: stdlib, numpy, scipy, game engine imports
"""

import gzip
import base64
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Compressed neural network weights (gzip + base64)
WEIGHTS_B64 = """
{weights_b64}
"""

# ============================================================================
# NumPy Neural Network
# ============================================================================

class NumpyMLP:
    """Simple MLP using only NumPy."""
    
    def __init__(self, layers: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Args:
            layers: List of (weight, bias) tuples
        """
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activations (except last layer)."""
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            # ReLU for all but last layer
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
        return x
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get action from observation."""
        logits = self.forward(obs)
        return logits


def load_model() -> NumpyMLP:
    """Load the embedded model weights."""
    # Decode and decompress
    compressed = base64.b64decode(WEIGHTS_B64.strip())
    pickled = gzip.decompress(compressed)
    weights_dict = pickle.loads(pickled)
    
    return NumpyMLP(weights_dict['layers'])


# ============================================================================
# Feature Extraction (must match training environment)
# ============================================================================

class FeatureExtractor:
    """Extract observation vector from game state via RobotController."""
    
    def __init__(self, rc, team):
        self.rc = rc
        self.team = team
        self.enemy_team = self._get_enemy_team(team)
        self.max_bots = 4
        self.obs_size = {obs_size}
        
    def _get_enemy_team(self, team):
        from game_constants import Team
        return Team.BLUE if team == Team.RED else Team.RED
    
    def extract(self) -> np.ndarray:
        """Extract observation vector."""
        obs = []
        
        # My bots
        my_bot_ids = self.rc.get_team_bot_ids(self.team)
        for i in range(self.max_bots):
            if i < len(my_bot_ids):
                bot_state = self.rc.get_bot_state(my_bot_ids[i])
                obs.extend(self._encode_bot(bot_state))
            else:
                obs.extend([0.0] * 15)
        
        # Enemy bots
        enemy_bot_ids = self.rc.get_team_bot_ids(self.enemy_team)
        for i in range(self.max_bots):
            if i < len(enemy_bot_ids):
                bot_state = self.rc.get_bot_state(enemy_bot_ids[i])
                obs.extend(self._encode_bot(bot_state))
            else:
                obs.extend([0.0] * 15)
        
        # Global state
        my_money = self.rc.get_team_money(self.team)
        enemy_money = self.rc.get_team_money(self.enemy_team)
        turn = self.rc.get_turn()
        
        obs.append(my_money / 5000.0)
        obs.append(enemy_money / 5000.0)
        obs.append(turn / 500.0)
        
        # Orders
        orders = self.rc.get_orders(self.team)
        active_orders = [o for o in orders if o.get('is_active', False)][:10]
        
        for i in range(10):
            if i < len(active_orders):
                order = active_orders[i]
                # Required foods (simplified encoding)
                req = order.get('required', [])
                obs.append(1.0 if 'EGG' in req else 0.0)
                obs.append(1.0 if 'ONIONS' in req else 0.0)
                obs.append(1.0 if 'MEAT' in req else 0.0)
                obs.append(1.0 if 'NOODLES' in req else 0.0)
                obs.append(1.0 if 'SAUCE' in req else 0.0)
                # Deadline and reward
                deadline = order.get('expires_turn', 0) - turn
                obs.append(max(0, deadline) / 100.0)
                obs.append(order.get('reward', 0) / 100.0)
            else:
                obs.extend([0.0] * 7)
        
        # Pad to expected size
        while len(obs) < self.obs_size:
            obs.append(0.0)
        
        return np.array(obs[:self.obs_size], dtype=np.float32)
    
    def _encode_bot(self, bot_state: Optional[Dict]) -> List[float]:
        """Encode a single bot's state."""
        if bot_state is None:
            return [0.0] * 15
        
        features = []
        
        # Position (normalized, assume 50x50 max)
        features.append(bot_state.get('x', 0) / 50.0)
        features.append(bot_state.get('y', 0) / 50.0)
        
        # Holding type (one-hot)
        holding = bot_state.get('holding')
        hold_enc = [0.0] * 8
        
        if holding is None:
            hold_enc[0] = 1.0
        elif holding.get('type') == 'Food':
            hold_enc[1] = 1.0
            if holding.get('chopped', False):
                hold_enc[2] = 1.0
            if holding.get('cooked_stage', 0) > 0:
                hold_enc[3] = 1.0
        elif holding.get('type') == 'Plate':
            hold_enc[4] = 1.0
            if holding.get('dirty', False):
                hold_enc[5] = 1.0
        elif holding.get('type') == 'Pan':
            hold_enc[6] = 1.0
            if holding.get('food') is not None:
                hold_enc[7] = 1.0
        
        features.extend(hold_enc)
        
        # Holding details
        if holding and holding.get('type') == 'Food':
            food_id = holding.get('food_id', 0)
            features.append(food_id / 5.0)
        else:
            features.append(0.0)
        
        if holding and holding.get('type') == 'Plate':
            food_count = len(holding.get('food', []))
            features.append(food_count / 5.0)
        else:
            features.append(0.0)
        
        # Padding
        features.extend([0.0] * 3)
        
        return features


# ============================================================================
# Action Decoder
# ============================================================================

class ActionDecoder:
    """Convert neural network output to RobotController actions."""
    
    # Action indices (must match training)
    ACTION_STAY = 0
    ACTION_MOVE_UP = 1
    ACTION_MOVE_DOWN = 2
    ACTION_MOVE_LEFT = 3
    ACTION_MOVE_RIGHT = 4
    ACTION_PICKUP = 5
    ACTION_PLACE = 6
    ACTION_CHOP = 7
    ACTION_START_COOK = 8
    ACTION_TAKE_FROM_PAN = 9
    ACTION_TAKE_CLEAN_PLATE = 10
    ACTION_PUT_DIRTY_PLATE = 11
    ACTION_WASH_SINK = 12
    ACTION_ADD_FOOD_TO_PLATE = 13
    ACTION_SUBMIT = 14
    ACTION_BUY_PLATE = 15
    
    MOVE_DELTAS = {{
        1: (0, -1),  # UP
        2: (0, 1),   # DOWN
        3: (-1, 0),  # LEFT
        4: (1, 0),   # RIGHT
    }}
    
    def __init__(self, rc):
        self.rc = rc
    
    def execute(self, bot_id: int, action: int):
        """Execute a single action for a bot."""
        try:
            if action == self.ACTION_STAY:
                pass
            
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                self.rc.move(bot_id, dx, dy)
            
            elif action == self.ACTION_PICKUP:
                self.rc.pickup(bot_id)
            
            elif action == self.ACTION_PLACE:
                self._try_neighbors(self.rc.place, bot_id)
            
            elif action == self.ACTION_CHOP:
                self._try_neighbors(self.rc.chop, bot_id)
            
            elif action == self.ACTION_START_COOK:
                self._try_neighbors(self.rc.start_cook, bot_id)
            
            elif action == self.ACTION_TAKE_FROM_PAN:
                self._try_neighbors(self.rc.take_from_pan, bot_id)
            
            elif action == self.ACTION_TAKE_CLEAN_PLATE:
                self._try_neighbors(self.rc.take_clean_plate, bot_id)
            
            elif action == self.ACTION_PUT_DIRTY_PLATE:
                self._try_neighbors(self.rc.put_dirty_plate_in_sink, bot_id)
            
            elif action == self.ACTION_WASH_SINK:
                self._try_neighbors(self.rc.wash_sink, bot_id)
            
            elif action == self.ACTION_ADD_FOOD_TO_PLATE:
                self._try_neighbors(self.rc.add_food_to_plate, bot_id)
            
            elif action == self.ACTION_SUBMIT:
                self._try_neighbors(self.rc.submit, bot_id)
            
            elif action == self.ACTION_BUY_PLATE:
                from game_constants import ShopCosts
                self._try_neighbors(
                    lambda bid, x, y: self.rc.buy(bid, ShopCosts.PLATE, x, y),
                    bot_id
                )
        except:
            pass
    
    def _try_neighbors(self, action_fn, bot_id: int):
        """Try action on current tile and all neighbors."""
        bot_state = self.rc.get_bot_state(bot_id)
        if bot_state is None:
            return
        
        x, y = bot_state['x'], bot_state['y']
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                try:
                    if action_fn(bot_id, x + dx, y + dy):
                        return
                except:
                    pass


# ============================================================================
# Bot Player (Game Engine Interface)
# ============================================================================

class BotPlayer:
    """
    Competition-legal RL bot using NumPy-only inference.
    """
    
    def __init__(self, map_state):
        self.map_state = map_state
        self.model = load_model()
        self.num_actions = 16
        self.max_bots = 4
        
    def play_turn(self, rc):
        """Main entry point called by game engine."""
        from game_constants import Team
        
        team = rc.get_team()
        
        # Extract features
        extractor = FeatureExtractor(rc, team)
        obs = extractor.extract()
        
        # Get actions from neural network
        logits = self.model.predict(obs)
        
        # Decode actions (reshape to per-bot actions)
        # Output is [action_size] = 16 * max_bots = 64
        bot_ids = rc.get_team_bot_ids(team)
        decoder = ActionDecoder(rc)
        
        for i, bot_id in enumerate(bot_ids):
            if i >= self.max_bots:
                break
            
            # Get logits for this bot
            start_idx = i * self.num_actions
            end_idx = start_idx + self.num_actions
            
            if end_idx <= len(logits):
                bot_logits = logits[start_idx:end_idx]
                action = int(np.argmax(bot_logits))
            else:
                action = 0  # Stay if no logits
            
            decoder.execute(bot_id, action)
'''
    
    # Insert weights
    code = template.format(
        weights_b64=weights_b64,
        obs_size=obs_size,
    )
    
    return code


def main():
    parser = argparse.ArgumentParser(description="Export SB3 model to NumPy bot")
    parser.add_argument("model_path", help="Path to trained SB3 model (.zip)")
    parser.add_argument("--output", "-o", default="../bots/dpinto/numpy_rl_bot.py",
                        help="Output bot file path")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_path}")
    weights = extract_weights_from_sb3(args.model_path)
    
    print(f"\\nCompressing weights...")
    weights_b64 = compress_weights(weights)
    
    print(f"\\nGenerating bot code...")
    code = generate_bot_code(
        weights_b64,
        weights['obs_size'],
        weights['action_size'],
    )
    
    # Write output
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(code)
    
    print(f"\\n{'='*60}")
    print(f"Bot exported to: {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
