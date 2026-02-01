"""
NumPy RL Bot - Competition Legal (Template)
=============================================
This is a TEMPLATE bot with placeholder weights.

To create a trained bot:
1. Train: python3 train.py --timesteps 100000
2. Export: python3 export_model.py models/best/best_model.zip -o ../bots/dpinto/numpy_rl_bot.py

Allowed packages: stdlib, numpy, scipy, game engine imports
NO PyTorch/TensorFlow allowed at runtime!
"""

import gzip
import base64
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Placeholder weights - will be replaced by export_model.py
# This creates a random network for testing the infrastructure
WEIGHTS_B64 = None  # Will be set by export script


# ============================================================================
# NumPy Neural Network
# ============================================================================

class NumpyMLP:
    """Simple MLP using only NumPy."""
    
    def __init__(self, layers: List[Tuple[np.ndarray, np.ndarray]]):
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activations (except last layer)."""
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
        return x
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.forward(obs)


def load_model() -> NumpyMLP:
    """Load model - either from embedded weights or create random."""
    if WEIGHTS_B64 is not None:
        compressed = base64.b64decode(WEIGHTS_B64.strip())
        pickled = gzip.decompress(compressed)
        weights_dict = pickle.loads(pickled)
        return NumpyMLP(weights_dict['layers'])
    else:
        # Random fallback for testing (with proper Xavier initialization)
        print("[NumPy RL Bot] No trained weights - using random policy")
        np.random.seed(42)
        layers = [
            (np.random.randn(64, 203).astype(np.float32) * np.sqrt(2.0 / 203),
             np.zeros(64, dtype=np.float32)),
            (np.random.randn(64, 64).astype(np.float32) * np.sqrt(2.0 / 64),
             np.zeros(64, dtype=np.float32)),
            (np.random.randn(64, 64).astype(np.float32) * np.sqrt(2.0 / 64),
             np.zeros(64, dtype=np.float32)),
        ]
        return NumpyMLP(layers)



# ============================================================================
# Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract observation vector from game state via RobotController."""
    
    def __init__(self, rc, team):
        self.rc = rc
        self.team = team
        self.enemy_team = self._get_enemy_team(team)
        self.max_bots = 4
        self.obs_size = 203
        
    def _get_enemy_team(self, team):
        from game_constants import Team
        return Team.BLUE if team == Team.RED else Team.RED
    
    def extract(self) -> np.ndarray:
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
        obs.append(self.rc.get_team_money(self.team) / 5000.0)
        obs.append(self.rc.get_team_money(self.enemy_team) / 5000.0)
        obs.append(self.rc.get_turn() / 500.0)
        
        # Orders
        orders = self.rc.get_orders(self.team)
        turn = self.rc.get_turn()
        active_orders = [o for o in orders if o.get('is_active', False)][:10]
        
        for i in range(10):
            if i < len(active_orders):
                order = active_orders[i]
                req = order.get('required', [])
                obs.append(1.0 if 'EGG' in req else 0.0)
                obs.append(1.0 if 'ONIONS' in req else 0.0)
                obs.append(1.0 if 'MEAT' in req else 0.0)
                obs.append(1.0 if 'NOODLES' in req else 0.0)
                obs.append(1.0 if 'SAUCE' in req else 0.0)
                deadline = order.get('expires_turn', 0) - turn
                obs.append(max(0, deadline) / 100.0)
                obs.append(order.get('reward', 0) / 100.0)
            else:
                obs.extend([0.0] * 7)
        
        # Map features placeholder
        obs.extend([0.0] * 10)
        
        while len(obs) < self.obs_size:
            obs.append(0.0)
        
        return np.array(obs[:self.obs_size], dtype=np.float32)
    
    def _encode_bot(self, bot_state: Optional[Dict]) -> List[float]:
        if bot_state is None:
            return [0.0] * 15
        
        features = []
        features.append(bot_state.get('x', 0) / 50.0)
        features.append(bot_state.get('y', 0) / 50.0)
        
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
        
        if holding and holding.get('type') == 'Food':
            features.append(holding.get('food_id', 0) / 5.0)
        else:
            features.append(0.0)
        
        if holding and holding.get('type') == 'Plate':
            features.append(len(holding.get('food', [])) / 5.0)
        else:
            features.append(0.0)
        
        features.extend([0.0] * 3)
        return features


# ============================================================================
# Action Decoder
# ============================================================================

class ActionDecoder:
    MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    def __init__(self, rc):
        self.rc = rc
    
    def execute(self, bot_id: int, action: int):
        try:
            if action == 0:  # STAY
                pass
            elif action in self.MOVE_DELTAS:
                dx, dy = self.MOVE_DELTAS[action]
                self.rc.move(bot_id, dx, dy)
            elif action == 5:  # PICKUP
                self.rc.pickup(bot_id)
            elif action == 6:  # PLACE
                self._try_neighbors(self.rc.place, bot_id)
            elif action == 7:  # CHOP
                self._try_neighbors(self.rc.chop, bot_id)
            elif action == 8:  # START_COOK
                self._try_neighbors(self.rc.start_cook, bot_id)
            elif action == 9:  # TAKE_FROM_PAN
                self._try_neighbors(self.rc.take_from_pan, bot_id)
            elif action == 10:  # TAKE_CLEAN_PLATE
                self._try_neighbors(self.rc.take_clean_plate, bot_id)
            elif action == 11:  # PUT_DIRTY_PLATE
                self._try_neighbors(self.rc.put_dirty_plate_in_sink, bot_id)
            elif action == 12:  # WASH_SINK
                self._try_neighbors(self.rc.wash_sink, bot_id)
            elif action == 13:  # ADD_FOOD_TO_PLATE
                self._try_neighbors(self.rc.add_food_to_plate, bot_id)
            elif action == 14:  # SUBMIT
                self._try_neighbors(self.rc.submit, bot_id)
            elif action == 15:  # BUY_PLATE
                from game_constants import ShopCosts
                self._try_neighbors(
                    lambda bid, x, y: self.rc.buy(bid, ShopCosts.PLATE, x, y),
                    bot_id
                )
        except:
            pass
    
    def _try_neighbors(self, action_fn, bot_id: int):
        bot_state = self.rc.get_bot_state(bot_id)
        if not bot_state:
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
# Bot Player
# ============================================================================

class BotPlayer:
    """Competition-legal RL bot using NumPy-only inference."""
    
    def __init__(self, map_state):
        self.map_state = map_state
        self.model = load_model()
        self.num_actions = 16
        self.max_bots = 4
        
    def play_turn(self, rc):
        from game_constants import Team
        
        team = rc.get_team()
        extractor = FeatureExtractor(rc, team)
        obs = extractor.extract()
        
        logits = self.model.predict(obs)
        
        bot_ids = rc.get_team_bot_ids(team)
        decoder = ActionDecoder(rc)
        
        for i, bot_id in enumerate(bot_ids):
            if i >= self.max_bots:
                break
            
            start_idx = i * self.num_actions
            end_idx = start_idx + self.num_actions
            
            if end_idx <= len(logits):
                bot_logits = logits[start_idx:end_idx]
                action = int(np.argmax(bot_logits))
            else:
                action = 0
            
            decoder.execute(bot_id, action)
