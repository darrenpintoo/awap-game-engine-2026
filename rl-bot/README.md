# RL Bot Training Framework

This folder contains the reinforcement learning pipeline to train a competitive bot
using data from tournament simulations.

## Structure
```
rl-bot/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── env/                # Custom Gym environment
│   └── awap_env.py     # Game wrapper as RL environment
├── agents/             # RL algorithms
│   └── ppo_agent.py    # PPO implementation
├── data/               # Training data extraction
│   └── parse_matches.py # Extract state-action pairs from JSON
├── train.py            # Main training script
└── rl_bot.py           # Trained model wrapped as BotPlayer
```

## Quick Start
```bash
cd rl-bot
pip install -r requirements.txt
python train.py
```
