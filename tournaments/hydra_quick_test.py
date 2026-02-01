"""
Quick test of Hydra improvements against key competitors
"""

import subprocess
import sys
import os
from pathlib import Path

HERO_BOT = "bots/hareshm/BEST-Hydra-Bot.py"

# Test against all key bots
COMPETITORS = [
    "bots/hareshm/champion_bot.py",
    "bots/hareshm/SOVEREIGN.py",
    "bots/eric/BEST-IronChefOptimized.py",
    "bots/dpinto/BEST_solo_bot_v1.py",
    "bots/test-bots/rush_bot.py",
    "bots/test-bots/efficiency_maximizer.py",
    "bots/test-bots/turtle_defender.py",
]

# Focus on official maps
MAPS = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt",
]


def run_game(red_bot: str, blue_bot: str, map_path: str, base_dir: str) -> str:
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--timeout", "0.5"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120,
            cwd=base_dir
        )
        output = result.stdout + result.stderr
        
        if "RED WINS" in output:
            return "WIN"
        elif "BLUE WINS" in output:
            return "LOSS"
        elif "DRAW" in output:
            return "DRAW"
        else:
            return "ERROR"
    except:
        return "ERROR"


def main():
    base_dir = str(Path(__file__).parent.parent)
    
    print("=" * 60)
    print("HYDRA QUICK TEST - Key Matchups")
    print("=" * 60)
    
    for competitor in COMPETITORS:
        comp_name = Path(competitor).stem
        wins = 0
        losses = 0
        draws = 0
        
        print(f"\nHydra vs {comp_name}:")
        
        for map_path in MAPS:
            map_name = Path(map_path).stem
            result = run_game(HERO_BOT, competitor, map_path, base_dir)
            
            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1
            else:
                draws += 1
            
            print(f"  {map_name:15s}: {result}")
        
        total = wins + losses + draws
        win_rate = wins / total * 100 if total > 0 else 0
        print(f"  Summary: {wins}W-{losses}L-{draws}D ({win_rate:.1f}%)")


if __name__ == "__main__":
    main()
