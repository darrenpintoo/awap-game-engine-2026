#!/usr/bin/env python3
"""Round-robin tournament for all bots in bots/ folder"""

import subprocess
import sys
import os
from itertools import combinations
from collections import defaultdict

# All bots
BOTS = [
    "bots/dpinto/planner_bot.py",
    "bots/duo_noodle_bot.py",
    "bots/eric/iron_chef_bot.py",
    "bots/eric/ironclad_bot.py",
    "bots/eric/PipelineChefBot.py",
    "bots/hareshm/optimal_bot.py",
]

# All maps
MAPS = [
    "maps/dpinto/chaos_kitchen.txt",
    "maps/dpinto/multi_ingredient.txt",
    "maps/dpinto/time_pressure.txt",
    "maps/dpinto/resource_war.txt",
    "maps/dpinto/easy_ramp.txt",
    "maps/dpinto/rush_orders.txt",
    "maps/eric/map2.txt",
    "maps/eric/map3_sprint.txt",
    "maps/eric/map4_chaos.txt",
    "maps/eric/map6_overload.txt",
    "maps/haresh/map1.txt",
]

TURNS = 200
TIMEOUT = 0

def run_game(red_bot, blue_bot, map_file):
    """Run a single game and return (red_score, blue_score, winner)"""
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_file,
        "--turns", str(TURNS),
        "--timeout", str(TIMEOUT)
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":src"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        output = result.stdout + result.stderr
        
        # Parse result
        for line in output.split('\n'):
            if "[GAME OVER]" in line:
                # Parse: [GAME OVER] money scores: RED=$X, BLUE=$Y
                parts = line.split('$')
                red_score = int(parts[1].split(',')[0])
                blue_score = int(parts[2].strip())
                winner = "RED" if red_score > blue_score else ("BLUE" if blue_score > red_score else "TIE")
                return red_score, blue_score, winner
                
        return 0, 0, "ERROR"
    except Exception as e:
        print(f"Error running game: {e}")
        return 0, 0, "ERROR"

def get_bot_name(path):
    """Extract short name from bot path"""
    return os.path.basename(path).replace(".py", "")

def main():
    print("=" * 60)
    print("ROUND ROBIN TOURNAMENT")
    print("=" * 60)
    print(f"Bots: {len(BOTS)}")
    print(f"Maps: {len(MAPS)}")
    print(f"Total matches: {len(list(combinations(BOTS, 2)))}")
    print("=" * 60)
    
    # Track overall stats
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    total_score = defaultdict(int)
    match_results = []
    
    for map_file in MAPS:
        map_name = os.path.basename(map_file)
        print(f"\n--- Map: {map_name} ---")
        
        for red_bot, blue_bot in combinations(BOTS, 2):
            red_name = get_bot_name(red_bot)
            blue_name = get_bot_name(blue_bot)
            
            print(f"  {red_name} vs {blue_name}...", end=" ", flush=True)
            
            red_score, blue_score, winner = run_game(red_bot, blue_bot, map_file)
            
            total_score[red_name] += red_score
            total_score[blue_name] += blue_score
            
            if winner == "RED":
                wins[red_name] += 1
                losses[blue_name] += 1
                print(f"RED WINS ${red_score} vs ${blue_score}")
            elif winner == "BLUE":
                wins[blue_name] += 1
                losses[red_name] += 1
                print(f"BLUE WINS ${blue_score} vs ${red_score}")
            elif winner == "TIE":
                ties[red_name] += 1
                ties[blue_name] += 1
                print(f"TIE ${red_score}")
            else:
                print(f"ERROR")
            
            match_results.append({
                "map": map_name,
                "red": red_name,
                "blue": blue_name,
                "red_score": red_score,
                "blue_score": blue_score,
                "winner": winner
            })
    
    # Print final standings
    print("\n" + "=" * 60)
    print("FINAL STANDINGS")
    print("=" * 60)
    
    standings = []
    for bot in BOTS:
        name = get_bot_name(bot)
        standings.append({
            "name": name,
            "wins": wins[name],
            "losses": losses[name],
            "ties": ties[name],
            "score": total_score[name],
            "win_rate": wins[name] / max(1, wins[name] + losses[name] + ties[name])
        })
    
    standings.sort(key=lambda x: (-x["wins"], -x["score"]))
    
    print(f"{'Bot':<20} {'W':>4} {'L':>4} {'T':>4} {'WR%':>6} {'Total$':>10}")
    print("-" * 50)
    for s in standings:
        print(f"{s['name']:<20} {s['wins']:>4} {s['losses']:>4} {s['ties']:>4} {s['win_rate']*100:>5.1f}% ${s['score']:>9}")
    
    print("\n" + "=" * 60)
    print(f"WINNER: {standings[0]['name']} with {standings[0]['wins']} wins!")
    print("=" * 60)

if __name__ == "__main__":
    main()
