"""
Big Round Robin Tournament Runner
=================================
Runs a round robin tournament between:
- New Bots: Aggressive Stealer, Super Rush, Turtle Defense
- Existing Bots: DPinto Planner, Eric IronChefOptimized, Haresh Champion
On official competition maps.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from itertools import combinations

# Configuration
BOTS = {
    "Stealer": "bots/dpinto/aggressive_stealer_bot.py",
    "Rush": "bots/dpinto/super_rush_bot.py",
    "Turtle": "bots/dpinto/turtle_defense_bot.py",
    "Planner": "bots/dpinto/planner_bot.py",
    "IronChef": "bots/eric/IronChefOptimized.py",
    "HareshChamp": "bots/hareshm/BEST-champion_bot.py"
}

# Official competition maps
MAPS = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

def run_match(red_name, red_path, blue_name, blue_path, map_path):
    print(f"MATCH: {red_name} (Red) vs {blue_name} (Blue) on {map_path}")
    
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_path,
        "--blue", blue_path,
        "--blue", blue_path,
        "--map", map_path,
        "--timeout", "0.2"
    ]
    
    try:
        # Run with timeout to prevent hangs
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120, # 2 minutes max per game
        )
        duration = time.time() - start_time
        
        output = result.stdout + result.stderr
        
        output = result.stdout + result.stderr
        
        if "RED WINS" in output:
            winner = red_name
        elif "BLUE WINS" in output:
            winner = blue_name
        elif "DRAW" in output:
            winner = "DRAW"
        else:
            print(f"  -> ERROR: Output snippet:\n{output[-500:]}\n")
            winner = "ERROR"
            
        print(f"  -> Winner: {winner} ({duration:.1f}s)")
        return winner
        
    except subprocess.TimeoutExpired:
        print("  -> TIMEOUT")
        return "TIMEOUT"
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return "ERROR"

def main():
    print("="*60)
    print("BIG ROUND ROBIN TOURNAMENT")
    print("="*60)
    
    # Initialize Structure
    # results[bot_name] = {wins, losses, draws, points}
    stats = {name: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0} for name in BOTS}
    match_log = []
    
    # Generate Schedule (Round Robin)
    # Each pair plays TWICE (swapping Red/Blue sides) on ALL maps
    bot_names = list(BOTS.keys())
    pairs = list(combinations(bot_names, 2))
    
    total_matches = len(pairs) * 2 * len(MAPS)
    match_count = 0
    
    print(f"Total Matches to Run: {total_matches}")
    
    for map_path in MAPS:
        print(f"\n--- MAP: {map_path} ---")
        for p1, p2 in pairs:
            # Game 1: p1=Red, p2=Blue
            match_count += 1
            print(f"[{match_count}/{total_matches}] ", end="")
            w1 = run_match(p1, BOTS[p1], p2, BOTS[p2], map_path)
            
            if w1 == p1:
                stats[p1]['wins'] += 1; stats[p1]['points'] += 3
                stats[p2]['losses'] += 1
            elif w1 == p2:
                stats[p2]['wins'] += 1; stats[p2]['points'] += 3
                stats[p1]['losses'] += 1
            elif w1 == "DRAW":
                stats[p1]['draws'] += 1; stats[p1]['points'] += 1
                stats[p2]['draws'] += 1; stats[p2]['points'] += 1
            
            # Game 2: p2=Red, p1=Blue
            match_count += 1
            print(f"[{match_count}/{total_matches}] ", end="")
            w2 = run_match(p2, BOTS[p2], p1, BOTS[p1], map_path)
            
            if w2 == p2:
                stats[p2]['wins'] += 1; stats[p2]['points'] += 3
                stats[p1]['losses'] += 1
            elif w2 == p1:
                stats[p1]['wins'] += 1; stats[p1]['points'] += 3
                stats[p2]['losses'] += 1
            elif w2 == "DRAW":
                stats[p2]['draws'] += 1; stats[p2]['points'] += 1
                stats[p1]['draws'] += 1; stats[p1]['points'] += 1

    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    print(f"{'BOT NAME':<15} | {'PTS':<5} | {'W':<3} {'L':<3} {'D':<3} | {'Win Rate':<8}")
    print("-" * 60)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['points'], reverse=True)
    
    for name, s in sorted_stats:
        total = s['wins'] + s['losses'] + s['draws']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        print(f"{name:<15} | {s['points']:<5} | {s['wins']:<3} {s['losses']:<3} {s['draws']:<3} | {wr:.1f}%")

    # Save details
    with open("big_tournament_results.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
