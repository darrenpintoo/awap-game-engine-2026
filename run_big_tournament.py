"""
Big Round Robin Tournament Runner (LITERAL NAMES)
=================================================
Runs a round robin tournament between 9 specific bots in PARALLEL.
Using literal filenames as bot identifiers.
"""

import subprocess
import sys
import json
import time
from itertools import combinations
import concurrent.futures
import threading
import os

# Configuration
BOTS = {
    # New Variants
    "aggressive_stealer_bot.py": "bots/dpinto/aggressive_stealer_bot.py",
    "super_rush_bot.py": "bots/dpinto/super_rush_bot.py",
    "turtle_defense_bot.py": "bots/dpinto/turtle_defense_bot.py",
    
    # DPinto
    "best_solo_bot_v1.py": "bots/dpinto/best_solo_bot_v1.py",
    "ultimate_champion_bot.py": "bots/dpinto/ultimate_champion_bot.py",
    "champion_sabotage_optimized.py": "bots/dpinto/champion_sabotage_optimized.py",
    
    # Haresh
    "BEST^2-champion_bot.py": "bots/hareshm/BEST^2-champion_bot.py",
    "SOVEREIGN.py": "bots/hareshm/SOVEREIGN.py",
    
    # Eric
    "BEST-IronChefOptimized.py": "bots/eric/BEST-IronChefOptimized.py"
}

MAPS = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

stats = {name: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0} for name in BOTS}
stats_lock = threading.Lock()
completed_matches = 0
total_matches = 0

def run_match_task(red_name, red_path, blue_name, blue_path, map_path):
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_path,
        "--blue", blue_path,
        "--map", map_path,
        "--timeout", "0.2"
    ]
    
    try:
        start_time = time.time()
        # Capture output to avoid interleaving in stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120, 
        )
        duration = time.time() - start_time
        output = result.stdout + result.stderr
        
        if "RED WINS" in output:
            winner = red_name
        elif "BLUE WINS" in output:
            winner = blue_name
        elif "DRAW" in output:
            winner = "DRAW"
        else:
            winner = "ERROR"
            
        return winner, duration, red_name, blue_name, map_path
        
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 120, red_name, blue_name, map_path
    except Exception as e:
        return "ERROR", 0, red_name, blue_name, map_path

def main():
    global completed_matches, total_matches
    
    # Determine workers
    workers = min(10, os.cpu_count() + 2)
    print("="*60)
    print(f"PARALLEL ROUND ROBIN TOURNAMENT ({workers} workers)")
    print(f"Bots: {len(BOTS)} | Maps: {len(MAPS)}")
    print("="*60)
    
    # Generate Schedule
    bot_names = list(BOTS.keys())
    pairs = list(combinations(bot_names, 2))
    total_matches = len(pairs) * 2 * len(MAPS)
    print(f"Total Matches to Run: {total_matches}")
    
    start_all = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        for map_path in MAPS:
            for p1, p2 in pairs:
                # Game 1: p1 vs p2
                futures.append(executor.submit(run_match_task, p1, BOTS[p1], p2, BOTS[p2], map_path))
                # Game 2: p2 vs p1
                futures.append(executor.submit(run_match_task, p2, BOTS[p2], p1, BOTS[p1], map_path))
        
        for future in concurrent.futures.as_completed(futures):
            winner, duration, red, blue, map_name = future.result()
            
            with stats_lock:
                completed_matches += 1
                current_match = completed_matches
                
                # Update Stats
                if winner == red:
                    stats[red]['wins'] += 1; stats[red]['points'] += 3
                    stats[blue]['losses'] += 1
                elif winner == blue:
                    stats[blue]['wins'] += 1; stats[blue]['points'] += 3
                    stats[red]['losses'] += 1
                elif winner == "DRAW":
                    stats[red]['draws'] += 1; stats[red]['points'] += 1
                    stats[blue]['draws'] += 1; stats[blue]['points'] += 1
            
            # Print Progress safely
            map_short = map_name.split('/')[-1]
            
            # Shorten names for log readability
            red_short = red[:15]
            blue_short = blue[:15]
            win_short = winner[:15]
            
            print(f"[{current_match}/{total_matches}] {red_short}.. vs {blue_short}.. ({map_short}) -> {win_short}.. ({duration:.1f}s)")
            sys.stdout.flush()

    total_time = time.time() - start_all
    print("\n" + "="*60)
    print(f"TOURNAMENT RESULTS (Completed in {total_time:.1f}s)")
    print("="*60)
    print(f"{'BOT NAME':<35} | {'PTS':<5} | {'W':<3} {'L':<3} {'D':<3} | {'Win Rate':<8}")
    print("-" * 75)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['points'], reverse=True)
    
    for name, s in sorted_stats:
        total = s['wins'] + s['losses'] + s['draws']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        print(f"{name:<35} | {s['points']:<5} | {s['wins']:<3} {s['losses']:<3} {s['draws']:<3} | {wr:.1f}%")

    with open("big_tournament_results.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
