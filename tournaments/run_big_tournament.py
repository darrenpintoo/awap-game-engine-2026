"""
Big Round Robin Tournament Runner (EXTENDED STATS EDITION)
==========================================================
Runs a round robin tournament between 9 specific bots in PARALLEL.
Includes detailed statistical reporting.
"""

import subprocess
import sys
import json
import time
from itertools import combinations
import concurrent.futures
import threading
import os
import collections

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
    "BEST^3_defense_bot.py": "bots/dpinto/BEST^3_defense_bot.py",
    
    # Haresh
    "BEST^2-champion_bot.py": "bots/hareshm/BEST^2-champion_bot.py",
    "SOVEREIGN.py": "bots/hareshm/SOVEREIGN.py",
    
    # Eric
    "BEST-IronChefOptimized.py": "bots/eric/BEST-IronChefOptimized.py"
}

# Ensure we have the list
BOT_NAMES = list(BOTS.keys())

MAPS = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

# stats structure:
# {
#   bot_name: {
#     'wins': 0, 'losses': 0, 'draws': 0, 'points': 0,
#     'matches': 0,
#     'side': {'red': {'w':0, 'l':0, 'd':0}, 'blue': {'w':0, 'l':0, 'd':0}},
#     'vs': { opponent: {'w':0, 'l':0, 'd':0} },
#     'maps': { mapname: {'w':0, 'l':0, 'd':0} }
#   }
# }

def init_stats():
    s = {}
    for b in BOT_NAMES:
        s[b] = {
            'wins': 0, 'losses': 0, 'draws': 0, 'points': 0, 'matches': 0,
            'side': {'red': {'w':0, 'l':0, 'd':0}, 'blue': {'w':0, 'l':0, 'd':0}},
            'vs': {op: {'w':0, 'l':0, 'd':0} for op in BOT_NAMES if op != b},
            'maps': {m.split('/')[-1]: {'w':0, 'l':0, 'd':0} for m in MAPS}
        }
    return s

stats = init_stats()
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

def update_stats(winner, red, blue, map_path):
    map_name = map_path.split('/')[-1]
    
    with stats_lock:
        global completed_matches
        completed_matches += 1
        
        # Helper to update a single bot's stats
        def add_result(p, result, opponent, side):
            if p not in stats: return
            stats[p]['matches'] += 1
            entry = 'w' if result == 'win' else ('l' if result == 'loss' else 'd')
            
            # Global
            if result == 'win': stats[p]['wins'] += 1; stats[p]['points'] += 3
            elif result == 'loss': stats[p]['losses'] += 1
            else: stats[p]['draws'] += 1; stats[p]['points'] += 1
            
            # Side
            stats[p]['side'][side][entry] += 1
            
            # Vs
            if opponent in stats[p]['vs']:
                stats[p]['vs'][opponent][entry] += 1
                
            # Maps
            if map_name in stats[p]['maps']:
                stats[p]['maps'][map_name][entry] += 1

        if winner == red:
            add_result(red, 'win', blue, 'red')
            add_result(blue, 'loss', red, 'blue')
        elif winner == blue:
            add_result(blue, 'win', red, 'blue')
            add_result(red, 'loss', blue, 'red')
        elif winner == "DRAW":
            add_result(red, 'draw', blue, 'red')
            add_result(blue, 'draw', red, 'blue')

def print_report():
    print("\n" + "="*80)
    print("EXTENDED TOURNAMENT REPORT")
    print("="*80)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['points'], reverse=True)
    
    # 1. Main Leaderboard
    print(f"\nLEADERBOARD")
    print("-" * 80)
    print(f"{'BOT NAME':<30} | {'PTS':<5} | {'W':<3} {'L':<3} {'D':<3} | {'Win Rate':<8}")
    print("-" * 80)
    for name, s in sorted_stats:
        total = s['matches']
        wr = (s['wins'] / total * 100) if total > 0 else 0
        print(f"{name:<30} | {s['points']:<5} | {s['wins']:<3} {s['losses']:<3} {s['draws']:<3} | {wr:.1f}%")

    # 2. Side Balance
    print(f"\nSIDE BALANCE (Red vs Blue)")
    print("-" * 80)
    print(f"{'BOT NAME':<30} | {'Red WR':<10} | {'Blue WR':<10}")
    print("-" * 80)
    for name, s in sorted_stats:
        red = s['side']['red']
        blue = s['side']['blue']
        
        red_total = red['w'] + red['l'] + red['d']
        blue_total = blue['w'] + blue['l'] + blue['d']
        
        red_wr = (red['w']/red_total*100) if red_total > 0 else 0.0
        blue_wr = (blue['w']/blue_total*100) if blue_total > 0 else 0.0
        
        print(f"{name:<30} | {red_wr:5.1f}%     | {blue_wr:5.1f}%")

    # 3. Map Performance
    pp_maps = [m.split('/')[-1] for m in MAPS]
    print(f"\nMAP WIN RATES")
    print("-" * 80)
    header = f"{'BOT NAME':<25}" + "".join([f"{m[:8]:<10}" for m in pp_maps])
    print(header)
    print("-" * 80)
    for name, s in sorted_stats:
        line = f"{name[:24]:<25}"
        for m in pp_maps:
             ms = s['maps'].get(m, {'w':0, 'l':0, 'd':0})
             tot = ms['w'] + ms['l'] + ms['d']
             wr = (ms['w'] / tot * 100) if tot > 0 else 0
             line += f"{wr:5.1f}%    "
        print(line)

    # 4. Head to Head Matrix
    print(f"\nHEAD-TO-HEAD MATRIX (Row vs Column: Win/Loss/Draw)")
    print("-" * 120)
    
    # Map names to short index
    short_names = [n[:6] for n, _ in sorted_stats]
    print(f"{'':<25} | " + " | ".join([f"{n:<6}" for n in short_names]))
    print("-" * 120)

    for name_row, s_row in sorted_stats:
        line = f"{name_row[:24]:<25} | "
        for name_col, _ in sorted_stats:
            if name_row == name_col:
                line += f"{'-':<6} | "
            else:
                vs = s_row['vs'].get(name_col, {'w':0})
                # record as W-L-D from row perspective
                # To get loss/draw, we need to correct tracking or deduce
                valid = s_row['vs'].get(name_col, {'w':0, 'l':0, 'd':0})
                rec = f"{valid['w']}-{valid['l']}-{valid['d']}"
                line += f"{rec:<6} | "
        print(line)


def main():
    global completed_matches, total_matches
    
    workers = min(10, os.cpu_count() + 2)
    print("="*60)
    print(f"PARALLEL ROUND ROBIN (EXTENDED) ({workers} workers)")
    print(f"Bots: {len(BOTS)} | Maps: {len(MAPS)}")
    print("="*60)
    
    # Generate Schedule
    pairs = list(combinations(BOT_NAMES, 2))
    total_matches = len(pairs) * 2 * len(MAPS)
    print(f"Total Matches to Run: {total_matches}")
    
    start_all = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        for map_path in MAPS:
            for p1, p2 in pairs:
                futures.append(executor.submit(run_match_task, p1, BOTS[p1], p2, BOTS[p2], map_path))
                futures.append(executor.submit(run_match_task, p2, BOTS[p2], p1, BOTS[p1], map_path))
        
        for future in concurrent.futures.as_completed(futures):
            winner, duration, red, blue, map_path = future.result()
            update_stats(winner, red, blue, map_path)
            
            # Short status
            map_short = map_path.split('/')[-1]
            print(f"[{completed_matches}/{total_matches}] {red[:10]}.. vs {blue[:10]}.. ({map_short}) -> {winner[:10]}..")
            sys.stdout.flush()

    total_time = time.time() - start_all
    print(f"\nTournament Completed in {total_time:.1f}s")
    
    print_report()

    # Save raw
    with open("big_tournament_results_extended.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
