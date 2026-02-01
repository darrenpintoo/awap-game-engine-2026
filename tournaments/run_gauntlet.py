"""
Gauntlet Tournament Runner: Heroes vs The World
===============================================
Runs 4 Hero Bots against EVERY other bot found in the `bots/` folder.
Hero Bots:
- BEST^3 (Dpinto)
- Ultimate Champion (Dpinto)
- BEST^2 (Haresh)
- SOVEREIGN (Haresh)
"""

import subprocess
import sys
import json
import time
import os
import glob
import concurrent.futures
import threading

# Configuration
HERO_BOTS = {
    "Hero_BEST^3": "bots/dpinto/BEST^3_defense_bot.py",
    "Hero_UltChamp": "bots/dpinto/ultimate_champion_bot.py",
    "Hero_BEST^2": "bots/hareshm/BEST^2-champion_bot.py",
    "Hero_Sovereign": "bots/hareshm/SOVEREIGN.py"
}

MAPS = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

def find_all_bots():
    """Recursively find all .py files in bots/ that act as bots"""
    all_files = glob.glob("bots/**/*.py", recursive=True)
    challengers = {}
    
    ignore_terms = ["__init__", "test", "generate", "helper", "util", "constants"]
    
    for f in all_files:
        if any(term in f.lower() for term in ignore_terms):
            continue
            
        # Also exclude the heroes themselves to avoid self-matches (unless desired, but user said 'against every OTHER bot')
        abs_path = os.path.abspath(f)
        is_hero = False
        for h_path in HERO_BOTS.values():
            if os.path.abspath(h_path) == abs_path:
                is_hero = True
                break
        
        if is_hero:
            continue
            
        name = os.path.basename(f)
        # Handle duplicate filenames in different folders
        if name in challengers:
            folder = os.path.dirname(f).split("/")[-1]
            name = f"{folder}_{name}"
            
        challengers[name] = f
        
    return challengers

stats = {} # Will init in main
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
            timeout=180, # Generous timeout
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
        return "TIMEOUT", 180, red_name, blue_name, map_path
    except Exception:
        return "ERROR", 0, red_name, blue_name, map_path

def main():
    global completed_matches, total_matches
    
    challengers = find_all_bots()
    print("="*60)
    print(f"GAUNTLET TOURNAMENT: 4 Heroes vs {len(challengers)} Challengers")
    print(f"Maps: {len(MAPS)}")
    print("="*60)
    
    # Init Stats
    # We want to track Hero performance primarily, but tracking challenger performance is good too
    all_participants = list(HERO_BOTS.keys()) + list(challengers.keys())
    for p in all_participants:
        stats[p] = {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0, 'games': 0}

    # Generate Tasks
    # Each Hero plays Each Challenger TWICE (once as Red, once as Blue) on ALL maps
    tasks = []
    
    for map_path in MAPS:
        for hero_name, hero_path in HERO_BOTS.items():
            for chal_name, chal_path in challengers.items():
                # Game 1: Hero=Red
                tasks.append((hero_name, hero_path, chal_name, chal_path, map_path))
                # Game 2: Chal=Red
                tasks.append((chal_name, chal_path, hero_name, hero_path, map_path))

    total_matches = len(tasks)
    workers = min(12, os.cpu_count() + 2)
    print(f"Total Matches: {total_matches} | Workers: {workers}")
    
    start_all = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_match_task, *t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            winner, duration, red, blue, map_name = future.result()
            
            with stats_lock:
                completed_matches += 1
                
                # Update Stats (if not error)
                if winner in stats: # winner is a name (red or blue)
                    stats[winner]['wins'] += 1; stats[winner]['points'] += 3
                    
                    loser = blue if winner == red else red
                    if loser in stats:
                         stats[loser]['losses'] += 1
                    
                    if winner in stats: stats[winner]['games'] += 1
                    if loser in stats: stats[loser]['games'] += 1
                    
                elif winner == "DRAW":
                    stats[red]['draws'] += 1; stats[red]['points'] += 1; stats[red]['games'] += 1
                    stats[blue]['draws'] += 1; stats[blue]['points'] += 1; stats[blue]['games'] += 1
            
            # Print periodic updates only, or significant results to avoid clutter
            if completed_matches % 10 == 0 or winner == "ERROR":
                map_short = map_name.split('/')[-1]
                print(f"[{completed_matches}/{total_matches}] {red[:15]} vs {blue[:15]} -> {winner[:15]}")
                sys.stdout.flush()

    total_time = time.time() - start_all
    
    # Filter stats to show HEROES first, top of the leaderboard style
    print("\n" + "="*60)
    print(f"GAUNTLET RESULTS (Completed in {total_time:.1f}s)")
    print("="*60)
    print(f"{'BOT NAME':<30} | {'PTS':<5} | {'W':<3} {'L':<3} {'D':<3} | {'Win Rate':<8}")
    print("-" * 70)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['points'], reverse=True)
    
    for name, s in sorted_stats:
        total = s['games']
        if total == 0: continue
        wr = (s['wins'] / total * 100)
        prefix = "★ " if name in HERO_BOTS else "  "
        print(f"{prefix}{name:<28} | {s['points']:<5} | {s['wins']:<3} {s['losses']:<3} {s['draws']:<3} | {wr:.1f}%")

    with open("gauntlet_results.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
