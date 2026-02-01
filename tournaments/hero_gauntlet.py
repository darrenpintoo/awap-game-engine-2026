"""
Hero Gauntlet Runner
====================
Runs specified Hero Bots against ALL bots in test-bots/.
Outputs JSON compatible with the tournament-dashboard.
"""

import subprocess
import sys
import json
import time
import os
import glob
import concurrent.futures
import threading
import re

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def get_path(rel_path):
    return os.path.join(ROOT_DIR, rel_path)

# Configuration: Hero Bots (display name -> path)
HERO_BOTS = {
    "Hero_BEST-Hydra": get_path("bots/hareshm/BEST-Hydra-Bot.py"),
    "Hero_Relay": get_path("bots/eric/Relay.py"),
    "Hero_Champion": get_path("bots/hareshm/champion_bot.py"),
}

# Maps to use
MAPS_OFFICIAL = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

MAPS_TEST = [
    "maps/test-maps-revised-for-fairness/01_tiny_sprint.txt",
    "maps/test-maps-revised-for-fairness/02_balanced_medium.txt",
    "maps/test-maps-revised-for-fairness/03_grand_kitchen.txt",
    "maps/test-maps-revised-for-fairness/04_varied_orders.txt",
    "maps/test-maps-revised-for-fairness/05_parallel_paths.txt",
    "maps/test-maps-revised-for-fairness/06_high_throughput.txt",
    "maps/test-maps-revised-for-fairness/07_tight_timing.txt",
    "maps/test-maps-revised-for-fairness/08_resource_sharing.txt",
    "maps/test-maps-revised-for-fairness/09_sabotage_ready.txt",
    "maps/test-maps-revised-for-fairness/10_endgame_crunch.txt"
]

# Output paths
JSON_OUTPUT_PATH = get_path("tournament-dashboard/public/data.json")
GAME_SCRIPT = get_path("src/game.py")

def find_test_bots():
    """Find all bots in bots/test-bots/"""
    search_pattern = os.path.join(ROOT_DIR, "bots/test-bots/*.py")
    all_files = glob.glob(search_pattern)
    
    challengers = {}
    ignore_terms = ["__init__", "helper", "util", "constants"]
    
    for f in all_files:
        basename = os.path.basename(f)
        if any(term in basename.lower() for term in ignore_terms):
            continue
        
        abs_path = os.path.abspath(f)
        if not os.path.exists(abs_path):
            continue
        
        # Use clean name without .py
        name = basename.replace(".py", "")
        challengers[name] = abs_path
    
    return challengers

# Thread-safe results storage
results_log = []
results_lock = threading.Lock()
completed_matches = 0
total_matches = 0

def run_match_task(red_name, red_path, blue_name, blue_path, map_rel_path, map_type):
    """Run a single match and return the result."""
    map_abs_path = get_path(map_rel_path)
    
    # Validate paths exist
    if not os.path.exists(red_path):
        return {
            "red_name": red_name, "blue_name": blue_name,
            "map_name": os.path.basename(map_rel_path), "map_type": map_type,
            "winner": "ERROR", "duration": 0, "red_score": 0, "blue_score": 0,
            "timestamp": time.time(), "error": f"Red bot not found: {red_path}"
        }
    if not os.path.exists(blue_path):
        return {
            "red_name": red_name, "blue_name": blue_name,
            "map_name": os.path.basename(map_rel_path), "map_type": map_type,
            "winner": "ERROR", "duration": 0, "red_score": 0, "blue_score": 0,
            "timestamp": time.time(), "error": f"Blue bot not found: {blue_path}"
        }
    if not os.path.exists(map_abs_path):
        return {
            "red_name": red_name, "blue_name": blue_name,
            "map_name": os.path.basename(map_rel_path), "map_type": map_type,
            "winner": "ERROR", "duration": 0, "red_score": 0, "blue_score": 0,
            "timestamp": time.time(), "error": f"Map not found: {map_abs_path}"
        }

    cmd = [
        sys.executable, GAME_SCRIPT,
        "--red", red_path,
        "--blue", blue_path,
        "--map", map_abs_path,
        "--timeout", "0.2"
    ]
    
    match_data = {
        "red_name": red_name,
        "blue_name": blue_name,
        "map_name": os.path.basename(map_rel_path),
        "map_type": map_type,
        "winner": "ERROR",
        "duration": 0,
        "red_score": 0,
        "blue_score": 0,
        "timestamp": time.time()
    }
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=ROOT_DIR
        )
        match_data["duration"] = round(time.time() - start_time, 2)
        output = result.stdout + result.stderr
        
        # Parse Winner
        if "RED WINS" in output:
            match_data["winner"] = red_name
        elif "BLUE WINS" in output:
            match_data["winner"] = blue_name
        elif "DRAW" in output:
            match_data["winner"] = "DRAW"
        elif "both bots failed" in output.lower():
            match_data["winner"] = "ERROR"
        
        # Parse Scores
        score_match = re.search(r"money scores: RED=\$(-?\d+), BLUE=\$(-?\d+)", output)
        if score_match:
            match_data["red_score"] = int(score_match.group(1))
            match_data["blue_score"] = int(score_match.group(2))
            
    except subprocess.TimeoutExpired:
        match_data["winner"] = "TIMEOUT"
        match_data["duration"] = 120
    except Exception as e:
        match_data["winner"] = "ERROR"
        match_data["error"] = str(e)
        
    return match_data

def save_json(matches, total):
    """Atomically save results to JSON file."""
    data = {
        "metadata": {
            "total_matches": total,
            "completed_matches": len(matches),
            "last_updated": time.time()
        },
        "matches": matches
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    
    # Atomic write
    tmp_path = JSON_OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    
    # On Windows, we need to remove the target file first if it exists
    if os.path.exists(JSON_OUTPUT_PATH):
        os.remove(JSON_OUTPUT_PATH)
    os.rename(tmp_path, JSON_OUTPUT_PATH)

def main():
    global completed_matches, total_matches
    
    # Validate hero bots exist
    print("=" * 60)
    print("HERO GAUNTLET: Validating Bots")
    print("=" * 60)
    
    valid_heroes = {}
    for name, path in HERO_BOTS.items():
        if os.path.exists(path):
            print(f"  [OK] {name}: {path}")
            valid_heroes[name] = path
        else:
            print(f"  [MISSING] {name}: {path}")
    
    if not valid_heroes:
        print("\nERROR: No valid hero bots found!")
        return
    
    # Find challengers
    challengers = find_test_bots()
    print(f"\nFound {len(challengers)} challenger bots in test-bots/:")
    for name, path in challengers.items():
        print(f"  - {name}")
    
    if not challengers:
        print("\nERROR: No challenger bots found!")
        return
    
    # Build task list
    # Each hero plays each challenger twice (once as red, once as blue) on each map
    tasks = []
    
    for map_path in MAPS_OFFICIAL:
        for hero_name, hero_path in valid_heroes.items():
            for chal_name, chal_path in challengers.items():
                # Hero as Red
                tasks.append((hero_name, hero_path, chal_name, chal_path, map_path, "official"))
                # Challenger as Red
                tasks.append((chal_name, chal_path, hero_name, hero_path, map_path, "official"))
    
    for map_path in MAPS_TEST:
        for hero_name, hero_path in valid_heroes.items():
            for chal_name, chal_path in challengers.items():
                # Hero as Red
                tasks.append((hero_name, hero_path, chal_name, chal_path, map_path, "test"))
                # Challenger as Red
                tasks.append((chal_name, chal_path, hero_name, hero_path, map_path, "test"))
    
    total_matches = len(tasks)
    workers = min(8, (os.cpu_count() or 4) + 2)
    
    print("\n" + "=" * 60)
    print(f"HERO GAUNTLET: {len(valid_heroes)} Heroes vs {len(challengers)} Challengers")
    print(f"Maps: {len(MAPS_OFFICIAL)} official + {len(MAPS_TEST)} test = {len(MAPS_OFFICIAL) + len(MAPS_TEST)} total")
    print(f"Total Matches: {total_matches}")
    print(f"Workers: {workers}")
    print(f"Output: {JSON_OUTPUT_PATH}")
    print("=" * 60)
    
    # Initialize empty JSON
    save_json([], total_matches)
    print(f"\nStarting tournament...")
    
    start_all = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_match_task, *t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            
            with results_lock:
                results_log.append(res)
                completed_matches += 1
                
                # Progress update every 10 matches
                if completed_matches % 10 == 0 or completed_matches == total_matches:
                    elapsed = time.time() - start_all
                    rate = completed_matches / elapsed if elapsed > 0 else 0
                    eta = (total_matches - completed_matches) / rate if rate > 0 else 0
                    print(f"[{completed_matches}/{total_matches}] ({completed_matches*100//total_matches}%) - {rate:.1f} matches/sec - ETA: {eta:.0f}s")
                    save_json(results_log, total_matches)
                    sys.stdout.flush()
    
    # Final save
    save_json(results_log, total_matches)
    
    total_time = time.time() - start_all
    print("\n" + "=" * 60)
    print(f"GAUNTLET COMPLETE!")
    print(f"Duration: {total_time:.1f}s")
    print(f"Matches: {completed_matches}")
    print(f"Output saved to: {JSON_OUTPUT_PATH}")
    print("=" * 60)
    
    # Print summary
    hero_stats = {h: {"w": 0, "l": 0, "d": 0, "e": 0} for h in valid_heroes}
    for m in results_log:
        for hero in valid_heroes:
            if m["red_name"] == hero:
                if m["winner"] == hero:
                    hero_stats[hero]["w"] += 1
                elif m["winner"] == "DRAW":
                    hero_stats[hero]["d"] += 1
                elif m["winner"] in ["ERROR", "TIMEOUT"]:
                    hero_stats[hero]["e"] += 1
                else:
                    hero_stats[hero]["l"] += 1
            elif m["blue_name"] == hero:
                if m["winner"] == hero:
                    hero_stats[hero]["w"] += 1
                elif m["winner"] == "DRAW":
                    hero_stats[hero]["d"] += 1
                elif m["winner"] in ["ERROR", "TIMEOUT"]:
                    hero_stats[hero]["e"] += 1
                else:
                    hero_stats[hero]["l"] += 1
    
    print("\nHERO PERFORMANCE:")
    print("-" * 50)
    for hero, stats in sorted(hero_stats.items(), key=lambda x: x[1]["w"], reverse=True):
        total = stats["w"] + stats["l"] + stats["d"]
        wr = (stats["w"] / total * 100) if total > 0 else 0
        print(f"  {hero}: {stats['w']}W / {stats['l']}L / {stats['d']}D ({wr:.1f}% WR)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        save_json(results_log, total_matches)
        print(f"Partial results saved to: {JSON_OUTPUT_PATH}")
        sys.exit(0)
