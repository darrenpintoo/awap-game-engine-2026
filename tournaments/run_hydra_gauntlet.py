"""
Hydra Gauntlet Runner
======================
Runs Hydra Bot as the SOLE HERO against ALL challengers including RL bots.
Outputs to JSON for React Dashboard.
"""

import subprocess
import sys
import json
import time
import os
import glob
import concurrent.futures
import threading
import random
import re

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def get_path(rel_path):
    return os.path.join(ROOT_DIR, rel_path)

# ============================================================================
# HYDRA BOT AS SOLE HERO
# ============================================================================
HERO_BOTS = {
    "Hero_BEST-Hydra": get_path("bots/hareshm/BEST-Hydra Bot.py"),
}

# All official maps
MAPS_OFFICIAL = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt",
    "maps/official/cooking_for_dummies.txt",
]

# Test maps
MAPS_TEST = [
    "maps/test-maps-revised-for-fairness/01_tiny_sprint.txt",
    "maps/test-maps-revised-for-fairness/02_balanced_medium.txt",
    "maps/test-maps-revised-for-fairness/03_grand_kitchen.txt",
    "maps/test-maps-revised-for-fairness/04_varied_orders.txt",
    "maps/test-maps-revised-for-fairness/05_parallel_paths.txt",
]

# Output
JSON_OUTPUT_PATH = get_path("tournament-dashboard/public/data.json")
LOG_DIR = get_path("logs")
os.makedirs(LOG_DIR, exist_ok=True)
GAME_SCRIPT = get_path("src/game.py")

def find_all_challengers():
    """Find ALL bots including RL bots, excluding only the hero."""
    search_pattern = os.path.join(ROOT_DIR, "bots/**/*.py")
    all_files = glob.glob(search_pattern, recursive=True)
    
    challengers = {}
    ignore_terms = ["__init__", "generate", "helper", "util", "constants"]
    
    hero_paths = [os.path.abspath(p) for p in HERO_BOTS.values()]
    
    for f in all_files:
        basename = os.path.basename(f).lower()
        
        # Skip utility files
        if any(term in basename for term in ignore_terms):
            continue
        
        abs_path = os.path.abspath(f)
        
        # Skip if it's a hero
        if abs_path in hero_paths:
            continue
        
        if not os.path.exists(abs_path):
            continue
        
        # Create unique name
        name = os.path.basename(f).replace(".py", "")
        folder = os.path.basename(os.path.dirname(f))
        
        # Prefix with folder for uniqueness
        full_name = f"{folder}/{name}"
        
        challengers[full_name] = abs_path
    
    return challengers

results_log = []
results_lock = threading.Lock()
completed_matches = 0
total_matches = 0

def run_match_task(red_name, red_path, blue_name, blue_path, map_rel_path, map_type):
    map_abs_path = get_path(map_rel_path)
    
    if not os.path.exists(red_path) or not os.path.exists(blue_path) or not os.path.exists(map_abs_path):
        return {
            "red_name": red_name, "blue_name": blue_name,
            "map_name": os.path.basename(map_rel_path), "map_type": map_type,
            "winner": "ERROR", "duration": 0, "red_score": 0, "blue_score": 0,
            "timestamp": time.time(), "error": "FileNotFound"
        }

    cmd = [sys.executable, GAME_SCRIPT, "--red", red_path, "--blue", blue_path, 
           "--map", map_abs_path, "--timeout", "0.3"]
    
    match_data = {
        "red_name": red_name, "blue_name": blue_name,
        "map_name": os.path.basename(map_rel_path), "map_type": map_type,
        "winner": "ERROR", "duration": 0,
        "red_score": 0, "blue_score": 0,
        "timestamp": time.time()
    }
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=ROOT_DIR)
        match_data["duration"] = round(time.time() - start_time, 2)
        output = result.stdout + result.stderr
        
        if "RED WINS" in output:
            match_data["winner"] = red_name
        elif "BLUE WINS" in output:
            match_data["winner"] = blue_name
        elif "DRAW" in output:
            match_data["winner"] = "DRAW"
        elif "both bots failed" in output.lower():
            match_data["winner"] = "ERROR"
        
        score_match = re.search(r"money scores: RED=\$(-?\d+), BLUE=\$(-?\d+)", output)
        if score_match:
            match_data["red_score"] = int(score_match.group(1))
            match_data["blue_score"] = int(score_match.group(2))
            
    except subprocess.TimeoutExpired:
        match_data["winner"] = "TIMEOUT"
        
    return match_data

def save_json(matches, total):
    data = {
        "metadata": {
            "tournament_name": "Hydra Gauntlet",
            "hero_bot": "BEST-Hydra Bot",
            "total_matches": total,
            "completed_matches": len(matches),
            "last_updated": time.time()
        },
        "matches": matches
    }
    
    tmp_path = JSON_OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.rename(tmp_path, JSON_OUTPUT_PATH)

def main():
    global completed_matches, total_matches
    
    challengers = find_all_challengers()
    
    print("=" * 70)
    print("🐉 HYDRA GAUNTLET: 1 Hero vs ALL Challengers (inc. RL bots)")
    print("=" * 70)
    print(f"Hero: BEST-Hydra Bot")
    print(f"Challengers: {len(challengers)}")
    for name in sorted(challengers.keys()):
        print(f"  - {name}")
    print(f"Maps: {len(MAPS_OFFICIAL)} official + {len(MAPS_TEST)} test")
    print("=" * 70)
    
    tasks = []
    
    # Schedule all matches
    all_maps = [(m, "official") for m in MAPS_OFFICIAL] + [(m, "test") for m in MAPS_TEST]
    
    for map_path, map_type in all_maps:
        for hero_name, hero_path in HERO_BOTS.items():
            for challenger_name, challenger_path in challengers.items():
                # Hero as RED
                tasks.append((hero_name, hero_path, challenger_name, challenger_path, map_path, map_type))
                # Hero as BLUE
                tasks.append((challenger_name, challenger_path, hero_name, hero_path, map_path, map_type))
    
    random.shuffle(tasks)
    total_matches = len(tasks)
    workers = min(8, os.cpu_count() or 4)
    
    print(f"\nTotal matches: {total_matches}")
    print(f"Workers: {workers}")
    print(f"Output: {JSON_OUTPUT_PATH}")
    print("=" * 70 + "\n")
    
    save_json([], total_matches)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_match_task, *t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            with results_lock:
                results_log.append(res)
                completed_matches += 1
                
                if completed_matches % 5 == 0 or completed_matches == total_matches:
                    pct = 100 * completed_matches / total_matches
                    w = sum(1 for r in results_log if "Hero" in r.get("winner", ""))
                    l = sum(1 for r in results_log if r.get("winner", "") not in ["DRAW", "ERROR", "TIMEOUT", ""] and "Hero" not in r.get("winner", ""))
                    d = sum(1 for r in results_log if r.get("winner") == "DRAW")
                    print(f"[{completed_matches}/{total_matches}] {pct:.0f}% | W:{w} L:{l} D:{d}")
                    save_json(results_log, total_matches)
                    sys.stdout.flush()

    save_json(results_log, total_matches)
    
    # Final stats
    wins = sum(1 for r in results_log if "Hero" in r.get("winner", ""))
    losses = sum(1 for r in results_log if r.get("winner", "") not in ["DRAW", "ERROR", "TIMEOUT", ""] and "Hero" not in r.get("winner", ""))
    draws = sum(1 for r in results_log if r.get("winner") == "DRAW")
    
    print("\n" + "=" * 70)
    print(f"🏆 HYDRA GAUNTLET COMPLETE")
    print(f"   Wins: {wins} | Losses: {losses} | Draws: {draws}")
    print(f"   Win Rate: {100*wins/(wins+losses+draws):.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if total_matches > 0:
            save_json(results_log, total_matches)
        print("\nInterrupted - partial results saved")
        sys.exit(0)
