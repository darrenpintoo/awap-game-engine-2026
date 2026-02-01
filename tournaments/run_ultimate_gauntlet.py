"""
Ultimate Gauntlet Runner (JSON Backend)
=======================================
Runs 3 Hero Bots against ALL Challengers.
Features:
- Validates all file paths before running.
- Restructured logging to `logs/`.
- Atomic JSON updates for React Dashboard.
- Path-aware: Can be run from root or tournaments/ folder.
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
import math

# --- Path Configuration ---
# Determine Root Directory (One level up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR) # Assumes script is in tournaments/

# Helper to get absolute paths
def get_path(rel_path):
    return os.path.join(ROOT_DIR, rel_path)

# Configuration
HERO_BOTS = {
    "Hero_BEST^3": get_path("bots/dpinto/BEST^3_defense_bot.py"),
    "Hero_BEST^2": get_path("bots/hareshm/BEST^2-champion_bot.py"),
    "Hero_Sovereign": get_path("bots/hareshm/SOVEREIGN.py")
}

# Maps (Stored as relative paths for readability in logs, but absolute for game.py arg)
MAPS_OFFICIAL = [
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt"
]

MAPS_TEST = [
    "maps/test/01_tiny_sprint.txt",
    "maps/test/02_balanced_medium.txt",
    "maps/test/03_grand_kitchen.txt",
    "maps/test/04_varied_orders.txt",
    "maps/test/05_chokepoint_chaos.txt",
    "maps/test/06_resource_crunch.txt",
    "maps/test/07_pressure_cooker.txt",
    "maps/test/08_sabotage_alley.txt",
    "maps/test/09_remote_pantry.txt",
    "maps/test/10_burn_risk.txt"
]

# Output Path
JSON_OUTPUT_PATH = get_path("tournament-dashboard/public/data.json")
LOG_DIR = get_path("logs")
os.makedirs(LOG_DIR, exist_ok=True)
GAME_SCRIPT = get_path("src/game.py")

def find_all_bots():
    # Search from Root
    search_pattern = os.path.join(ROOT_DIR, "bots/**/*.py")
    all_files = glob.glob(search_pattern, recursive=True)
    
    challengers = {}
    ignore_terms = ["__init__", "test", "generate", "helper", "util", "constants"]
    
    for f in all_files:
        if any(term in os.path.basename(f).lower() for term in ignore_terms): continue
        
        abs_path = os.path.abspath(f)
        
        if not os.path.exists(abs_path): continue

        is_hero = False
        for h_path in HERO_BOTS.values():
            if os.path.abspath(h_path) == abs_path:
                is_hero = True
                break
        if is_hero: continue
            
        name = os.path.basename(f)
        if name in challengers:
            folder = os.path.dirname(f).split("/")[-1]
            name = f"{folder}_{name}"
        challengers[name] = abs_path # Store Absolute Path
    return challengers

results_log = []
results_lock = threading.Lock()
completed_matches = 0

def run_match_task(red_name, red_path, blue_name, blue_path, map_rel_path, map_type):
    map_abs_path = get_path(map_rel_path)
    
    # Final validation
    if not os.path.exists(red_path) or not os.path.exists(blue_path) or not os.path.exists(map_abs_path):
        return {
            "red_name": red_name, "blue_name": blue_name, 
            "map_name": os.path.basename(map_rel_path), "map_type": map_type,
            "winner": "ERROR", "duration": 0, "red_score": 0, "blue_score": 0,
            "timestamp": time.time(), "error": "FileNotFound"
        }

    # Run Game (CWD must be ROOT for relative imports inside game.py to work correctly? 
    # game.py adds src to path but relies on CWD for some things? 
    # Safest is to set cwd=ROOT_DIR in subprocess)
    
    cmd = [sys.executable, GAME_SCRIPT, "--red", red_path, "--blue", blue_path, "--map", map_abs_path, "--timeout", "0.2"]
    
    match_data = {
        "red_name": red_name, "blue_name": blue_name, 
        "map_name": os.path.basename(map_rel_path), "map_type": map_type,
        "winner": "ERROR", "duration": 0,
        "red_score": 0, "blue_score": 0,
        "timestamp": time.time()
    }
    
    try:
        start_time = time.time()
        # Ensure CWD is ROOT_DIR so game.py finds its resources
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT_DIR) 
        match_data["duration"] = round(time.time() - start_time, 2)
        output = result.stdout + result.stderr
        
        # Parse Winner
        if "RED WINS" in output: match_data["winner"] = red_name
        elif "BLUE WINS" in output: match_data["winner"] = blue_name
        elif "DRAW" in output: match_data["winner"] = "DRAW"
        elif "both bots failed" in output.lower(): match_data["winner"] = "ERROR"
        
        # Parse Scores
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
    global completed_matches
    challengers = find_all_bots()
    print("="*60)
    print(f"ULTIMATE GAUNTLET (ORGANIZED): 3 Heroes vs {len(challengers)} Challengers")
    print(f"Running from: {ROOT_DIR}")
    print("="*60)
    
    tasks = []
    # Schedule
    for m in MAPS_OFFICIAL:
        for h, h_path in HERO_BOTS.items():
            for c, c_path in challengers.items():
                tasks.append((h, h_path, c, c_path, m, "official"))
                tasks.append((c, c_path, h, h_path, m, "official"))

    for m in MAPS_TEST:
        for h, h_path in HERO_BOTS.items():
            for c, c_path in challengers.items():
                tasks.append((h, h_path, c, c_path, m, "test"))
                tasks.append((c, c_path, h, h_path, m, "test"))

    random.shuffle(tasks)
    total_matches = len(tasks)
    workers = min(12, os.cpu_count() + 2)
    
    print(f"Streaming data to: {JSON_OUTPUT_PATH}")
    save_json([], total_matches) # Init empty
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_match_task, *t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            with results_lock:
                results_log.append(res)
                completed_matches += 1
                
                if completed_matches % 10 == 0:
                    print(f"[{completed_matches}/{total_matches}] Saved JSON.")
                    save_json(results_log, total_matches)
                    sys.stdout.flush()

    save_json(results_log, total_matches)
    print("\nTournament Complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        save_json(results_log, total_matches)
        sys.exit(0)
