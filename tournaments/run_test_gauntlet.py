"""
Test Bot Gauntlet Runner (JSON Backend)
=======================================
Runs 'test.py' (Hero_Test) against ALL Challengers.
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

# Configuration
HERO_BOTS = {
    "Hero_Test": get_path("tournaments/test.py")
}

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
    "maps/test-maps-revised-for-fairness/10_endgame_crunch.txt"
]

# Output Path
JSON_OUTPUT_PATH = get_path("tournament-dashboard/public/data.json")
LOG_DIR = get_path("logs")
os.makedirs(LOG_DIR, exist_ok=True)
GAME_SCRIPT = get_path("src/game.py")

def find_all_bots():
    search_pattern = os.path.join(ROOT_DIR, "bots/**/*.py")
    all_files = glob.glob(search_pattern, recursive=True)
    
    challengers = {}
    ignore_terms = ["__init__", "generate", "helper", "util", "constants"]
    
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
        challengers[name] = abs_path
    return challengers

results_log = []
results_lock = threading.Lock()
completed_matches = 0

def run_match_task(red_name, red_path, blue_name, blue_path, map_rel_path, map_type):
    map_abs_path = get_path(map_rel_path)
    if not os.path.exists(red_path) or not os.path.exists(blue_path) or not os.path.exists(map_abs_path):
        return None

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT_DIR) 
        match_data["duration"] = round(time.time() - start_time, 2)
        output = result.stdout + result.stderr
        
        if "RED WINS" in output: match_data["winner"] = red_name
        elif "BLUE WINS" in output: match_data["winner"] = blue_name
        elif "DRAW" in output: match_data["winner"] = "DRAW"
        
        score_match = re.search(r"money scores: RED=\$(-?\d+), BLUE=\$(-?\d+)", output)
        if score_match:
            match_data["red_score"] = int(score_match.group(1))
            match_data["blue_score"] = int(score_match.group(2))
            
    except Exception:
        match_data["winner"] = "TIMEOUT"
        
    return match_data

def save_json(matches, total):
    data = {
        "metadata": {
            "total_matches": total,
            "completed_matches": len(matches),
            "last_updated": time.time()
        },
        "matches": [m for m in matches if m is not None]
    }
    tmp_path = JSON_OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.rename(tmp_path, JSON_OUTPUT_PATH)

def main():
    global completed_matches
    challengers = find_all_bots()
    print("="*60)
    print(f"TEST BOT GAUNTLET: Hero_Test vs {len(challengers)} Challengers")
    print("="*60)
    
    tasks = []
    maps = MAPS_OFFICIAL + MAPS_TEST
    for m in maps:
        m_type = "official" if m in MAPS_OFFICIAL else "test"
        for h, h_path in HERO_BOTS.items():
            for c, c_path in challengers.items():
                tasks.append((h, h_path, c, c_path, m, m_type))
                tasks.append((c, c_path, h, h_path, m, m_type))

    random.shuffle(tasks)
    total_matches = len(tasks)
    workers = 12
    
    save_json([], total_matches)
    
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
    main()
