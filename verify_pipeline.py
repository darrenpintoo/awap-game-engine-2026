"""
Pipeline Verification Script
----------------------------
Runs 1 singular match.
Writes data to 'tournament-dashboard/public/data.json'.
Verifies the file exists and has content.
"""
import sys
import os
import json
import time
import subprocess
import re

OUTPUT_PATH = "tournament-dashboard/public/data.json"

def run_test_match():
    print("Running test match...")
    red = "bots/dpinto/BEST^3_defense_bot.py"
    blue = "bots/hareshm/SOVEREIGN.py"
    m = "maps/official/v1.txt"
    
    cmd = [sys.executable, "src/game.py", "--red", red, "--blue", blue, "--map", m, "--timeout", "0.2"]
    
    start_time = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    output = res.stdout + res.stderr
    print(f"Match finished in {duration:.2f}s")
    
    # Parse Score
    red_score = 0
    blue_score = 0
    score_match = re.search(r"money scores: RED=\$(-?\d+), BLUE=\$(-?\d+)", output)
    if score_match:
        red_score = int(score_match.group(1))
        blue_score = int(score_match.group(2))
        print(f"Scores found: Red={red_score}, Blue={blue_score}")
    else:
        print("WARNING: No scores found in output!")

    match_data = {
        "red_name": "Hero_BEST^3", "blue_name": "Hero_Sovereign",
        "map_name": "v1.txt", "map_type": "official",
        "winner": "Hero_Sovereign" if "BLUE WINS" in output else "Hero_BEST^3",
        "duration": duration,
        "red_score": red_score, "blue_score": blue_score,
        "timestamp": time.time()
    }
    
    data = {
        "metadata": {
            "total_matches": 1, 
            "completed_matches": 1, 
            "last_updated": time.time()
        },
        "matches": [match_data]
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)
    
    print(f"Wrote JSON to {OUTPUT_PATH}")
    
    # Verify
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        print("SUCCESS: JSON file created and is not empty.")
        with open(OUTPUT_PATH) as f:
            print("Content snippet:", f.read()[:200])
    else:
        print("FAILURE: JSON file missing or empty.")

if __name__ == "__main__":
    run_test_match()
