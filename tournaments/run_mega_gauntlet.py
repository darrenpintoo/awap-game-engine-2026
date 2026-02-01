#!/usr/bin/env python3
"""
Mega Gauntlet Tournament
==============================
Runs a roster of HERO bots against all test bots and a random selection
of other bots on official and revised test maps.

HERO BOTS:
- hareshm/BEST-Hydra-Bot.py
- hareshm/champion_bot.py
- eric/Relay.py
- eric/Relay copy.py
- eric/bot.py
- dpinto/ultimate_champion_bot.py

Outputs to tournament-dashboard/public/data.json for visualization.
"""

import os
import sys
import json
import time
import subprocess
import random
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
BOTS_DIR = ROOT / "bots"
MAPS_DIR = ROOT / "maps"
OUTPUT_JSON = ROOT / "tournament-dashboard" / "public" / "data.json"

# Hero bots (The Gauntlet Runners)
HERO_BOTS = {
    "Hero_BEST-Hydra": str(BOTS_DIR / "hareshm" / "BEST-Hydra-Bot.py"),
    "Hero_Champion": str(BOTS_DIR / "hareshm" / "champion_bot.py"),
    "Hero_Relay": str(BOTS_DIR / "eric" / "Relay.py"),
    "Hero_Relay_Copy": str(BOTS_DIR / "eric" / "Relay copy.py"),
    "Hero_Bot": str(BOTS_DIR / "eric" / "bot.py"),
    "Hero_UltimateChampion": str(BOTS_DIR / "dpinto" / "ultimate_champion_bot.py"),
}

# Directories to pull opponents from
OPPONENT_DIRS = [
    BOTS_DIR / "test-bots",
    BOTS_DIR / "hareshm",
    BOTS_DIR / "eric",
    BOTS_DIR / "dpinto"
]

# Map Directories
OFFICIAL_MAPS_DIR = MAPS_DIR / "official"
TEST_MAPS_REVISED_DIR = MAPS_DIR / "test-maps-revised-for-fairness"


def find_opponents():
    """Find all test bots and a random selection of other bots."""
    opponents = {}
    
    # 1. Add all test bots
    test_bots_dir = BOTS_DIR / "test-bots"
    if test_bots_dir.exists():
        for bot_file in test_bots_dir.glob("*.py"):
            if not bot_file.name.startswith("_"):
                name = f"test-bots/{bot_file.stem}"
                opponents[name] = str(bot_file)
    
    # 2. Add random selection from other dirs (excluding heroes)
    other_bots = []
    hero_paths = set(HERO_BOTS.values())
    
    for d in [BOTS_DIR / "hareshm", BOTS_DIR / "eric", BOTS_DIR / "dpinto"]:
        if d.exists():
            for bot_file in d.glob("*.py"):
                if str(bot_file) not in hero_paths and not bot_file.name.startswith("_"):
                    other_bots.append((f"{d.name}/{bot_file.stem}", str(bot_file)))
    
    # Shuffle and pick 5 random "wildcard" opponents
    random.shuffle(other_bots)
    for name, path in other_bots[:5]:
        opponents[name] = path
        
    return opponents


def run_match(red_path, red_name, blue_path, blue_name, map_path, map_name, map_type):
    """Run a single match and return result."""
    start = time.time()
    try:
        # Enforce 60s timeout per match to keep things moving
        result = subprocess.run(
            [sys.executable, str(SRC / "game.py"), 
             "--red", red_path, "--blue", blue_path, "--map", str(map_path)],
            capture_output=True, text=True, timeout=60
        )
        duration = time.time() - start
        
        output = result.stdout + result.stderr
        
        # Parse scores
        red_score, blue_score = 0, 0
        winner = "ERROR"
        
        for line in output.split('\n'):
            if "money scores" in line:
                m = re.search(r'RED=\$?(-?\d+).*BLUE=\$?(-?\d+)', line)
                if m:
                    red_score = int(m.group(1))
                    blue_score = int(m.group(2))
            if "RESULT" in line:
                if "RED WINS" in line:
                    winner = red_name
                elif "BLUE WINS" in line:
                    winner = blue_name
                elif "DRAW" in line:
                    winner = "DRAW"
        
        # Fallback for timeout/crash that didn't print RESULT
        if winner == "ERROR" and (red_score != 0 or blue_score != 0):
             if red_score > blue_score: winner = red_name
             elif blue_score > red_score: winner = blue_name
             else: winner = "DRAW"

        return {
            "red_name": red_name,
            "blue_name": blue_name,
            "map_name": map_name,
            "map_type": map_type,
            "winner": winner,
            "duration": round(duration, 2),
            "red_score": red_score,
            "blue_score": blue_score,
            "timestamp": time.time()
        }
    except subprocess.TimeoutExpired:
        return {
            "red_name": red_name,
            "blue_name": blue_name,
            "map_name": map_name,
            "map_type": map_type,
            "winner": "TIMEOUT",
            "duration": 60,
            "red_score": 0,
            "blue_score": 0,
            "timestamp": time.time(),
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "red_name": red_name,
            "blue_name": blue_name,
            "map_name": map_name,
            "map_type": map_type,
            "winner": "ERROR",
            "duration": 0,
            "red_score": 0,
            "blue_score": 0,
            "timestamp": time.time(),
            "error": str(e)[:50]
        }


def save_results(results, total_matches):
    """Save results to JSON."""
    data = {
        "tournament_name": "Mega Gauntlet 2026",
        "hero_bots": list(HERO_BOTS.keys()),
        "metadata": {
            "total_matches": total_matches,
            "completed_matches": len(results),
            "last_updated": time.time()
        },
        "matches": results
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("=" * 60)
    print("MEGA GAUNTLET 2026 STARTED")
    print("=" * 60)
    
    # Validate hero bots exist
    valid_heroes = {}
    for name, path in HERO_BOTS.items():
        if not os.path.exists(path):
            print(f"WARNING: Hero bot not found, skipping: {path}")
        else:
            print(f"Hero Loaded: {name}")
            valid_heroes[name] = path
    
    if not valid_heroes:
        print("Error: No valid hero bots found!")
        return

    # Find opponents
    opponents = find_opponents()
    
    # Also reduce hero pool for self-play checks (heroes are also opponents to each other)
    # But for a "Gauntlet", usually heroes play "the field". 
    # Let's add the OTHER heroes as opponents for a given hero.
    
    # Get maps
    official_maps = [(m, m.name, "official") for m in OFFICIAL_MAPS_DIR.glob("*.txt")]
    revised_maps = [(m, m.name, "test_revised") for m in TEST_MAPS_REVISED_DIR.glob("*.txt")]
    all_maps = official_maps + revised_maps
    
    print(f"\nOpponents Pool: {len(opponents)} bots")
    print(f"Map Pool: {len(all_maps)} maps ({len(official_maps)} official, {len(revised_maps)} revised)")
    
    # Schedule matches
    matches = []
    
    # 1. Heroes vs The World (Opponents)
    for hero_name, hero_path in valid_heroes.items():
        for opp_name, opp_path in opponents.items():
            for map_path, map_name, map_type in all_maps:
                # Hero as Red, Opponent as Blue
                matches.append((hero_path, hero_name, opp_path, opp_name, map_path, map_name, map_type))
                # Opponent as Red, Hero as Blue (fairness)
                matches.append((opp_path, opp_name, hero_path, hero_name, map_path, map_name, map_type))
    
    # 2. Heroes vs Heroes (Clash of Titans)
    hero_keys = list(valid_heroes.keys())
    for i in range(len(hero_keys)):
        for j in range(i + 1, len(hero_keys)):
            h1 = hero_keys[i]
            h2 = hero_keys[j]
            for map_path, map_name, map_type in all_maps:
                matches.append((valid_heroes[h1], h1, valid_heroes[h2], h2, map_path, map_name, map_type))
                matches.append((valid_heroes[h2], h2, valid_heroes[h1], h1, map_path, map_name, map_type))

    random.shuffle(matches)
    total = len(matches)
    print(f"Total Scheduled Matches: {total}")
    print("=" * 60 + "\n")
    
    results = []
    
    # Parallel Execution
    # We use a modest number of workers to allow for the game engines to run without starving CPU
    MAX_WORKERS = 10 
    
    print(f"Spinning up {MAX_WORKERS} parallel arenas...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_match, *m): m for m in matches}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Print succinct progress
            winner_display = result["winner"].replace("Hero_", "")[:15]
            map_display = result["map_name"].replace(".txt", "")[:15]
            print(f"[{i}/{total}] {map_display} | {winner_display} WON (${max(result['red_score'], result['blue_score'])})")
            
            # Save periodically (every 10 matches for live updates)
            if i % 10 == 0:
                save_results(results, total)
    
    # Final save
    save_results(results, total)
    print("\n" + "=" * 60)
    print("TOURNAMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
