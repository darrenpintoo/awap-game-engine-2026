#!/usr/bin/env python3
"""
Dual Hero Gauntlet Tournament
==============================
Runs BEST-Hydra-Bot AND eric/Relay.py as heroes against:
- All test bots
- All previous HERO bots (UltimateChampion, BEST^3, Sovereign)
- On official + test maps

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

# Hero bots (the two we're testing)
HERO_BOTS = {
    "Hero_BEST-Hydra": str(BOTS_DIR / "hareshm" / "BEST-Hydra-Bot.py"),
    "Hero_Relay": str(BOTS_DIR / "eric" / "Relay.py"),
}

# Previous hero bots to test against
PREVIOUS_HEROES = {
    "Hero_UltimateChampion": str(BOTS_DIR / "dpinto" / "ultimate_champion_bot.py"),
    "Hero_Sovereign": str(BOTS_DIR / "hareshm" / "SOVEREIGN.py"),
}

# Test bots directory
TEST_BOTS_DIR = BOTS_DIR / "test-bots"

# Maps
OFFICIAL_MAPS = list((MAPS_DIR / "official").glob("*.txt"))
TEST_MAPS = list((MAPS_DIR / "test-maps").glob("*.txt"))


def find_test_bots():
    """Find all test bots."""
    bots = {}
    if TEST_BOTS_DIR.exists():
        for bot_file in TEST_BOTS_DIR.glob("*.py"):
            if not bot_file.name.startswith("_"):
                name = f"test-bots/{bot_file.stem}"
                bots[name] = str(bot_file)
    return bots


def run_match(red_path, red_name, blue_path, blue_name, map_path, map_name, map_type):
    """Run a single match and return result."""
    start = time.time()
    try:
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
        "tournament_name": "Dual Hero Gauntlet (Hydra + Relay)",
        "hero_bots": list(HERO_BOTS.keys()),
        "total_matches": total_matches,
        "completed_matches": len(results),
        "start_time": results[0]["timestamp"] if results else time.time(),
        "last_update": time.time(),
        "matches": results
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("=" * 60)
    print("DUAL HERO GAUNTLET: HYDRA + RELAY")
    print("=" * 60)
    
    # Validate hero bots exist
    for name, path in HERO_BOTS.items():
        if not os.path.exists(path):
            print(f"ERROR: Hero bot not found: {path}")
            return
        print(f"Hero: {name}")
    
    # Find challengers
    test_bots = find_test_bots()
    challengers = {**test_bots, **PREVIOUS_HEROES}
    
    # Also add each hero as a challenger for the other
    for hname, hpath in HERO_BOTS.items():
        challengers[hname] = hpath
    
    # Get maps
    official_maps = [(m, m.name, "official") for m in OFFICIAL_MAPS if m.exists()]
    test_maps = [(m, m.name, "test") for m in TEST_MAPS if m.exists()]
    all_maps = official_maps + test_maps
    
    print(f"\nChallengers: {len(challengers)}")
    print(f"Official maps: {len(official_maps)}")
    print(f"Test maps: {len(test_maps)}")
    
    # Schedule matches
    matches = []
    for hero_name, hero_path in HERO_BOTS.items():
        for cname, cpath in challengers.items():
            if cname == hero_name:
                continue  # Don't play against self
            for map_path, map_name, map_type in all_maps:
                # Hero as red
                matches.append((hero_path, hero_name, cpath, cname, map_path, map_name, map_type))
                # Hero as blue
                matches.append((cpath, cname, hero_path, hero_name, map_path, map_name, map_type))
    
    random.shuffle(matches)
    total = len(matches)
    print(f"Total matches: {total}")
    print("=" * 60 + "\n")
    
    results = []
    hero_wins = {h: 0 for h in HERO_BOTS.keys()}
    hero_matches = {h: 0 for h in HERO_BOTS.keys()}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_match, *m): m for m in matches}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Track hero stats
            for hero in HERO_BOTS.keys():
                if result["red_name"] == hero or result["blue_name"] == hero:
                    hero_matches[hero] += 1
                    if result["winner"] == hero:
                        hero_wins[hero] += 1
            
            # Print progress
            winner_short = result["winner"][:20] if len(result["winner"]) > 20 else result["winner"]
            print(f"[{i}/{total}] {result['red_name'][:15]} vs {result['blue_name'][:15]} on {result['map_name'][:15]} → {winner_short}")
            
            # Save periodically
            if i % 20 == 0:
                save_results(results, total)
                print(f"[SAVED] {i}/{total} matches")
    
    # Final save
    save_results(results, total)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for hero in HERO_BOTS.keys():
        if hero_matches[hero] > 0:
            wr = hero_wins[hero] / hero_matches[hero] * 100
            print(f"{hero}: {hero_wins[hero]}/{hero_matches[hero]} wins ({wr:.1f}%)")
    
    print(f"\nDashboard: http://localhost:5174")


if __name__ == "__main__":
    main()
