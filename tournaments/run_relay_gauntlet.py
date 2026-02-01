#!/usr/bin/env python3
"""
Relay Bot Gauntlet Tournament
==============================
Runs eric/Relay.py as the hero against all other bots.
Outputs results to tournament-dashboard/public/data.json for visualization.
"""

import os
import sys
import json
import time
import subprocess
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Paths
ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
BOTS_DIR = ROOT / "bots"
MAPS_DIR = ROOT / "maps"
OUTPUT_JSON = ROOT / "tournament-dashboard" / "public" / "data.json"

# Hero bot
HERO_BOT = BOTS_DIR / "eric" / "Relay.py"
HERO_NAME = "Hero_Relay"

# Official maps
OFFICIAL_MAPS = [
    MAPS_DIR / "official" / "chopped.txt",
    MAPS_DIR / "official" / "v1.txt",
    MAPS_DIR / "official" / "orbit.txt",
    MAPS_DIR / "official" / "small_wall.txt",
    MAPS_DIR / "official" / "throughput.txt",
]

def find_all_bots():
    """Find all challenger bots (excluding the hero)."""
    bots = {}
    
    for folder in BOTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        for bot_file in folder.glob("*.py"):
            if bot_file.name.startswith("_") or bot_file.name.startswith("."):
                continue
            # Skip the hero bot itself
            if bot_file.resolve() == HERO_BOT.resolve():
                continue
            
            name = f"{folder.name}/{bot_file.stem}"
            bots[name] = str(bot_file)
    
    return bots


def run_match(hero_path, hero_name, challenger_path, challenger_name, map_path, map_name, hero_as_red=True):
    """Run a single match and return result."""
    if hero_as_red:
        red_path, red_name = hero_path, hero_name
        blue_path, blue_name = challenger_path, challenger_name
    else:
        red_path, red_name = challenger_path, challenger_name
        blue_path, blue_name = hero_path, hero_name
    
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
                import re
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
            "map_type": "official",
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
            "map_type": "official",
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
        "tournament_name": "Relay Gauntlet",
        "hero_bots": [HERO_NAME],
        "total_matches": total_matches,
        "completed_matches": len(results),
        "start_time": results[0]["timestamp"] if results else time.time(),
        "last_update": time.time(),
        "matches": results
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[SAVED] {len(results)} matches to {OUTPUT_JSON}")


def main():
    print("=" * 60)
    print("RELAY BOT GAUNTLET TOURNAMENT")
    print("=" * 60)
    
    if not HERO_BOT.exists():
        print(f"ERROR: Hero bot not found: {HERO_BOT}")
        return
    
    challengers = find_all_bots()
    maps = [m for m in OFFICIAL_MAPS if m.exists()]
    
    print(f"Hero: {HERO_NAME} ({HERO_BOT})")
    print(f"Challengers: {len(challengers)}")
    print(f"Maps: {len(maps)}")
    
    # Schedule matches: hero vs each challenger on each map (both sides)
    matches = []
    for cname, cpath in challengers.items():
        for map_path in maps:
            map_name = map_path.name
            matches.append((str(HERO_BOT), HERO_NAME, cpath, cname, map_path, map_name, True))
            matches.append((str(HERO_BOT), HERO_NAME, cpath, cname, map_path, map_name, False))
    
    random.shuffle(matches)
    total = len(matches)
    print(f"Total matches: {total}")
    print("=" * 60)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_match, *m): m for m in matches}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Print progress
            hero_won = result["winner"] == HERO_NAME
            status = "✓" if hero_won else ("=" if result["winner"] == "DRAW" else "✗")
            opponent = result["blue_name"] if result["red_name"] == HERO_NAME else result["red_name"]
            print(f"[{i}/{total}] {status} vs {opponent} on {result['map_name']} | {result['red_score']}-{result['blue_score']}")
            
            # Save periodically
            if i % 10 == 0:
                save_results(results, total)
    
    # Final save
    save_results(results, total)
    
    # Print summary
    wins = sum(1 for r in results if r["winner"] == HERO_NAME)
    draws = sum(1 for r in results if r["winner"] == "DRAW")
    losses = total - wins - draws
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Wins:   {wins} ({wins/total*100:.1f}%)")
    print(f"Draws:  {draws} ({draws/total*100:.1f}%)")
    print(f"Losses: {losses} ({losses/total*100:.1f}%)")
    print(f"\nDashboard: http://localhost:5174")


if __name__ == "__main__":
    main()
