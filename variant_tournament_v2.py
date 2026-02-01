import os
import subprocess
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

BOT_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public/bots/dpinto/champion_variants"
HARESH_BOT = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public/bots/hareshm/BEST-champion_bot.py"
BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"

# Diverse set of maps to test different aspects
MAPS = [
    # Original test set
    "maps/eric/map5_grind.txt",
    "maps/eric/map6_overload.txt",
    "maps/eric/map4_chaos.txt",
    
    # New challenges
    "maps/eric/map3_sprint.txt",      # Speed test
    "maps/eric/throughput.txt",       # High volume
    "maps/dpinto/mega_warehouse.txt", # Large map navigation
    "maps/dpinto/easy_ramp.txt",      # Simple baseline
    "maps/dpinto/multi_order.txt",    # Order complexity
    "maps/dpinto/resource_war.txt",   # Resource contention
    "maps/haresh/map_compact.txt"     # Tight spaces
]

FULL_MAP_PATHS = [os.path.join(BASE_DIR, m) for m in MAPS]

TURNS = 200
MAX_WORKERS = 6  # Increased slightly for more throughput

def run_match(red_bot, blue_bot, map_path):
    cmd = [
        "python3", "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--turns", str(TURNS)
    ]
    try:
        # Reduced timeout to fail fast on deadlocks
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        output = result.stdout
        
        scores_line = [line for line in output.split('\n') if "money scores:" in line]
        if not scores_line: return None
            
        line = scores_line[0]
        parts = line.split("scores:")[1].split(",")
        red_score = int(parts[0].split("=$")[1])
        blue_score = int(parts[1].split("=$")[1])
        
        return {
            "red": os.path.basename(red_bot),
            "blue": os.path.basename(blue_bot),
            "map": os.path.basename(map_path),
            "red_score": red_score,
            "blue_score": blue_score,
            "winner": "RED" if red_score > blue_score else ("BLUE" if blue_score > red_score else "DRAW")
        }
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {os.path.basename(red_bot)} vs {os.path.basename(blue_bot)} on {os.path.basename(map_path)}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    variants = [os.path.join(BOT_DIR, f) for f in os.listdir(BOT_DIR) if f.endswith(".py")]
    # Test against the champion
    all_bots = variants + [HARESH_BOT]
    
    # To save time, we will NOT do full round robin between variants.
    # We will test each variant against the CHAMPION bot on all maps.
    # And maybe a few inter-variant matches if time permits.
    # Actually, let's just do Variant vs Champion for now to find the best improvements.
    
    matches_to_run = []
    
    # 1. Variant vs Champion (Both sides)
    for v in variants:
        for m in FULL_MAP_PATHS:
            matches_to_run.append((v, HARESH_BOT, m))
            matches_to_run.append((HARESH_BOT, v, m))
            
    total_tasks = len(matches_to_run)
    print(f"Starting Focused Tournament: {len(variants)} variants vs Champion, {len(FULL_MAP_PATHS)} maps, {total_tasks} matches.")
    
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for r, b, m in matches_to_run:
            futures.append(executor.submit(run_match, r, b, m))
        
        count = 0
        for future in futures:
            res = future.result()
            if res: all_results.append(res)
            count += 1
            if count % 10 == 0: print(f"Completed {count}/{total_tasks}...")

    # Aggregation
    summary = {os.path.basename(b): {"wins": 0, "losses": 0, "draws": 0, "total_money": 0, "matches": 0} for b in all_bots}
    
    for res in all_results:
        summary[res["red"]]["total_money"] += res["red_score"]
        summary[res["blue"]]["total_money"] += res["blue_score"]
        summary[res["red"]]["matches"] += 1
        summary[res["blue"]]["matches"] += 1
        
        if res["winner"] == "RED":
            summary[res["red"]]["wins"] += 1
            summary[res["blue"]]["losses"] += 1
        elif res["winner"] == "BLUE":
            summary[res["blue"]]["wins"] += 1
            summary[res["red"]]["losses"] += 1
        else:
            summary[res["red"]]["draws"] += 1
            summary[res["blue"]]["draws"] += 1

    leaderboard = sorted(summary.items(), key=lambda x: (x[1]["wins"], x[1]["total_money"]), reverse=True)
    with open("variant_results_v2.json", "w") as f:
        json.dump({"leaderboard": leaderboard, "matches": all_results}, f, indent=4)
        
    print("\nVARIANT LEADERBOARD (vs Champion):")
    for i, (bot, stats) in enumerate(leaderboard):
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        print(f"{i+1}. {bot}: Wins={stats['wins']}, Matches={stats['matches']}, WinRate={win_rate:.1f}%, Money=${stats['total_money']}")

if __name__ == "__main__":
    main()
