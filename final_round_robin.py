
import os
import subprocess
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"
BOT_DIR = os.path.join(BASE_DIR, "bots/dpinto/champion_variants")
HARESH_BOT = os.path.join(BASE_DIR, "bots/hareshm/BEST-champion_bot.py")
ULTIMATE_BOT = os.path.join(BASE_DIR, "bots/dpinto/ultimate_champion_bot.py")
IRON_CHEF = os.path.join(BASE_DIR, "bots/eric/IronChefOptimized.py")

# Include top variants for robustness check
VARIANTS_TO_TEST = [
    os.path.join(BOT_DIR, "v09_no_sabotage.py"),
    os.path.join(BOT_DIR, "v18_no_recycling.py"),
    os.path.join(BOT_DIR, "v4_no_sabotage.py")
]

MAPS = [
    os.path.join(BASE_DIR, "maps/eric/map5_grind.txt"),
    os.path.join(BASE_DIR, "maps/eric/map6_overload.txt"),
    os.path.join(BASE_DIR, "maps/eric/map4_chaos.txt"),
    os.path.join(BASE_DIR, "maps/eric/map3_sprint.txt"),
    os.path.join(BASE_DIR, "maps/eric/throughput.txt"),
    os.path.join(BASE_DIR, "maps/dpinto/mega_warehouse.txt"),
    os.path.join(BASE_DIR, "maps/dpinto/easy_ramp.txt"),
    os.path.join(BASE_DIR, "maps/dpinto/multi_order.txt"),
    os.path.join(BASE_DIR, "maps/dpinto/resource_war.txt"),
    os.path.join(BASE_DIR, "maps/haresh/map_compact.txt")
]

TURNS = 200
MAX_WORKERS = 8

def run_match(red_bot, blue_bot, map_path):
    cmd = [
        "python3", "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--turns", str(TURNS)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    competitors = [ULTIMATE_BOT, HARESH_BOT, IRON_CHEF] + VARIANTS_TO_TEST
    competitors = [c for c in competitors if os.path.exists(c)]
    
    print(f"Starting Final Complete Round-Robin with {len(competitors)} bots on {len(MAPS)} maps.")
    
    matches = []
    # Full round robin: every pair plays on every map
    # Since side matters (Red/Blue), we test both permutations A vs B and B vs A
    # Calculate all permutations of length 2
    pairs = list(itertools.permutations(competitors, 2))
    
    total_tasks = len(pairs) * len(MAPS)
    print(f"Total matches to run: {total_tasks}")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for map_path in MAPS:
            if not os.path.exists(map_path):
                print(f"Warning: Map {map_path} not found")
                continue
            for b1, b2 in pairs:
                futures.append(executor.submit(run_match, b1, b2, map_path))
        
        count = 0
        for future in futures:
            res = future.result()
            if res:
                all_results.append(res)
            count += 1
            if count % 10 == 0:
                print(f"Completed {count}/{total_tasks}...")

    # Aggregating
    summary = {}
    for bot in competitors:
        name = os.path.basename(bot)
        summary[name] = {"wins": 0, "losses": 0, "draws": 0, "total_money": 0, "matches": 0}

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

    # Sorting
    leaderboard = sorted(summary.items(), key=lambda x: (x[1]["wins"], x[1]["total_money"]), reverse=True)
    
    results_data = {
        "leaderboard": leaderboard,
        "matches": all_results
    }
    
    with open("final_tournament_results.json", "w") as f:
        json.dump(results_data, f, indent=4)
        
    print("\nFINAL LEADERBOARD:")
    for i, (bot, stats) in enumerate(leaderboard):
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        print(f"{i+1}. {bot}: Wins={stats['wins']}, Matches={stats['matches']}, WinRate={win_rate:.1f}%, Total Money=${stats['total_money']}")

if __name__ == "__main__":
    main()
