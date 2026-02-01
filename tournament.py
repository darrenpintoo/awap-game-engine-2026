import subprocess
import os
import json
import itertools
from concurrent.futures import ThreadPoolExecutor

# Configuration
BOTS = [
    "bots/duo_noodle_bot.py",
    "bots/eric/TrueUltimateChefBot.py",
    "bots/eric/UltimateChefBot.py",
    "bots/eric/Apex_Chef.py",
    "bots/eric/iron_chef_bot.py",
    "bots/eric/IronChefOptimized.py",
    "bots/eric/StrategicKitchen.py",
    "bots/dpinto/solo_bot.py",
    "bots/dpinto/pair_bot.py",
    "bots/dpinto/planner_bot.py",
    "bots/dpinto/team_bot.py",
    "bots/hareshm/optimal_bot.py",
    "bots/hareshm/BEST - champion_bot.py"
]

MAPS = [
    "maps/eric/map3_sprint.txt",
    "maps/eric/throughput.txt",
    "maps/eric/map5_grind.txt",
    "maps/eric/map2.txt",
    "maps/eric/map6_overload.txt",
    "maps/eric/map4_chaos.txt",
    "maps/haresh/map_challenge.txt",
    "maps/haresh/map1.txt",
    "maps/haresh/map_compact.txt",
    "maps/dpinto/mega_warehouse.txt",
    "maps/dpinto/mega_maze.txt",
    "maps/dpinto/easy_ramp.txt",
    "maps/dpinto/multi_ingredient.txt",
    "maps/dpinto/rush_orders.txt",
    "maps/dpinto/multi_order.txt",
    "maps/dpinto/time_pressure.txt",
    "maps/dpinto/resource_war.txt",
    "maps/dpinto/stress_test.txt",
    "maps/dpinto/chaos_kitchen.txt"
]

TURNS = 200
MAX_WORKERS = 12  # Increased for faster execution

def run_match(red_bot, blue_bot, map_path):
    cmd = [
        "python3", "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--turns", str(TURNS)
    ]
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        # Parse score from output: "money scores: RED=$568, BLUE=$1412"
        scores_line = [line for line in output.split('\n') if "money scores:" in line]
        if not scores_line:
            return None
            
        line = scores_line[0]
        # RED=$568, BLUE=$1412
        parts = line.split("scores:")[1].split(",")
        red_score = int(parts[0].split("=$")[1])
        blue_score = int(parts[1].split("=$")[1])
        
        return {
            "red": red_bot,
            "blue": blue_bot,
            "map": map_path,
            "red_score": red_score,
            "blue_score": blue_score,
            "winner": "RED" if red_score > blue_score else ("BLUE" if blue_score > red_score else "DRAW")
        }
    except Exception as e:
        print(f"Error running match {red_bot} vs {blue_bot} on {map_path}: {e}")
        return None

def main():
    # Filter non-existent files
    existing_bots = [b for b in BOTS if os.path.exists(b)]
    existing_maps = [m for m in MAPS if os.path.exists(m)]
    
    print(f"Starting tournament with {len(existing_bots)} bots and {len(existing_maps)} maps.")
    
    # All pairs (A vs B) - including symmetric pairs if order matters (RED vs BLUE)
    # For a true round robin, we should run each pair once, or twice (swapped colors)
    # Let's run each unique pair once to save time.
    matches = []
    pairs = list(itertools.combinations(existing_bots, 2))
    
    total_tasks = len(pairs) * len(existing_maps)
    print(f"Total matches to run: {total_tasks}")
    
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for map_path in existing_maps:
            for b1, b2 in pairs:
                futures.append(executor.submit(run_match, b1, b2, map_path))
        
        count = 0
        for future in futures:
            res = future.result()
            if res:
                all_results.append(res)
            count += 1
            if count % 10 == 0:
                print(f"Completed {count}/{total_tasks} matches...")

    # Aggregating
    summary = {}
    for bot in existing_bots:
        summary[bot] = {"wins": 0, "losses": 0, "draws": 0, "total_money": 0, "matches": 0}

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
    
    with open("tournament_results.json", "w") as f:
        json.dump(results_data, f, indent=4)
        
    print("\nLEADERBOARD:")
    for i, (bot, stats) in enumerate(leaderboard):
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        print(f"{i+1}. {bot}: Wins={stats['wins']}, Matches={stats['matches']}, WinRate={win_rate:.1f}%, Total Money={stats['total_money']}")

if __name__ == "__main__":
    main()
