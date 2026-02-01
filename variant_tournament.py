import os
import subprocess
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

BOT_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public/bots/dpinto/champion_variants"
HARESH_BOT = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public/bots/hareshm/BEST-champion_bot.py"
MAP_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public/maps/eric"

MAPS = [
    os.path.join(MAP_DIR, "map5_grind.txt"),
    os.path.join(MAP_DIR, "map6_overload.txt"),
    os.path.join(MAP_DIR, "map4_chaos.txt")
]

TURNS = 200
MAX_WORKERS = 4

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
    variants = [os.path.join(BOT_DIR, f) for f in os.listdir(BOT_DIR) if f.endswith(".py")]
    all_bots = variants + [HARESH_BOT]
    
    matches = []
    pairs = list(itertools.combinations(all_bots, 2))
    total_tasks = len(pairs) * len(MAPS)
    print(f"Starting Variant Tournament: {len(all_bots)} bots, {len(MAPS)} maps, {total_tasks} matches.")
    
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for map_path in MAPS:
            for b1, b2 in pairs:
                futures.append(executor.submit(run_match, b1, b2, map_path))
        
        count = 0
        for future in futures:
            res = future.result()
            if res: all_results.append(res)
            count += 1
            if count % 10 == 0: print(f"Completed {count}/{total_tasks}...")

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
    with open("variant_results.json", "w") as f:
        json.dump({"leaderboard": leaderboard, "matches": all_results}, f, indent=4)
        
    print("\nVARIANT LEADERBOARD:")
    for i, (bot, stats) in enumerate(leaderboard):
        win_rate = (stats["wins"] / stats["matches"]) * 100 if stats["matches"] > 0 else 0
        print(f"{i+1}. {bot}: Wins={stats['wins']}, Matches={stats['matches']}, WinRate={win_rate:.1f}%, Total Money={stats['total_money']}")

if __name__ == "__main__":
    main()
