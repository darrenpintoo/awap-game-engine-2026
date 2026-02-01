import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

RED_BOT = "bots/hareshm/BEST-champion_bot.py"
BLUE_BOT = "bots/dpinto/champion_variants/v_final_efficiency.py"
BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"

MAPS = [
    "maps/eric/map5_grind.txt",
    "maps/eric/map6_overload.txt",
    "maps/eric/map4_chaos.txt",
    "maps/eric/map3_sprint.txt",
    "maps/eric/throughput.txt",
    "maps/dpinto/mega_warehouse.txt",
    "maps/dpinto/easy_ramp.txt",
    "maps/dpinto/multi_order.txt",
    "maps/dpinto/resource_war.txt",
    "maps/haresh/map_compact.txt"
]

FULL_MAP_PATHS = [os.path.join(BASE_DIR, m) for m in MAPS]
TURNS = 200
MAX_WORKERS = 5

def run_match(map_path):
    # Run twice: sway sides
    results = []
    
    # Match 1: Red=Champion, Blue=Final
    cmd1 = ["python3", "src/game.py", "--red", RED_BOT, "--blue", BLUE_BOT, "--map", map_path, "--turns", str(TURNS)]
    res1 = _exec(cmd1, "CHAMPION", "FINAL", map_path)
    if res1: results.append(res1)
    
    # Match 2: Red=Final, Blue=Champion
    cmd2 = ["python3", "src/game.py", "--red", BLUE_BOT, "--blue", RED_BOT, "--map", map_path, "--turns", str(TURNS)]
    res2 = _exec(cmd2, "FINAL", "CHAMPION", map_path)
    if res2: results.append(res2)
    
    return results

def _exec(cmd, red_name, blue_name, map_path):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        scores_line = [line for line in output.split('\n') if "money scores:" in line]
        if not scores_line: return None
        line = scores_line[0]
        parts = line.split("scores:")[1].split(",")
        red_score = int(parts[0].split("=$")[1])
        blue_score = int(parts[1].split("=$")[1])
        
        winner = "DRAW"
        if red_score > blue_score: winner = red_name
        elif blue_score > red_score: winner = blue_name
            
        return {
            "map": os.path.basename(map_path),
            "red": red_name,
            "blue": blue_name,
            "red_score": red_score,
            "blue_score": blue_score,
            "winner": winner
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"Starting Final Duel: CHAMPION vs FINAL on {len(MAPS)} maps (2 matches each)...")
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_match, m) for m in FULL_MAP_PATHS]
        for f in futures:
            res_list = f.result()
            if res_list: all_results.extend(res_list)

    # Summary
    final_wins = 0
    champion_wins = 0
    draws = 0
    final_total_money = 0
    champion_total_money = 0
    
    for r in all_results:
        if r["winner"] == "FINAL": final_wins += 1
        elif r["winner"] == "CHAMPION": champion_wins += 1
        else: draws += 1
        
        if r["red"] == "FINAL":
            final_total_money += r["red_score"]
            champion_total_money += r["blue_score"]
        else:
            final_total_money += r["blue_score"]
            champion_total_money += r["red_score"]
            
    print("\nFINAL DUEL RESULTS:")
    print(f"FINAL Bot Wins: {final_wins}")
    print(f"CHAMPION Bot Wins: {champion_wins}")
    print(f"Draws: {draws}")
    print(f"FINAL Money: ${final_total_money}")
    print(f"CHAMPION Money: ${champion_total_money}")
    
    print("\nMatch Details:")
    for r in all_results:
        print(f"{r['map']}: {r['red']} vs {r['blue']} -> {r['winner']} ({r['red_score']} - {r['blue_score']})")

if __name__ == "__main__":
    main()
