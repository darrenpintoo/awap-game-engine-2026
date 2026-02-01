
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"
ULTIMATE_BOT = os.path.join(BASE_DIR, "bots/dpinto/ultimate_champion_bot.py")

# Opponents categorized by potential difficulty
OPPONENTS = [
    # Top Tier
    "bots/hareshm/BEST-champion_bot.py",
    "bots/eric/IronChefOptimized.py",
    "bots/eric/TrueUltimateChefBot.py",
    
    # Mid Tier / Distinct Strategies
    "bots/eric/Apex_Chef.py",
    "bots/eric/StrategicKitchen.py",
    "bots/eric/UltimateChefBot.py",
    "bots/dpinto/planner_bot.py",
    "bots/dpinto/solo_bot.py",
    "bots/dpinto/team_bot.py",
    "bots/duo_noodle_bot.py",
    "bots/hareshm/optimal_bot.py",
    
    # Also include the variants that did well
    "bots/dpinto/champion_variants/v09_no_sabotage.py",
    "bots/dpinto/champion_variants/v4_no_sabotage.py"
]

FULL_OPPONENT_PATHS = [os.path.join(BASE_DIR, op) for op in OPPONENTS if os.path.exists(os.path.join(BASE_DIR, op))]

# Representative map set
MAPS = [
    "maps/eric/map5_grind.txt",
    "maps/eric/map6_overload.txt",
    "maps/eric/map4_chaos.txt",
    "maps/eric/throughput.txt",
    "maps/dpinto/easy_ramp.txt",
    "maps/dpinto/multi_order.txt"
]

FULL_MAP_PATHS = [os.path.join(BASE_DIR, m) for m in MAPS]

TURNS = 200
MAX_WORKERS = 8

def run_match(opponent, map_path):
    # Run Ultimate vs Opponent (Forward)
    cmd1 = ["python3", "src/game.py", "--red", ULTIMATE_BOT, "--blue", opponent, "--map", map_path, "--turns", str(TURNS)]
    res1 = _exec(cmd1, "ULTIMATE", os.path.basename(opponent), map_path)
    
    # Run Opponent vs Ultimate (Reverse)
    cmd2 = ["python3", "src/game.py", "--red", opponent, "--blue", ULTIMATE_BOT, "--map", map_path, "--turns", str(TURNS)]
    res2 = _exec(cmd2, os.path.basename(opponent), "ULTIMATE", map_path)
    
    return [r for r in [res1, res2] if r]

def _exec(cmd, red_name, blue_name, map_path):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
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
    print(f"Starting Global Benchmark: ULTIMATE vs {len(FULL_OPPONENT_PATHS)} bots on {len(FULL_MAP_PATHS)} maps (2 matches each)...")
    
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for op in FULL_OPPONENT_PATHS:
            for m in FULL_MAP_PATHS:
                futures.append(executor.submit(run_match, op, m))
        
        count = 0
        total_tasks = len(futures)
        for f in futures:
            res = f.result()
            if res: all_results.extend(res)
            count += 1
            if count % 10 == 0:
                print(f"Completed {count}/{total_tasks} match pairs...")

    # Summary Stats
    stats = {}
    for op in FULL_OPPONENT_PATHS:
        name = os.path.basename(op)
        stats[name] = {"matches": 0, "losses_to_ultimate": 0, "wins_vs_ultimate": 0, "draws": 0}
        
    ultimate_wins = 0
    ultimate_losses = 0
    ultimate_draws = 0
    
    for r in all_results:
        opponent = r["blue"] if r["red"] == "ULTIMATE" else r["red"]
        
        if opponent not in stats: continue # Should not happen
        
        stats[opponent]["matches"] += 1
        
        if r["winner"] == "ULTIMATE":
            ultimate_wins += 1
            stats[opponent]["losses_to_ultimate"] += 1
        elif r["winner"] == opponent:
            ultimate_losses += 1
            stats[opponent]["wins_vs_ultimate"] += 1
        else:
            ultimate_draws += 1
            stats[opponent]["draws"] += 1

    print("\nGLOBAL BENCHMARK RESULTS (Ultimate Champion Bot Performance):")
    print(f"Total Matches: {len(all_results)}")
    print(f"Overall Win Rate: {(ultimate_wins/len(all_results))*100:.1f}%")
    print(f"Overall Loss Rate: {(ultimate_losses/len(all_results))*100:.1f}%")
    print(f"Overall Draw Rate: {(ultimate_draws/len(all_results))*100:.1f}%")
    
    print("\nDetailed Breakdown vs Opponents:")
    print(f"{'Opponent':<25} | {'Wins':<5} | {'Losses':<6} | {'Draws':<5} | {'Win Rate'}")
    print("-" * 65)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['wins_vs_ultimate'], reverse=True)
    
    for name, s in sorted_stats:
        # Win rate here is clearly wins / matches
        # Note: from Ultimate's perspective, a "Loss" for Ultimate is a "Win" for opponent
        wr = (s['losses_to_ultimate'] / s['matches']) * 100 if s['matches'] > 0 else 0
        print(f"{name:<25} | {s['losses_to_ultimate']:<5} | {s['wins_vs_ultimate']:<6} | {s['draws']:<5} | {wr:.1f}%")

if __name__ == "__main__":
    main()
