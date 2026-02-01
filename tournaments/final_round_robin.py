
import os
import subprocess
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()

# Bots to include (Active & Relevant)
BOTS = [
    "bots/dpinto/ultimate_champion_bot.py",
    "bots/dpinto/champion_sabotage_optimized.py",
    "bots/hareshm/BEST-champion_bot.py",
    "bots/eric/IronChefOptimized.py",
    "bots/eric/TrueUltimateChefBot.py",
    "bots/eric/UltimateChefBot.py",
    "bots/eric/Apex_Chef.py",
    "bots/eric/StrategicKitchen.py",
    "bots/hareshm/optimal_bot.py",
    "bots/duo_noodle_bot.py",
    "bots/eric/iron_chef_bot.py"
]

# Reduced Map Set for Speed & Relevance
MAPS = [
    "maps/eric/map5_grind.txt",      # Standard competitive
    "maps/eric/map6_overload.txt",   # Complex competitive
    "maps/eric/throughput.txt",      # Open speed test
    "maps/dpinto/choke_point.txt",   # New Edge Case: Congestion
    "maps/dpinto/starvation.txt",    # New Edge Case: Resource Scarcity
    "maps/dpinto/split_kitchen.txt"  # New Edge Case: Coordination
]

TURNS = 150 # Reduced for speed (Still enough to show strategy difference)
TIMEOUT = 20 # Strict timeout

# --- EXECUTION ---

def run_single_match(args):
    """Run a single match and return the result"""
    red_bot, blue_bot, map_path = args
    
    cmd = [
        "python3", "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--turns", str(TURNS)
    ]
    
    start_time = time.time()
    try:
        # Capture stdout/stderr but don't print
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        duration = time.time() - start_time
        
        output = result.stdout
        error_out = result.stderr
        
        # Parse output for score
        game_over_line = None
        for line in output.splitlines():
            if "[GAME OVER]" in line:
                game_over_line = line
                break
                
        if not game_over_line:
            return {
                "red": red_bot, "blue": blue_bot, "map": map_path,
                "error": "No result line found", 
                "output": output[-500:], # Last 500 chars 
                "stderr": error_out[-500:]
            }
            
        # Extract scores: "[GAME OVER] money scores: RED=$123, BLUE=$456"
        parts = game_over_line.split("scores:")[1].split(",")
        red_score = int(parts[0].split("=$")[1])
        blue_score = int(parts[1].split("=$")[1])
        
        winner = "DRAW"
        if red_score > blue_score: winner = red_bot
        elif blue_score > red_score: winner = blue_bot
        
        return {
            "red": red_bot, "blue": blue_bot, "map": map_path,
            "red_score": red_score, "blue_score": blue_score,
            "winner": winner,
            "duration": duration
        }
        
    except subprocess.TimeoutExpired:
        return {
            "red": red_bot, "blue": blue_bot, "map": map_path,
            "error": "Timeout", "red_score": 0, "blue_score": 0
        }
    except Exception as e:
        return {
            "red": red_bot, "blue": blue_bot, "map": map_path,
            "error": str(e), "red_score": 0, "blue_score": 0
        }

def main():
    print(f"🏆 STARTING OPTIMIZED TOURNAMENT (v3 Fast) 🏆")
    print(f"Bots: {len(BOTS)}")
    print(f"Maps: {len(MAPS)}")
    
    matches = []
    pairs = []
    for i in range(len(BOTS)):
        for j in range(len(BOTS)):
            if i == j: continue
            pairs.append((BOTS[i], BOTS[j]))
            
    for p in pairs:
        for m in MAPS:
            matches.append((p[0], p[1], m))
            
    total_matches = len(matches)
    workers = min(multiprocessing.cpu_count(), 12) 
    
    print(f"Total Matches Scheduled: {total_matches}")
    print(f"Estimated Time (@0.5s/match): {total_matches * 0.5 / workers / 60:.1f} minutes")
    print(f"Using {workers} workers with strict {TIMEOUT}s timeout...")

    results = []
    start_global = time.time()
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_match, m) for m in matches]
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            completed += 1
            if "error" in res: error_count += 1
            
            if completed % 50 == 0:
                elapsed = time.time() - start_global
                rate = completed/elapsed if elapsed>0 else 0
                remaining = (total_matches - completed)/rate if rate>0 else 0
                print(f"Progress: {completed}/{total_matches} ({completed/total_matches*100:.1f}%) - Errors: {error_count} - ETA: {remaining/60:.1f} min")

    end_global = time.time()
    print(f"\nTournament Complete in {(end_global - start_global)/60:.1f} minutes!")
    print(f"Total Errors: {error_count}")

    # --- ANALYSIS ---
    leaderboard = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "score": 0, "matches": 0})
    
    for r in results:
        if "error" in r: continue
            
        r_bot = os.path.basename(r["red"])
        b_bot = os.path.basename(r["blue"])
        
        leaderboard[r_bot]["matches"] += 1
        leaderboard[b_bot]["matches"] += 1
        
        leaderboard[r_bot]["score"] += r["red_score"]
        leaderboard[b_bot]["score"] += r["blue_score"]
        
        winner = r["winner"]
        if winner == r["red"]:
            leaderboard[r_bot]["wins"] += 1
            leaderboard[b_bot]["losses"] += 1
        elif winner == r["blue"]:
            leaderboard[b_bot]["wins"] += 1
            leaderboard[r_bot]["losses"] += 1
        else:
            leaderboard[r_bot]["draws"] += 1
            leaderboard[b_bot]["draws"] += 1
            
    # Print Results
    print("\n" + "="*90)
    print("FINAL LEADERBOARD")
    print("="*90)
    print(f"{'Bot Name':<35} | {'Wins':<5} | {'Loss':<5} | {'Draw':<5} | {'Win %':<6} | {'Avg Score':<10}")
    print("-" * 90)
    
    sorted_bots = sorted(leaderboard.items(), key=lambda x: (x[1]['wins'], x[1]['score']), reverse=True)
    
    for name, stats in sorted_bots:
        matches = stats['matches']
        if matches == 0: continue
        win_rate = (stats['wins'] / matches) * 100
        avg_score = stats['score'] / matches
        print(f"{name:<35} | {stats['wins']:<5} | {stats['losses']:<5} | {stats['draws']:<5} | {win_rate:6.1f}% | ${avg_score:<9.1f}")

    with open("final_tournament_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
