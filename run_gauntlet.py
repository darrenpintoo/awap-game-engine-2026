
import sys
import os
import glob
import contextlib
import io
import time
from concurrent.futures import ThreadPoolExecutor

# Add 'src' directory to path so we can import game module directly
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from game import Game
    from game_constants import Team, GameConstants
except ImportError as e:
    print(f"Error: Could not import game module: {e}")
    sys.exit(1)

# Configuration
MAPS = [
    "maps/chess.txt",
    "maps/donut.txt", 
    "maps/messy.txt",
    "maps/simple_map.txt",
    "maps/split.txt"
]

# Define core bots
CORE_BOTS = {
    "TrueBot": "bots/eric/truebot.py",
    "TestBot": "testbots/test.py", 
    "JuniorChampion": "bots/dpinto/JuniorChampion.py",
    "Reaper": "bots/hareshm/reaper.py"
}

# Find opponent test bots
TEST_BOTS_PATTERN = "testbots/bot-*.py"
OPPONENT_BOTS = {}
for path in glob.glob(TEST_BOTS_PATTERN):
    name = os.path.basename(path).split("-")[1] # e.g. bot-Burger-... -> Burger
    OPPONENT_BOTS[name] = path

# Combine all bots
ALL_BOTS = {**CORE_BOTS, **OPPONENT_BOTS}

print(f"Loaded {len(ALL_BOTS)} bots: {list(ALL_BOTS.keys())}")
print(f"Loaded {len(MAPS)} maps: {MAPS}")

# Statistics storage
# {bot_name: {wins: 0, matches: 0, total_score: 0, total_margin: 0}}
STATS = {name: {"wins": 0, "matches": 0, "total_score": 0, "total_margin": 0} for name in ALL_BOTS}

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout during game execution to minimalize clutter."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def run_match(map_path, red_name, red_path, blue_name, blue_path):
    """Run a single match and return (red_money, blue_money, winner_name)."""
    
    # Initialize Game
    # We set timeout=0 to disable threading overhead since we are running headless
    # But wait, some bots might need threads? 
    # game.py says: "Optimization: bypass threading if timeout is disabled (<= 0)"
    # This is safer for performance in a gauntlet.
    
    try:
        # Create game instance
        # We catch stdout to avoid spam
        with suppress_stdout():
            game = Game(
                red_bot_path=red_path,
                blue_bot_path=blue_path,
                map_path=map_path,
                render=False,
                turn_limit=2000, # Standard game length
                per_turn_timeout_s=0, # Disable timeout for speed/debug
                fps_cap=999
            )
            game.run_game()
            
        # Extract scores
        red_money = game.game_state.get_team_money(Team.RED)
        blue_money = game.game_state.get_team_money(Team.BLUE)
        
        game.close()
        return red_money, blue_money
        
    except Exception as e:
        print(f"CRASH: {red_name} vs {blue_name} on {map_path}: {e}")
        return 0, 0

def main():
    print("\n=== STARTING GAUNTLET ===\n")
    
    # We want to run:
    # 1. Internal Round Robin (Core vs Core)
    # 2. Gauntlet (Core vs Opponents)
    
    # Actually, simpler: just run Core vs All (including other Core)
    # To save time, we skipped Opponent vs Opponent
    
    tasks = []
    
    # Generate match list
    matches = []
    
    for map_path in MAPS:
        map_name = os.path.basename(map_path)
        
        # Core vs Everyone (Round Robin style involving Core)
        for core_name, core_path in CORE_BOTS.items():
            for opp_name, opp_path in ALL_BOTS.items():
                if core_name == opp_name: continue
                
                # Check if we already added the reverse match to avoid duplication?
                # Actually, side matters (Red/Blue). We should run both ways for fairness.
                # But to save time, let's doing ONE match per pair?
                # No, map bias exists. Let's do PAIRS (Home/Away).
                
                # To avoid double counting (A vs B AND B vs A when iterating B's loop),
                # we sort the names.
                # But we only iterate Core as primary.
                # If opp is also Core, we will encounter (True, Reaper) and later (Reaper, True).
                # This works out perfectly: everyone gets to be Red once against everyone else.
                
                matches.append({
                    "map": map_path,
                    "red": (core_name, core_path),
                    "blue": (opp_name, opp_path)
                })

    print(f"Scheduled {len(matches)} matches.")
    
    start_time = time.time()
    
    for i, m in enumerate(matches):
        red_name, red_path = m["red"]
        blue_name, blue_path = m["blue"]
        map_path = m["map"]
        
        sys.stdout.write(f"\r[{i+1}/{len(matches)}] {red_name} vs {blue_name} on {os.path.basename(map_path)}...")
        sys.stdout.flush()
        
        r_score, b_score = run_match(map_path, red_name, red_path, blue_name, blue_path)
        
        # Update Stats
        # RED Stats
        STATS[red_name]["matches"] += 1
        STATS[red_name]["total_score"] += r_score
        STATS[red_name]["total_margin"] += (r_score - b_score)
        if r_score > b_score: STATS[red_name]["wins"] += 1
        
        # BLUE Stats
        STATS[blue_name]["matches"] += 1
        STATS[blue_name]["total_score"] += b_score
        STATS[blue_name]["total_margin"] += (b_score - r_score)
        if b_score > r_score: STATS[blue_name]["wins"] += 1

    total_time = time.time() - start_time
    print(f"\n\nGauntlet Completed in {total_time:.2f}s")
    
    # Print Results Table
    print("\n=== RESULTS ===\n")
    print(f"{'Bot Name':<20} | {'Win Rate':<10} | {'Avg Score':<10} | {'Avg Margin':<10} | {'Matches':<8}")
    print("-" * 75)
    
    # Sort by Win Rate then Avg Margin
    ranked = sorted(STATS.items(), key=lambda x: (x[1]["wins"]/max(1, x[1]["matches"]), x[1]["total_margin"]), reverse=True)
    
    for name, s in ranked:
        if s["matches"] == 0: continue
        win_rate = (s["wins"] / s["matches"]) * 100
        avg_score = s["total_score"] / s["matches"]
        avg_margin = s["total_margin"] / s["matches"]
        
        print(f"{name:<20} | {win_rate:6.1f}%   | {avg_score:9.0f}  | {avg_margin:+9.0f}  | {s['matches']:<8}")

if __name__ == "__main__":
    main()
