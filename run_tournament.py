"""
Tournament Runner - Test champion_bot against all competitors
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
CHAMPION_BOT = "bots/hareshm/champion_bot.py"
COMPETITOR_BOTS = [
    "bots/duo_noodle_bot.py",
    "bots/dpinto/planner_bot.py", 
    "bots/eric/iron_chef_bot.py",
    "bots/eric/PipelineChefBot.py",
]

MAPS = [
    # Eric's maps
    "maps/eric/map2.txt",
    "maps/eric/map3_sprint.txt",
    "maps/eric/map4_chaos.txt",
    "maps/eric/map5_grind.txt",
    "maps/eric/map6_overload.txt",
    # Darren's maps
    "maps/dpinto/easy_ramp.txt",
    "maps/dpinto/multi_order.txt",
    "maps/dpinto/rush_orders.txt",
    "maps/dpinto/stress_test.txt",
    # Haresh's maps
    "maps/haresh/map1.txt",
    "maps/haresh/map_compact.txt",
    "maps/haresh/map_challenge.txt",
]

def run_game(red_bot: str, blue_bot: str, map_path: str) -> str:
    """Run a game and return the result"""
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--timeout", "0.5"  # Timeout per turn
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 min max per game
            cwd=str(Path(__file__).parent)
        )
        output = result.stdout + result.stderr
        
        # Parse result
        if "RED WINS" in output:
            return "RED"
        elif "BLUE WINS" in output:
            return "BLUE"
        elif "DRAW" in output:
            return "DRAW"
        else:
            return "ERROR"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    print("=" * 70)
    print("AWAP 2026 Tournament - Champion Bot vs Competitors")
    print("=" * 70)
    
    results = {}
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_games = 0
    
    for competitor in COMPETITOR_BOTS:
        competitor_name = competitor.split("/")[-1].replace(".py", "")
        results[competitor_name] = {"wins": 0, "losses": 0, "draws": 0}
        
        print(f"\n--- Champion vs {competitor_name} ---")
        
        for map_path in MAPS:
            map_name = map_path.split("/")[-1].replace(".txt", "")
            
            # Run game with champion as RED
            result = run_game(CHAMPION_BOT, competitor, map_path)
            
            if result == "RED":
                outcome = "WIN"
                results[competitor_name]["wins"] += 1
                total_wins += 1
            elif result == "BLUE":
                outcome = "LOSS"
                results[competitor_name]["losses"] += 1
                total_losses += 1
            elif result == "DRAW":
                outcome = "DRAW"
                results[competitor_name]["draws"] += 1
                total_draws += 1
            else:
                outcome = result
            
            total_games += 1
            print(f"  {map_name:20s} : {outcome}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TOURNAMENT SUMMARY")
    print("=" * 70)
    
    for competitor, stats in results.items():
        win_rate = stats['wins'] / (stats['wins'] + stats['losses'] + stats['draws']) * 100 if (stats['wins'] + stats['losses'] + stats['draws']) > 0 else 0
        print(f"{competitor:25s}: {stats['wins']}W - {stats['losses']}L - {stats['draws']}D ({win_rate:.1f}%)")
    
    print("-" * 70)
    overall_rate = total_wins / total_games * 100 if total_games > 0 else 0
    print(f"{'OVERALL':25s}: {total_wins}W - {total_losses}L - {total_draws}D ({overall_rate:.1f}%)")
    print(f"\nTarget: Win 7+ out of 12 maps per competitor = {7/12*100:.1f}% win rate")
    print("=" * 70)


if __name__ == "__main__":
    main()
