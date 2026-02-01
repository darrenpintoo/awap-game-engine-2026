"""
Hydra Tournament - Test Hydra Bot against all competitors on all maps
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Configuration
HERO_BOT = "bots/hareshm/BEST-Hydra-Bot.py"

COMPETITOR_BOTS = [
    # Core bots
    "bots/hareshm/champion_bot.py",
    "bots/hareshm/SOVEREIGN.py",
    "bots/eric/BEST-IronChefOptimized.py",
    "bots/dpinto/BEST_solo_bot_v1.py",
    # Test bots
    "bots/test-bots/aggressive_saboteur.py",
    "bots/test-bots/turtle_defender.py",
    "bots/test-bots/efficiency_maximizer.py",
    "bots/test-bots/greedy_picker.py",
    "bots/test-bots/rush_bot.py",
    "bots/test-bots/adaptive_switcher.py",
]

MAPS = [
    # Official maps
    "maps/official/chopped.txt",
    "maps/official/orbit.txt",
    "maps/official/small_wall.txt",
    "maps/official/throughput.txt",
    "maps/official/v1.txt",
    # Test maps
    "maps/test-maps/01_tiny_sprint.txt",
    "maps/test-maps/02_balanced_medium.txt",
    "maps/test-maps/03_grand_kitchen.txt",
    "maps/test-maps/04_varied_orders.txt",
    "maps/test-maps/05_chokepoint_chaos.txt",
    "maps/test-maps/06_resource_crunch.txt",
    "maps/test-maps/07_pressure_cooker.txt",
    "maps/test-maps/08_sabotage_alley.txt",
    "maps/test-maps/09_remote_pantry.txt",
    "maps/test-maps/10_burn_risk.txt",
    # Revised fair maps
    "maps/test-maps-revised-for-fairness/01_tiny_sprint.txt",
    "maps/test-maps-revised-for-fairness/02_balanced_medium.txt",
    "maps/test-maps-revised-for-fairness/03_grand_kitchen.txt",
    "maps/test-maps-revised-for-fairness/04_varied_orders.txt",
    "maps/test-maps-revised-for-fairness/05_parallel_paths.txt",
    "maps/test-maps-revised-for-fairness/06_high_throughput.txt",
    "maps/test-maps-revised-for-fairness/07_tight_timing.txt",
    "maps/test-maps-revised-for-fairness/08_resource_sharing.txt",
    "maps/test-maps-revised-for-fairness/09_sabotage_ready.txt",
    "maps/test-maps-revised-for-fairness/10_endgame_crunch.txt",
]


def run_game(red_bot: str, blue_bot: str, map_path: str, base_dir: str) -> dict:
    """Run a game and return the result with scores"""
    cmd = [
        sys.executable, "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--timeout", "0.5"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=180,  # 3 min max per game
            cwd=base_dir
        )
        output = result.stdout + result.stderr
        
        # Parse result
        winner = None
        red_score = 0
        blue_score = 0
        
        if "RED WINS" in output:
            winner = "RED"
        elif "BLUE WINS" in output:
            winner = "BLUE"
        elif "DRAW" in output:
            winner = "DRAW"
        else:
            winner = "ERROR"
        
        # Try to parse scores
        for line in output.split("\n"):
            if "RED:" in line and "BLUE:" in line:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "RED:" in p:
                            red_score = int(parts[i+1].replace(",", ""))
                        if "BLUE:" in p:
                            blue_score = int(parts[i+1].replace(",", ""))
                except:
                    pass
        
        return {"winner": winner, "red_score": red_score, "blue_score": blue_score}
        
    except subprocess.TimeoutExpired:
        return {"winner": "TIMEOUT", "red_score": 0, "blue_score": 0}
    except Exception as e:
        return {"winner": f"ERROR: {e}", "red_score": 0, "blue_score": 0}


def main():
    base_dir = str(Path(__file__).parent.parent)
    
    print("=" * 80)
    print("HYDRA TOURNAMENT - Testing Against All Competitors")
    print("=" * 80)
    print(f"Hero: {HERO_BOT}")
    print(f"Competitors: {len(COMPETITOR_BOTS)}")
    print(f"Maps: {len(MAPS)}")
    print(f"Total games: {len(COMPETITOR_BOTS) * len(MAPS) * 2} (home and away)")
    print("=" * 80)
    
    results = {}
    all_game_results = []
    
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_games = 0
    
    for competitor in COMPETITOR_BOTS:
        competitor_name = Path(competitor).stem
        results[competitor_name] = {"wins": 0, "losses": 0, "draws": 0, "score_diff": 0}
        
        print(f"\n{'='*60}")
        print(f"Hydra vs {competitor_name}")
        print(f"{'='*60}")
        
        for map_path in MAPS:
            map_name = Path(map_path).stem
            
            # Check if map exists
            full_map_path = os.path.join(base_dir, map_path)
            if not os.path.exists(full_map_path):
                print(f"  {map_name:25s} : SKIPPED (map not found)")
                continue
            
            # Check if competitor exists
            full_competitor_path = os.path.join(base_dir, competitor)
            if not os.path.exists(full_competitor_path):
                print(f"  {map_name:25s} : SKIPPED (competitor not found)")
                continue
            
            # Run game with Hydra as RED
            game_result = run_game(HERO_BOT, competitor, map_path, base_dir)
            
            if game_result["winner"] == "RED":
                outcome = "WIN"
                results[competitor_name]["wins"] += 1
                total_wins += 1
            elif game_result["winner"] == "BLUE":
                outcome = "LOSS"
                results[competitor_name]["losses"] += 1
                total_losses += 1
            elif game_result["winner"] == "DRAW":
                outcome = "DRAW"
                results[competitor_name]["draws"] += 1
                total_draws += 1
            else:
                outcome = game_result["winner"]
            
            score_diff = game_result["red_score"] - game_result["blue_score"]
            results[competitor_name]["score_diff"] += score_diff
            
            total_games += 1
            print(f"  {map_name:25s} : {outcome:8s} (Hydra: {game_result['red_score']:4d} vs {game_result['blue_score']:4d})")
            
            all_game_results.append({
                "hero": "Hydra",
                "competitor": competitor_name,
                "map": map_name,
                "outcome": outcome,
                "hero_score": game_result["red_score"],
                "competitor_score": game_result["blue_score"]
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("TOURNAMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Competitor':<30} {'Wins':>6} {'Losses':>8} {'Draws':>7} {'Win%':>8} {'ScoreDiff':>10}")
    print("-" * 80)
    
    for competitor, stats in sorted(results.items(), key=lambda x: x[1]['wins'], reverse=True):
        total = stats['wins'] + stats['losses'] + stats['draws']
        win_rate = stats['wins'] / total * 100 if total > 0 else 0
        print(f"{competitor:<30} {stats['wins']:>6} {stats['losses']:>8} {stats['draws']:>7} {win_rate:>7.1f}% {stats['score_diff']:>+10}")
    
    print("-" * 80)
    overall_rate = total_wins / total_games * 100 if total_games > 0 else 0
    print(f"{'OVERALL':<30} {total_wins:>6} {total_losses:>8} {total_draws:>7} {overall_rate:>7.1f}%")
    print("=" * 80)
    
    # Determine if Hydra is winning
    success = overall_rate >= 60.0  # 60% win rate target
    
    if success:
        print("\n✓ HYDRA IS WINNING! Tournament successful.")
    else:
        print(f"\n✗ HYDRA NEEDS IMPROVEMENT. Current win rate: {overall_rate:.1f}%")
        print("  Analyzing weaknesses...")
        
        # Find worst matchups
        worst = sorted(results.items(), key=lambda x: x[1]['wins'] - x[1]['losses'])[:3]
        print(f"  Worst matchups:")
        for comp, stats in worst:
            print(f"    - {comp}: {stats['wins']}W-{stats['losses']}L")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "hero": HERO_BOT,
        "summary": {
            "total_games": total_games,
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
            "win_rate": overall_rate
        },
        "by_competitor": results,
        "games": all_game_results
    }
    
    output_path = os.path.join(base_dir, "results", "hydra_tournament_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return success, results, all_game_results


if __name__ == "__main__":
    main()
