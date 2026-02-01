"""
Atlas Bot Tournament Verification Script
=========================================

Tests atlas_bot.py against test.py (first-place bot) across all maps.
Target: ≥60% win rate

Usage:
    cd awap-game-engine-2026
    python tournaments/test_atlas_bot.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configuration
ATLAS_BOT_PATH = PROJECT_ROOT / "bots" / "atlas_bot.py"
TARGET_BOT_PATH = Path(r"m:\A0 Coding Projects\GitHub\AWAP\test.py")

# Map directories
MAP_DIRS = [
    PROJECT_ROOT / "maps" / "official",
    PROJECT_ROOT / "maps" / "test-maps",
    PROJECT_ROOT / "maps" / "test-maps-revised-for-fairness",
]

# Results output
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "atlas_bot_results.json"
ANALYSIS_MD = RESULTS_DIR / "counter_bot_analysis.md"


def get_all_maps() -> list:
    """Get all map files from configured directories."""
    maps = []
    for map_dir in MAP_DIRS:
        if map_dir.exists():
            for map_file in map_dir.glob("*.txt"):
                maps.append({
                    "path": str(map_file),
                    "name": map_file.stem,
                    "category": map_dir.name
                })
    return maps


def run_match(map_path: str, red_bot: str, blue_bot: str, 
              verbose: bool = False) -> dict:
    """Run a single match using subprocess and return results."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "game.py"),
        "--red", str(red_bot),
        "--blue", str(blue_bot),
        "--map", str(map_path),
        "--timeout", "0.5"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per game
            cwd=str(PROJECT_ROOT)
        )
        output = result.stdout + result.stderr
        
        # Parse result - format: [GAME OVER] money scores: RED=$950, BLUE=$986
        # and [RESULT] BLUE WINS by $36! or [RESULT] RED WINS or DRAW
        red_score = 0
        blue_score = 0
        winner = "error"
        
        for line in output.split('\n'):
            # Parse scores from: [GAME OVER] money scores: RED=$950, BLUE=$986
            if "money scores:" in line:
                try:
                    import re
                    red_match = re.search(r'RED=\$(\d+)', line)
                    blue_match = re.search(r'BLUE=\$(\d+)', line)
                    if red_match:
                        red_score = int(red_match.group(1))
                    if blue_match:
                        blue_score = int(blue_match.group(1))
                except:
                    pass
            
            # Parse winner from: [RESULT] BLUE WINS or RED WINS or DRAW
            if "[RESULT]" in line:
                if "RED WINS" in line:
                    winner = "red"
                elif "BLUE WINS" in line:
                    winner = "blue"
                elif "DRAW" in line or "TIE" in line:
                    winner = "tie"
        
        # Fallback based on scores
        if winner == "error" and (red_score > 0 or blue_score > 0):
            if red_score > blue_score:
                winner = "red"
            elif blue_score > red_score:
                winner = "blue"
            else:
                winner = "tie"
        
        return {
            "red_score": red_score,
            "blue_score": blue_score,
            "winner": winner,
            "error": None if winner != "error" else "Could not parse result",
            "output": output if verbose else None
        }
    except subprocess.TimeoutExpired:
        return {
            "red_score": 0,
            "blue_score": 0,
            "winner": "timeout",
            "error": "Game timed out"
        }
    except Exception as e:
        return {
            "red_score": 0,
            "blue_score": 0,
            "winner": "error",
            "error": str(e)
        }


def run_tournament(verbose: bool = False) -> dict:
    """Run full tournament: atlas_bot vs test.py on all maps."""
    print("=" * 60)
    print("ATLAS BOT TOURNAMENT VERIFICATION")
    print("=" * 60)
    print(f"Target Bot: {TARGET_BOT_PATH}")
    print(f"Atlas Bot:  {ATLAS_BOT_PATH}")
    print()
    
    # Check bots exist
    if not ATLAS_BOT_PATH.exists():
        print(f"  [FAIL] Atlas bot not found: {ATLAS_BOT_PATH}")
        return {"error": f"Atlas bot not found: {ATLAS_BOT_PATH}"}
    if not TARGET_BOT_PATH.exists():
        print(f"  [FAIL] Target bot not found: {TARGET_BOT_PATH}")
        return {"error": f"Target bot not found: {TARGET_BOT_PATH}"}
    print("  [OK] Both bots found")
    
    # Get maps
    maps = get_all_maps()
    print(f"\nFound {len(maps)} maps across {len(MAP_DIRS)} directories")
    
    # Results tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "atlas_bot": str(ATLAS_BOT_PATH),
        "target_bot": str(TARGET_BOT_PATH),
        "matches": [],
        "summary": {
            "total_matches": 0,
            "atlas_wins": 0,
            "target_wins": 0,
            "ties": 0,
            "errors": 0,
            "by_category": defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}),
            "by_side": {"red_wins": 0, "blue_wins": 0}
        }
    }
    
    # Run matches
    print("\nRunning matches...")
    for map_info in maps:
        map_name = map_info["name"]
        map_category = map_info["category"]
        map_path = map_info["path"]
        
        print(f"\n  {map_category}/{map_name}:")
        
        # Match 1: Atlas as RED, Target as BLUE
        result1 = run_match(map_path, str(ATLAS_BOT_PATH), str(TARGET_BOT_PATH), verbose)
        results["summary"]["total_matches"] += 1
        
        if result1["error"]:
            print(f"    RED: ERROR - {result1['error']}")
            results["summary"]["errors"] += 1
        else:
            if result1["winner"] == "red":
                results["summary"]["atlas_wins"] += 1
                results["summary"]["by_side"]["red_wins"] += 1
                results["summary"]["by_category"][map_category]["wins"] += 1
                print(f"    RED: ATLAS WIN ({result1['red_score']} vs {result1['blue_score']})")
            elif result1["winner"] == "blue":
                results["summary"]["target_wins"] += 1
                results["summary"]["by_category"][map_category]["losses"] += 1
                print(f"    RED: TARGET WIN ({result1['red_score']} vs {result1['blue_score']})")
            else:
                results["summary"]["ties"] += 1
                results["summary"]["by_category"][map_category]["ties"] += 1
                print(f"    RED: TIE ({result1['red_score']} vs {result1['blue_score']})")
        
        results["matches"].append({
            "map": map_name,
            "category": map_category,
            "atlas_side": "red",
            **result1
        })
        
        # Match 2: Atlas as BLUE, Target as RED
        result2 = run_match(map_path, str(TARGET_BOT_PATH), str(ATLAS_BOT_PATH), verbose)
        results["summary"]["total_matches"] += 1
        
        if result2["error"]:
            print(f"    BLUE: ERROR - {result2['error']}")
            results["summary"]["errors"] += 1
        else:
            if result2["winner"] == "blue":
                results["summary"]["atlas_wins"] += 1
                results["summary"]["by_side"]["blue_wins"] += 1
                results["summary"]["by_category"][map_category]["wins"] += 1
                print(f"    BLUE: ATLAS WIN ({result2['blue_score']} vs {result2['red_score']})")
            elif result2["winner"] == "red":
                results["summary"]["target_wins"] += 1
                results["summary"]["by_category"][map_category]["losses"] += 1
                print(f"    BLUE: TARGET WIN ({result2['blue_score']} vs {result2['red_score']})")
            else:
                results["summary"]["ties"] += 1
                results["summary"]["by_category"][map_category]["ties"] += 1
                print(f"    BLUE: TIE ({result2['blue_score']} vs {result2['red_score']})")
        
        results["matches"].append({
            "map": map_name,
            "category": map_category,
            "atlas_side": "blue",
            **result2
        })
    
    # Calculate win rate
    total_played = results["summary"]["atlas_wins"] + results["summary"]["target_wins"] + results["summary"]["ties"]
    if total_played > 0:
        win_rate = results["summary"]["atlas_wins"] / total_played * 100
    else:
        win_rate = 0
    
    results["summary"]["win_rate"] = win_rate
    
    # Convert defaultdict for JSON serialization
    results["summary"]["by_category"] = dict(results["summary"]["by_category"])
    
    return results


def generate_analysis_report(results: dict) -> str:
    """Generate Markdown analysis report."""
    report = []
    report.append("# Atlas Bot Counter-Strategy Analysis Report")
    report.append("")
    report.append(f"**Generated:** {results.get('timestamp', 'N/A')}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append("")
    summary = results.get("summary", {})
    report.append(f"- **Total Matches:** {summary.get('total_matches', 0)}")
    report.append(f"- **Atlas Wins:** {summary.get('atlas_wins', 0)}")
    report.append(f"- **Target Wins:** {summary.get('target_wins', 0)}")
    report.append(f"- **Ties:** {summary.get('ties', 0)}")
    report.append(f"- **Errors:** {summary.get('errors', 0)}")
    report.append(f"- **Overall Win Rate:** {summary.get('win_rate', 0):.1f}%")
    report.append("")
    
    # Target achievement
    target_rate = 60.0
    achieved = summary.get('win_rate', 0) >= target_rate
    status = "[ACHIEVED]" if achieved else "[NOT ACHIEVED]"
    report.append(f"### Target: ≥{target_rate}% Win Rate - {status}")
    report.append("")
    
    # Side balance
    report.append("## Side Balance")
    report.append("")
    by_side = summary.get("by_side", {})
    report.append(f"- **Wins as RED:** {by_side.get('red_wins', 0)}")
    report.append(f"- **Wins as BLUE:** {by_side.get('blue_wins', 0)}")
    report.append("")
    
    # Category breakdown
    report.append("## Performance by Map Category")
    report.append("")
    by_category = summary.get("by_category", {})
    for category, stats in by_category.items():
        total = stats.get("wins", 0) + stats.get("losses", 0) + stats.get("ties", 0)
        cat_rate = stats.get("wins", 0) / total * 100 if total > 0 else 0
        report.append(f"### {category}")
        report.append(f"- Wins: {stats.get('wins', 0)}")
        report.append(f"- Losses: {stats.get('losses', 0)}")
        report.append(f"- Ties: {stats.get('ties', 0)}")
        report.append(f"- Win Rate: {cat_rate:.1f}%")
        report.append("")
    
    # Counter-strategy analysis
    report.append("## Counter-Strategy Analysis")
    report.append("")
    report.append("### Target Bot Weaknesses Exploited")
    report.append("")
    report.append("1. **NO SABOTAGE DEFENSE** - Target never switches maps or protects resources")
    report.append("2. **NO PLATE RECYCLING** - Always buys new plates, wasting $2 each time")
    report.append("3. **HEAVY COOKER DEPENDENCY** - Pan theft cripples entire cooking pipeline")
    report.append("4. **PREDICTABLE ASSEMBLY POINTS** - Uses tiles adjacent to shop")
    report.append("5. **FIXED STUCK THRESHOLDS** - 15+ stuck count or 35+ no-progress triggers abort")
    report.append("")
    
    report.append("### Atlas Bot Counter-Tactics")
    report.append("")
    report.append("1. **Aggressive Pan Theft** - Prioritize stealing pans from cookers during switch")
    report.append("2. **Plate Disruption** - Steal plates to force repurchase")
    report.append("3. **Timing Attack** - Sabotage during cooking phase when pans are vulnerable")
    report.append("4. **Economic Warfare** - Target high-value items (MEAT > EGG)")
    report.append("5. **Throughput Matching** - Maintain competitive order completion")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations for Improvement")
    report.append("")
    if not achieved:
        report.append("The target win rate was not achieved. Consider:")
        report.append("")
        report.append("1. **Optimize sabotage timing** - Analyze when target is most vulnerable")
        report.append("2. **Improve order efficiency** - Match target's order selection algorithm")
        report.append("3. **Add parallel cooking** - Utilize multiple cookers like target does")
        report.append("4. **Reduce idle time** - Keep bots productive during non-sabotage phases")
        report.append("5. **Map-specific strategies** - Tune behavior for problematic maps")
    else:
        report.append("Target achieved! Consider:")
        report.append("")
        report.append("1. **Increase margin** - Push for higher win rate on weak maps")
        report.append("2. **Robustness** - Test against more opponents")
        report.append("3. **Edge cases** - Verify behavior on unusual map layouts")
    report.append("")
    
    return "\n".join(report)


def main():
    """Main entry point."""
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run tournament
    results = run_tournament(verbose=False)
    
    if "error" in results:
        print(f"\nTournament failed: {results['error']}")
        return 1
    
    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"Total Matches: {summary['total_matches']}")
    print(f"Atlas Wins:    {summary['atlas_wins']}")
    print(f"Target Wins:   {summary['target_wins']}")
    print(f"Ties:          {summary['ties']}")
    print(f"Errors:        {summary['errors']}")
    print(f"Win Rate:      {summary['win_rate']:.1f}%")
    print()
    
    # Check target
    target_rate = 60.0
    if summary['win_rate'] >= target_rate:
        print(f"[SUCCESS] TARGET ACHIEVED: {summary['win_rate']:.1f}% >= {target_rate}%")
    else:
        print(f"[FAIL] TARGET NOT MET: {summary['win_rate']:.1f}% < {target_rate}%")
    
    # Save results
    print(f"\nSaving results to {RESULTS_JSON}...")
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save analysis report
    print(f"Generating analysis report at {ANALYSIS_MD}...")
    report = generate_analysis_report(results)
    with open(ANALYSIS_MD, 'w') as f:
        f.write(report)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
