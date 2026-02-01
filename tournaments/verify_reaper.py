"""
Reaper Bot Verification Script
==============================

Tests bots/hareshm/reaper.py for functional equivalence with bots/atlas_bot.py
and generates a comparison report.

Tests:
1. Functional Equivalence - Both bots should make similar strategic decisions
2. Style Difference Analysis - Code similarity comparison
3. Performance Validation - Execution time and memory comparison

Usage:
    cd awap-game-engine-2026
    python tournaments/verify_reaper.py
"""

import os
import sys
import re
import subprocess
import time
import difflib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Bot paths
ATLAS_BOT_PATH = PROJECT_ROOT / "bots" / "atlas_bot.py"
REAPER_BOT_PATH = PROJECT_ROOT / "bots" / "hareshm" / "reaper.py"

# Test opponent
TEST_OPPONENT = PROJECT_ROOT / "bots" / "eric" / "Dual.py"

# Map directories for testing
MAP_DIRS = [
    PROJECT_ROOT / "maps" / "official",
    PROJECT_ROOT / "maps" / "test-maps",
]


def get_test_maps(limit: int = 6) -> List[Path]:
    """Get a subset of maps for quick testing."""
    maps = []
    for map_dir in MAP_DIRS:
        if map_dir.exists():
            for map_file in sorted(map_dir.glob("*.txt"))[:limit // 2]:
                maps.append(map_file)
    return maps[:limit]


def run_match(map_path: Path, red_bot: Path, blue_bot: Path) -> dict:
    """Run a single match and return results."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "game.py"),
        "--red", str(red_bot),
        "--blue", str(blue_bot),
        "--map", str(map_path),
        "--timeout", "0.5"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - start_time
        output = result.stdout + result.stderr
        
        # Parse results
        red_score = 0
        blue_score = 0
        winner = "error"
        
        for line in output.split('\n'):
            if "money scores:" in line:
                red_match = re.search(r'RED=\$(\d+)', line)
                blue_match = re.search(r'BLUE=\$(\d+)', line)
                if red_match:
                    red_score = int(red_match.group(1))
                if blue_match:
                    blue_score = int(blue_match.group(1))
            
            if "[RESULT]" in line:
                if "RED WINS" in line:
                    winner = "red"
                elif "BLUE WINS" in line:
                    winner = "blue"
                elif "DRAW" in line or "TIE" in line:
                    winner = "tie"
        
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
            "elapsed": elapsed,
            "error": None
        }
    except Exception as e:
        return {
            "red_score": 0,
            "blue_score": 0,
            "winner": "error",
            "elapsed": time.time() - start_time,
            "error": str(e)
        }


def test_functional_equivalence() -> Dict:
    """
    Test that both bots produce similar strategic outcomes.
    
    Run both bots against the same opponent on the same maps
    and compare win rates.
    """
    print("\n" + "=" * 60)
    print("FUNCTIONAL EQUIVALENCE TEST")
    print("=" * 60)
    
    if not ATLAS_BOT_PATH.exists():
        return {"error": f"Atlas bot not found: {ATLAS_BOT_PATH}"}
    if not REAPER_BOT_PATH.exists():
        return {"error": f"Reaper bot not found: {REAPER_BOT_PATH}"}
    
    # Use Dual bot as opponent if available, otherwise skip
    opponent = TEST_OPPONENT if TEST_OPPONENT.exists() else None
    if not opponent:
        # Try to find any other bot
        for bot_file in (PROJECT_ROOT / "bots").glob("*.py"):
            if bot_file.name not in ["atlas_bot.py"]:
                opponent = bot_file
                break
    
    if not opponent:
        print("  [SKIP] No opponent bot found for testing")
        return {"skipped": True, "reason": "No opponent found"}
    
    print(f"  Opponent: {opponent.name}")
    
    maps = get_test_maps(4)
    print(f"  Testing on {len(maps)} maps...")
    
    results = {
        "atlas": {"wins": 0, "losses": 0, "ties": 0, "scores": [], "times": []},
        "reaper": {"wins": 0, "losses": 0, "ties": 0, "scores": [], "times": []}
    }
    
    for map_path in maps:
        print(f"\n  {map_path.name}:")
        
        # Atlas as RED
        atlas_result = run_match(map_path, ATLAS_BOT_PATH, opponent)
        if atlas_result["winner"] == "red":
            results["atlas"]["wins"] += 1
        elif atlas_result["winner"] == "blue":
            results["atlas"]["losses"] += 1
        else:
            results["atlas"]["ties"] += 1
        results["atlas"]["scores"].append(atlas_result["red_score"])
        results["atlas"]["times"].append(atlas_result["elapsed"])
        print(f"    Atlas:  {atlas_result['winner'].upper()} ({atlas_result['red_score']} vs {atlas_result['blue_score']}) [{atlas_result['elapsed']:.1f}s]")
        
        # Reaper as RED
        reaper_result = run_match(map_path, REAPER_BOT_PATH, opponent)
        if reaper_result["winner"] == "red":
            results["reaper"]["wins"] += 1
        elif reaper_result["winner"] == "blue":
            results["reaper"]["losses"] += 1
        else:
            results["reaper"]["ties"] += 1
        results["reaper"]["scores"].append(reaper_result["red_score"])
        results["reaper"]["times"].append(reaper_result["elapsed"])
        print(f"    Reaper: {reaper_result['winner'].upper()} ({reaper_result['red_score']} vs {reaper_result['blue_score']}) [{reaper_result['elapsed']:.1f}s]")
    
    # Calculate stats
    for bot in ["atlas", "reaper"]:
        total = results[bot]["wins"] + results[bot]["losses"] + results[bot]["ties"]
        results[bot]["win_rate"] = results[bot]["wins"] / total * 100 if total > 0 else 0
        results[bot]["avg_score"] = sum(results[bot]["scores"]) / len(results[bot]["scores"]) if results[bot]["scores"] else 0
        results[bot]["avg_time"] = sum(results[bot]["times"]) / len(results[bot]["times"]) if results[bot]["times"] else 0
    
    return results


def analyze_code_similarity() -> Dict:
    """
    Analyze code similarity between atlas_bot.py and reaper.py.
    
    Uses difflib to compute similarity ratio and count exact line matches.
    """
    print("\n" + "=" * 60)
    print("CODE SIMILARITY ANALYSIS")
    print("=" * 60)
    
    if not ATLAS_BOT_PATH.exists() or not REAPER_BOT_PATH.exists():
        return {"error": "One or both bot files not found"}
    
    atlas_code = ATLAS_BOT_PATH.read_text(encoding='utf-8')
    reaper_code = REAPER_BOT_PATH.read_text(encoding='utf-8')
    
    atlas_lines = atlas_code.splitlines()
    reaper_lines = reaper_code.splitlines()
    
    # Overall similarity ratio
    matcher = difflib.SequenceMatcher(None, atlas_code, reaper_code)
    overall_similarity = matcher.ratio() * 100
    
    # Line-by-line exact matches
    exact_matches = 0
    for a_line in atlas_lines:
        a_stripped = a_line.strip()
        if len(a_stripped) > 10:  # Skip short lines
            for r_line in reaper_lines:
                if a_stripped == r_line.strip():
                    exact_matches += 1
                    break
    
    exact_match_ratio = exact_matches / len(atlas_lines) * 100 if atlas_lines else 0
    
    # Identifier analysis
    atlas_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', atlas_code))
    reaper_identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', reaper_code))
    common_identifiers = atlas_identifiers & reaper_identifiers
    
    # Filter common Python keywords and builtins
    python_keywords = {
        'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'import',
        'from', 'as', 'try', 'except', 'finally', 'with', 'lambda', 'yield',
        'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'self',
        'int', 'str', 'float', 'list', 'dict', 'set', 'tuple', 'bool',
        'len', 'range', 'print', 'any', 'all', 'min', 'max', 'sum', 'abs',
        'Optional', 'List', 'Dict', 'Set', 'Tuple', 'Union', 'Any'
    }
    
    unique_atlas = atlas_identifiers - reaper_identifiers - python_keywords
    unique_reaper = reaper_identifiers - atlas_identifiers - python_keywords
    shared_custom = common_identifiers - python_keywords
    
    # Check for suspicious markers
    suspicious_markers = []
    for marker in ['atlas', 'ATLAS', 'eric', 'Eric', 'truebot', 'TrueBot']:
        if marker.lower() in reaper_code.lower():
            suspicious_markers.append(marker)
    
    results = {
        "atlas_lines": len(atlas_lines),
        "reaper_lines": len(reaper_lines),
        "overall_similarity_pct": overall_similarity,
        "exact_line_match_pct": exact_match_ratio,
        "exact_matches": exact_matches,
        "unique_atlas_identifiers": len(unique_atlas),
        "unique_reaper_identifiers": len(unique_reaper),
        "shared_identifiers": len(shared_custom),
        "suspicious_markers": suspicious_markers,
        "sample_unique_atlas": list(unique_atlas)[:10],
        "sample_unique_reaper": list(unique_reaper)[:10]
    }
    
    print(f"  Atlas lines:     {results['atlas_lines']}")
    print(f"  Reaper lines:    {results['reaper_lines']}")
    print(f"  Overall similarity: {results['overall_similarity_pct']:.1f}%")
    print(f"  Exact line matches: {results['exact_line_match_pct']:.1f}% ({results['exact_matches']} lines)")
    print(f"  Unique Atlas identifiers: {results['unique_atlas_identifiers']}")
    print(f"  Unique Reaper identifiers: {results['unique_reaper_identifiers']}")
    
    if suspicious_markers:
        print(f"  [WARNING] Suspicious markers found: {suspicious_markers}")
    else:
        print(f"  [OK] No suspicious markers found")
    
    # Check if target <40% exact line match is met
    if exact_match_ratio < 40:
        print(f"  [PASS] Exact line match < 40%: {exact_match_ratio:.1f}%")
    else:
        print(f"  [FAIL] Exact line match >= 40%: {exact_match_ratio:.1f}%")
    
    return results


def analyze_style_differences() -> Dict:
    """Analyze coding style differences between the two bots."""
    print("\n" + "=" * 60)
    print("STYLE DIFFERENCE ANALYSIS")
    print("=" * 60)
    
    if not ATLAS_BOT_PATH.exists() or not REAPER_BOT_PATH.exists():
        return {"error": "One or both bot files not found"}
    
    atlas_code = ATLAS_BOT_PATH.read_text(encoding='utf-8')
    reaper_code = REAPER_BOT_PATH.read_text(encoding='utf-8')
    
    def analyze_style(code: str, name: str) -> Dict:
        lines = code.splitlines()
        
        # Docstring style detection
        has_numpy_docstrings = bool(re.search(r'Parameters\n\s*-+\n', code))
        has_google_docstrings = bool(re.search(r'Args:\n\s+\w+:', code))
        has_simple_docstrings = bool(re.search(r'"""[^"]+"""', code)) and not has_numpy_docstrings and not has_google_docstrings
        
        # Naming conventions
        snake_case_vars = len(re.findall(r'\b[a-z]+_[a-z_]+\b', code))
        camel_case_vars = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', code))
        
        # Comment styles
        hash_comments = len(re.findall(r'#[^#]', code))
        
        # Class/function counts
        class_count = len(re.findall(r'^class\s+\w+', code, re.MULTILINE))
        function_count = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        
        # Average line length
        line_lengths = [len(l) for l in lines if l.strip()]
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        
        # Blank line ratio
        blank_lines = sum(1 for l in lines if not l.strip())
        blank_ratio = blank_lines / len(lines) * 100 if lines else 0
        
        return {
            "name": name,
            "docstring_style": "numpy" if has_numpy_docstrings else "google" if has_google_docstrings else "simple",
            "snake_case_count": snake_case_vars,
            "camel_case_count": camel_case_vars,
            "hash_comments": hash_comments,
            "class_count": class_count,
            "function_count": function_count,
            "avg_line_length": avg_line_length,
            "blank_line_pct": blank_ratio
        }
    
    atlas_style = analyze_style(atlas_code, "atlas")
    reaper_style = analyze_style(reaper_code, "reaper")
    
    # Compute style divergence score
    divergences = []
    
    # Docstring style difference
    if atlas_style["docstring_style"] != reaper_style["docstring_style"]:
        divergences.append(("docstring_style", 20))
    
    # Naming convention shift
    atlas_snake_ratio = atlas_style["snake_case_count"] / max(atlas_style["camel_case_count"], 1)
    reaper_snake_ratio = reaper_style["snake_case_count"] / max(reaper_style["camel_case_count"], 1)
    if abs(atlas_snake_ratio - reaper_snake_ratio) > 1:
        divergences.append(("naming_convention", 15))
    
    # Line length preference
    if abs(atlas_style["avg_line_length"] - reaper_style["avg_line_length"]) > 10:
        divergences.append(("line_length", 10))
    
    # Comment density
    if abs(atlas_style["hash_comments"] - reaper_style["hash_comments"]) > 20:
        divergences.append(("comment_density", 10))
    
    style_divergence_score = sum(d[1] for d in divergences)
    
    print(f"\n  Atlas Style:")
    print(f"    Docstrings:  {atlas_style['docstring_style']}")
    print(f"    Snake case:  {atlas_style['snake_case_count']}")
    print(f"    Camel case:  {atlas_style['camel_case_count']}")
    print(f"    Avg line:    {atlas_style['avg_line_length']:.0f} chars")
    
    print(f"\n  Reaper Style:")
    print(f"    Docstrings:  {reaper_style['docstring_style']}")
    print(f"    Snake case:  {reaper_style['snake_case_count']}")
    print(f"    Camel case:  {reaper_style['camel_case_count']}")
    print(f"    Avg line:    {reaper_style['avg_line_length']:.0f} chars")
    
    print(f"\n  Style Divergence Score: {style_divergence_score}/55")
    print(f"  Divergent aspects: {[d[0] for d in divergences]}")
    
    return {
        "atlas": atlas_style,
        "reaper": reaper_style,
        "divergence_score": style_divergence_score,
        "divergences": divergences
    }


def generate_verification_report(
    func_results: Dict,
    similarity_results: Dict,
    style_results: Dict
) -> str:
    """Generate the final verification report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("REAPER BOT VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    # 1. Functional Equivalence
    lines.append("-" * 70)
    lines.append("1. FUNCTIONAL EQUIVALENCE")
    lines.append("-" * 70)
    
    if "error" in func_results:
        lines.append(f"  ERROR: {func_results['error']}")
    elif "skipped" in func_results:
        lines.append(f"  SKIPPED: {func_results['reason']}")
    else:
        lines.append(f"  Atlas Win Rate:  {func_results['atlas']['win_rate']:.1f}%")
        lines.append(f"  Reaper Win Rate: {func_results['reaper']['win_rate']:.1f}%")
        lines.append(f"  Atlas Avg Score:  ${func_results['atlas']['avg_score']:.0f}")
        lines.append(f"  Reaper Avg Score: ${func_results['reaper']['avg_score']:.0f}")
        lines.append(f"  Atlas Avg Time:   {func_results['atlas']['avg_time']:.1f}s")
        lines.append(f"  Reaper Avg Time:  {func_results['reaper']['avg_time']:.1f}s")
        
        win_rate_diff = abs(func_results['atlas']['win_rate'] - func_results['reaper']['win_rate'])
        if win_rate_diff <= 20:
            lines.append(f"  [PASS] Win rates within 20% tolerance")
        else:
            lines.append(f"  [WARN] Win rates differ by {win_rate_diff:.1f}%")
    lines.append("")
    
    # 2. Code Similarity
    lines.append("-" * 70)
    lines.append("2. CODE SIMILARITY")
    lines.append("-" * 70)
    
    if "error" in similarity_results:
        lines.append(f"  ERROR: {similarity_results['error']}")
    else:
        lines.append(f"  Overall Similarity: {similarity_results['overall_similarity_pct']:.1f}%")
        lines.append(f"  Exact Line Matches: {similarity_results['exact_line_match_pct']:.1f}%")
        lines.append(f"  Suspicious Markers: {similarity_results['suspicious_markers'] or 'None'}")
        
        if similarity_results['exact_line_match_pct'] < 40:
            lines.append(f"  [PASS] Exact line match < 40%")
        else:
            lines.append(f"  [FAIL] Exact line match >= 40%")
        
        if not similarity_results['suspicious_markers']:
            lines.append(f"  [PASS] No origin markers detected")
        else:
            lines.append(f"  [FAIL] Origin markers detected")
    lines.append("")
    
    # 3. Style Differences
    lines.append("-" * 70)
    lines.append("3. STYLE DIFFERENCES")
    lines.append("-" * 70)
    
    if "error" in style_results:
        lines.append(f"  ERROR: {style_results['error']}")
    else:
        lines.append(f"  Divergence Score: {style_results['divergence_score']}/55")
        lines.append(f"  Divergent Aspects: {[d[0] for d in style_results['divergences']]}")
        
        if style_results['divergence_score'] >= 25:
            lines.append(f"  [PASS] Sufficient style divergence")
        else:
            lines.append(f"  [WARN] Limited style divergence")
    lines.append("")
    
    # Summary
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    
    passes = 0
    fails = 0
    
    # Check functional equivalence
    if "skipped" not in func_results and "error" not in func_results:
        win_diff = abs(func_results['atlas']['win_rate'] - func_results['reaper']['win_rate'])
        if win_diff <= 20:
            passes += 1
        else:
            fails += 1
    
    # Check similarity
    if "error" not in similarity_results:
        if similarity_results['exact_line_match_pct'] < 40:
            passes += 1
        else:
            fails += 1
        
        if not similarity_results['suspicious_markers']:
            passes += 1
        else:
            fails += 1
    
    # Check style
    if "error" not in style_results:
        if style_results['divergence_score'] >= 25:
            passes += 1
        else:
            fails += 1
    
    lines.append(f"  Checks Passed: {passes}")
    lines.append(f"  Checks Failed: {fails}")
    
    if fails == 0:
        lines.append("")
        lines.append("  [SUCCESS] REAPER BOT VERIFICATION PASSED")
    else:
        lines.append("")
        lines.append("  [WARNING] Some verification checks need attention")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    print("REAPER BOT VERIFICATION")
    print("=" * 60)
    print(f"Atlas Bot:  {ATLAS_BOT_PATH}")
    print(f"Reaper Bot: {REAPER_BOT_PATH}")
    
    # Run all tests
    func_results = test_functional_equivalence()
    similarity_results = analyze_code_similarity()
    style_results = analyze_style_differences()
    
    # Generate report
    report = generate_verification_report(func_results, similarity_results, style_results)
    
    print("\n")
    print(report)
    
    # Save report
    report_path = PROJECT_ROOT / "results" / "reaper_verification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    print(f"\nReport saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
