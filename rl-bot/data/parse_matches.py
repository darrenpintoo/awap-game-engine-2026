"""
Match Data Parser
=================
Extracts state-action pairs from tournament JSON for imitation learning.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Any

def load_tournament_data(json_path: str) -> Dict[str, Any]:
    """Load tournament results JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_winning_bot_actions(data: Dict) -> List[Dict]:
    """
    Extract match outcomes to identify winning strategies.
    Returns list of matches with winner info.
    """
    winning_matches = []
    
    for match in data.get('matches', []):
        if match['winner'] not in ['DRAW', 'ERROR', 'TIMEOUT']:
            winning_matches.append({
                'winner': match['winner'],
                'winner_score': match['red_score'] if match['winner'] == match['red_name'] else match['blue_score'],
                'loser_score': match['blue_score'] if match['winner'] == match['red_name'] else match['red_score'],
                'margin': abs(match['red_score'] - match['blue_score']),
                'map': match['map_name'],
                'duration': match['duration']
            })
    
    return winning_matches

def analyze_bot_performance(data: Dict) -> Dict[str, Dict]:
    """
    Aggregate statistics per bot for reward shaping.
    """
    stats = {}
    
    for match in data.get('matches', []):
        for player, score in [(match['red_name'], match['red_score']), 
                               (match['blue_name'], match['blue_score'])]:
            if player not in stats:
                stats[player] = {
                    'total_games': 0,
                    'wins': 0,
                    'total_score': 0,
                    'scores': []
                }
            
            stats[player]['total_games'] += 1
            stats[player]['total_score'] += score
            stats[player]['scores'].append(score)
            
            if match['winner'] == player:
                stats[player]['wins'] += 1
    
    # Calculate derived stats
    for bot in stats:
        games = stats[bot]['total_games']
        stats[bot]['win_rate'] = stats[bot]['wins'] / games if games > 0 else 0
        stats[bot]['avg_score'] = stats[bot]['total_score'] / games if games > 0 else 0
        stats[bot]['score_std'] = np.std(stats[bot]['scores']) if stats[bot]['scores'] else 0
    
    return stats

def get_top_performers(stats: Dict, n: int = 5) -> List[str]:
    """Get top N bots by win rate."""
    sorted_bots = sorted(
        stats.items(),
        key=lambda x: (x[1]['win_rate'], x[1]['avg_score']),
        reverse=True
    )
    return [bot[0] for bot in sorted_bots[:n]]

def generate_reward_function_params(stats: Dict) -> Dict:
    """
    Analyze tournament data to derive reward function parameters.
    """
    all_scores = []
    all_margins = []
    
    for bot_stats in stats.values():
        all_scores.extend(bot_stats['scores'])
    
    return {
        'score_mean': np.mean(all_scores),
        'score_std': np.std(all_scores),
        'score_max': max(all_scores),
        # These can be used to normalize rewards during training
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "../tournament-dashboard/public/data.json"
    
    print(f"Loading tournament data from: {json_path}")
    data = load_tournament_data(json_path)
    
    print(f"\nTotal matches: {data['metadata']['completed_matches']}")
    
    stats = analyze_bot_performance(data)
    top_bots = get_top_performers(stats)
    
    print("\n=== Top 5 Performers ===")
    for i, bot in enumerate(top_bots, 1):
        s = stats[bot]
        print(f"{i}. {bot}: WR={s['win_rate']*100:.1f}%, Avg=${s['avg_score']:.0f}")
    
    params = generate_reward_function_params(stats)
    print(f"\n=== Reward Function Parameters ===")
    print(f"Score Mean: ${params['score_mean']:.0f}")
    print(f"Score Std: ${params['score_std']:.0f}")
    print(f"Score Max: ${params['score_max']:.0f}")
