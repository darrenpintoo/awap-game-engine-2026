
import os
import subprocess
import json
import time

# --- Config ---
BASE_DIR = os.getcwd()
DASHBOARD_DATA_PATH = os.path.join(BASE_DIR, "tournament-dashboard/public/data.json")
TURNS = 200

BOTS = [
    # Top Tier / Main Comparisons
    "bots/eric/truebot.py",
    "bots/dpinto/JuniorChampion.py",
    "testbots/test.py",
    
    # Complex Test Bots
    "testbots/bot-Burger-2026-02-01T02-27-09-2dd8d56a-a89e-44a5-b5ef-c9131ba69be0.py",
    "testbots/bot-Noodleheads-2026-02-01T05-40-38-9e379cec-19c7-4005-a542-dbb3067954c0.py",
    "testbots/bot-robocc-2026-02-01T07-33-15-505ba463-69e7-4818-a269-029515b7499f.py",
    
    # Test Bots (bots/test-bots/)
    "bots/test-bots/adaptive_switcher.py",
    "bots/test-bots/aggressive_saboteur.py",
    "bots/test-bots/do_nothing_bot.py",
    "bots/test-bots/efficiency_maximizer.py",
    "bots/test-bots/greedy_picker.py",
    "bots/test-bots/rush_bot.py",
    "bots/test-bots/turtle_defender.py",
    "bots/test-bots/zone_coordinator.py"
]

MAPS = [
    "maps/eric/throughput.txt",
    "maps/eric/map5_grind.txt"
]

# Ensure paths exist
BOTS = [b for b in BOTS if os.path.exists(os.path.join(BASE_DIR, b))]
MAPS = [m for m in MAPS if os.path.exists(os.path.join(BASE_DIR, m))]

print(f"Loaded {len(BOTS)} bots and {len(MAPS)} maps.")


def load_data():
    if os.path.exists(DASHBOARD_DATA_PATH):
        try:
            with open(DASHBOARD_DATA_PATH, 'r') as f:
                data = json.load(f)
            return data
        except:
            return None
    return None

def init_data():
    data = load_data()
    if data is None:
        data = {
            "metadata": {
                "total_matches": 0,
                "completed_matches": 0,
                "last_updated": time.time()
            },
            "matches": []
        }
    return data

def save_data(data):
    try:
        data["metadata"]["last_updated"] = time.time()
        with open(DASHBOARD_DATA_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving data: {e}")

def get_bot_name(path):
    return os.path.basename(path)

def run_match(red_bot, blue_bot, map_path):
    cmd = [
        "python3", "src/game.py",
        "--red", red_bot,
        "--blue", blue_bot,
        "--map", map_path,
        "--turns", str(TURNS)
    ]
    
    try:
        # print(f"Running {get_bot_name(red_bot)} vs {get_bot_name(blue_bot)} on {os.path.basename(map_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout
        
        scores_line = [line for line in output.split('\n') if "money scores:" in line]
        if not scores_line:
            # Check for crashes/timeouts if no score line
            # print("  > Failed to get scores")
            return None
            
        line = scores_line[0]
        parts = line.split("scores:")[1].split(",")
        red_score = int(parts[0].split("=$")[1])
        blue_score = int(parts[1].split("=$")[1])
        
        winner = "DRAW"
        if red_score > blue_score: winner = get_bot_name(red_bot)
        elif blue_score > red_score: winner = get_bot_name(blue_bot)
        
        # Parse duration (assuming it's roughly execution time or simulated turns, 
        # but game.py doesn't output time directly in easily parsable standard way usually, 
        # so we'll approximate or check if game.py outputs it. 
        # Dashboard expects duration. Let's infer or fake it if missing, 
        # but optimally we'd parse "Game took X seconds")
        duration = 0.0 # Placeholder
        
        return {
            "red_name": get_bot_name(red_bot),
            "blue_name": get_bot_name(blue_bot),
            "map_name": map_path, # Dashboard expects relative path usually or just name
            "map_type": "official",
            "winner": winner,
            "duration": duration,
            "red_score": red_score,
            "blue_score": blue_score,
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        print(f"  > Match Error: {e}")
        return None

def main():
    data = init_data()
    
    # Calculate total remaining matches
    # We will run Red vs Blue for every pair (including self if we wanted, but usually distinct)
    # Let's do Full Round Robin excluding self-play for now
    queue = []
    for m in MAPS:
        for i in range(len(BOTS)):
            for j in range(len(BOTS)):
                if i == j: continue # Skip self play
                queue.append((BOTS[i], BOTS[j], m))
                
    data["metadata"]["total_matches"] += len(queue)
    save_data(data)
    
    print(f"queued {len(queue)} matches.")
    
    completed = 0
    for red, blue, map_path in queue:
        res = run_match(red, blue, map_path)
        if res:
            data["matches"].append(res)
            data["metadata"]["completed_matches"] += 1
            completed += 1
            if completed % 1 == 0: # Save every match to see live updates
                save_data(data)
                print(f"[{completed}/{len(queue)}] {res['red_name']} vs {res['blue_name']}: {res['red_score']}-{res['blue_score']} ({res['winner']})")
        else:
            print(f"[{completed}/{len(queue)}] MATCH FAILED: {get_bot_name(red)} vs {get_bot_name(blue)}")
            
    print("Gauntlet Complete.")

if __name__ == "__main__":
    main()
