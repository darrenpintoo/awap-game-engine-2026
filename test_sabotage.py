import subprocess
import os

RED_BOT = "bots/dpinto/ultimate_champion_bot.py" # The Efficiency King
BLUE_BOT = "bots/dpinto/champion_sabotage_optimized.py" # The Bandit
MAP = "maps/eric/map6_overload.txt"

def run_test():
    print(f"Running SABOTAGE TEST: {os.path.basename(BLUE_BOT)} vs {os.path.basename(RED_BOT)}")
    cmd = [
        "python3", "src/game.py",
        "--red", RED_BOT,
        "--blue", BLUE_BOT,
        "--map", MAP,
        "--turns", "300" # Longer game to let sabotage play out
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
