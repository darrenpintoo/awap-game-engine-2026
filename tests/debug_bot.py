
import sys
import os

# Add src to path to import game engine modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from game import Game
from game_constants import Team
from robot_controller import RobotController

class DebugGame(Game):
    def __init__(self, red_bot_path, blue_bot_path, map_path):
        super().__init__(red_bot_path, blue_bot_path, map_path)
        self.map_path = map_path
        self.red_bot_path = red_bot_path
        self.blue_bot_path = blue_bot_path
        self.log_file = open("debug_log.txt", "w")

    def log(self, message):
        self.log_file.write(f"[Turn {self.game_state.turn}] {message}\n")
        self.log_file.flush()
        print(f"[Turn {self.game_state.turn}] {message}")

    def run_debug_game(self, max_turns=500):
        print(f"Starting Debug Game on {self.map_path}")
        self.log(f"Initialized game with Red: {self.red_bot_path}, Blue: {self.blue_bot_path}")
        
        while self.game_state.turn < max_turns:
            # 1. Inspect State Before Tick
            self._log_bot_states(Team.RED)
            self._log_bot_states(Team.BLUE)
            
            # 2. Run Tick
            try:
                self.game_state.start_turn()
                self.call_player(Team.RED)
                self.call_player(Team.BLUE)
            except Exception as e:
                self.log(f"CRASH: {e}")
                break
            
            if self.game_state.turn % 50 == 0:
                print(f"Turn {self.game_state.turn} completed.")

        self.log_file.close()

    def _log_bot_states(self, team):
        # Access internal state (cheating slightly for debug)
        bots = self.game_state.bots
        team_bots = [b for b_id, b in bots.items() if (b_id < 2 and team == Team.RED) or (b_id >= 2 and team == Team.BLUE)]
        
        orders = self.game_state.orders[team]
        
        for bot in team_bots:
            short_holding = "None"
            if bot.holding:
                if hasattr(bot.holding, 'food_name'):
                     short_holding = f"{bot.holding.food_name}"
                     if hasattr(bot.holding, 'chopped') and bot.holding.chopped: short_holding += "(CHOPPED)"
                     if hasattr(bot.holding, 'cooked_stage'): short_holding += f"(Ck:{bot.holding.cooked_stage})"
                elif hasattr(bot.holding, 'item_name'):
                     short_holding = bot.holding.item_name
                else: # Plate
                     contents = [f"{f.food_name}({'Ck' if f.cooked_stage==1 else 'Raw'})" for f in bot.holding.food]
                     short_holding = f"Plate{contents}"

            self.log(f"Bot {bot.bot_id} ({team.name}): Pos=({bot.x},{bot.y}) Holding={short_holding}")
            
            # Check if holding plate and attempting submit
            # (Note: We can't see 'intent' here easily without hooking the bot, but we can see if they are AT submit with a Plate)
            try:
                tile = self.game_state.red_map.tiles[bot.x][bot.y] if team == Team.RED else self.game_state.blue_map.tiles[bot.x][bot.y]
                if getattr(tile, 'tile_name', '') == 'SUBMIT' and short_holding.startswith("Plate"):
                     # Check against orders
                     self.log(f"    -> At SUBMIT with Plate. Active Orders: {[o.required for o in orders]}")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Run AWAP Debug Game')
    parser.add_argument('--map', type=str, required=True, help='Path to map file')
    parser.add_argument('--bot', type=str, help='Path to bot file (plays both if --red/--blue not set)')
    parser.add_argument('--red', type=str, help='Path to red bot file')
    parser.add_argument('--blue', type=str, help='Path to blue bot file')
    parser.add_argument('--turns', type=int, default=200, help='Max turns')
    args = parser.parse_args()

    red = args.red if args.red else args.bot
    blue = args.blue if args.blue else args.bot
    
    if not red or not blue:
        print("Error: Must specify --bot or both --red and --blue")
        return

    g = DebugGame(red, blue, args.map) 
    g.run_debug_game(args.turns)

if __name__ == "__main__":
    main()
