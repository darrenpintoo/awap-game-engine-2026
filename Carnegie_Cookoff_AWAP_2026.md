# Welcome to Carnegie Cookoff

Welcome to Carnegie Cookoff!

Hello! We are so excited to see you at **Algorithms with a Purpose 2026**, hosted by **ACM@CMU**. This document contains the game rules, a reference manual for your code, and a list of the in-game constants.

If you have any questions feel free to ask one of our AWAP staff at Office Hours or on Discord.

---

## Getting Started

### Important Links

- **Dashboard:** https://dashboard.awap.acmatcmu.com  
- **Discord:** https://discord.gg/2dcBVWMw  

If you weren’t automatically assigned the **@AWAP 2026** role, it can be claimed from `#role-select`. Scroll down or look in the channel list if you don’t see it! We’ll post an `@everyone` when the event starts as well.

We also encourage you to drop a cool message in `#intro` and follow your fellow community members :)

For this event, please join:
- `#awap-2026` for announcements  
- `#crew-log` for public questions and memes  
- `#support` for anything individual  

---

## Development & Submission Flow

### Local Development

1. Clone the game engine:  
   https://github.com/acm-cmu/awap-game-engine-2026-public
2. Write your bot and add it to the `bots/` directory.
3. Put any maps you’d like to test with in the `maps/` directory.
4. Specify players & maps in `config.json`, then run:
   ```
   python3 run_game.py -c config.json
   ```
   - Use `--render` to see live gameplay  
   - Use `--replay sample.json` to save a replay
5. View replays:
   ```
   python replay_game.py <file>.awap26r
   ```
6. Command-line replay viewer:
   ```
   python replay_game_cli.py <file>.awap26r
   ```

Extracting the `.gzip` archive is optional.

### Using a Codespace

- Upgrade GitHub using the **Student Developer Pack**: https://education.github.com/pack
- Use your `andrew.cmu.edu` email
- Open the repo → **Code** → **Codespaces**
- Start a new Codespace (VS Code in the browser)
- Follow the same development steps as local
- Run `git pull` periodically for hotfixes

This setup is recommended to keep environments consistent and debugging easier.

### Submission to Competition

- Log into the dashboard: https://dashboard.awap.acmatcmu.com
- Upload your bot
- Request unranked scrimmages
- Review results and match history
- Check leaderboard
- Download replays

---

## Game Overview

### Background and Objective

You open your doors with a dream and a burner stove. Across the street, your rival does the same. Orders flood in. Tempers flare. Claws come out, teeth bared. The broth gets hotter.

Make money. Complete orders. **PURR-fect your noodles.** Build your empire.

When night falls, sabotage your enemy and create a **CAT-astrophe** — but beware. Every hiss echoes back.

When the dust settles, only one shop will still be standing.

**Welcome to the NOODLE War.**

Each map represents a kitchen layout. You can cook, clean, and sabotage your opponent.

### Food Types

There are 5 ingredients:
- Eggs
- Onions
- Meat
- Noodles
- Sauce

Some ingredients can be chopped and/or cooked.

### Actions

Bots may:
- Move
- Pick up items
- Place items
- Trash ingredients
- Buy from shop
- Chop food
- Cook food
- Wash and collect dishes
- Switch maps to sabotage

### Map Overview

Maps are grid-based, ranging from **6×6 to 48×48**.

Each map contains kitchen elements like sinks, counters, boxes, trash, cookers, and submit stations.

### Gameplay Details

- Teams of **two bots**
- Starting coins: **150**
- Passive income: **1 coin per turn**
- Code runtime limit: **0.5 seconds per turn**
- One move **and** one action per bot per turn

Turn order:
1. Passive income
2. Cooking progress
3. Order expiration
4. Forced return from sabotage if timed out
5. Player code execution

### Tie Breaks

- Highest coin count wins
- Ties result in draws (or replays during ranked scrimmages)

### Allowed Packages

- Python standard library
- NumPy
- SciPy
- Anything already imported by the game engine

All others are disallowed unless approved.

---

## API Documentation

Game engine repo:  
https://github.com/acm-cmu/awap-game-engine-2026-public

Do **not** attempt to exploit internal state. This may result in disqualification.

### RobotController

#### State Access
- `get_turn() -> int`
- `get_team() -> Team`
- `get_enemy_team() -> Team`
- `get_map() -> Map`
- `get_orders() -> List[Dict]`
- `get_team_bot_ids() -> List[int]`
- `get_team_money() -> int`
- `get_bot_state(bot_id: int) -> Optional[Dict]`
- `get_tile(team: Team, x: int, y: int) -> Optional[Tile]`

#### Movement
- `can_move(bot_id, dx, dy) -> bool`
- `move(bot_id, dx, dy) -> bool`

#### Inventory
- `pickup(...)`
- `place(...)`
- `trash(...)`

#### Shop
- `can_buy(...)`
- `buy(...)`

#### Food Processing
- `chop(...)`
- `can_start_cook(...)`
- `start_cook(...)`
- `take_from_pan(...)`

#### Plates
- `take_clean_plate(...)`
- `put_dirty_plate_in_sink(...)`
- `wash_sink(...)`
- `add_food_to_plate(...)`

#### Submit
- `can_submit(...)`
- `submit(...)`

#### Switch
- `get_switch_info()`
- `can_switch_maps()`
- `switch_maps()`

---

## Game Constants

### Ingredients

| Ingredient | ID | Choppable | Cookable | Cost |
|-----------|----|-----------|----------|------|
| Egg | 0 | False | True | 20 |
| Onion | 1 | True | False | 30 |
| Meat | 2 | True | True | 80 |
| Noodles | 3 | False | False | 40 |
| Sauce | 4 | False | False | 10 |

### Items

| Item | Cost |
|-----|------|
| Plate | 2 |
| Pan | 4 |
