# AWAP 2026: Carnegie Cookoff - Complete Game Mechanics & Strategy Report

## Table of Contents
1. [Game Overview](#1-game-overview)
2. [Core Mechanics](#2-core-mechanics)
3. [Map Structure & Tiles](#3-map-structure--tiles)
4. [Items & Economy](#4-items--economy)
5. [Bot API Reference](#5-bot-api-reference)
6. [Order System](#6-order-system)
7. [Mid-Game Switch Mechanic](#7-mid-game-switch-mechanic)
8. [Current Bot Strategy](#8-current-bot-strategy)
9. [Known Weaknesses & Open Questions](#9-known-weaknesses--open-questions)

---

## 1. Game Overview

**Carnegie Cookoff** is a turn-based competitive cooking game where two teams (RED and BLUE) race to fulfill food orders and maximize their money score.

| Parameter | Value |
|-----------|-------|
| Total Turns | 500 |
| Teams | 2 (RED vs BLUE) |
| Bots per Team | 2 (configurable by map) |
| Starting Money | $150 per team |
| Passive Income | +$1 per team per turn |

**Win Condition**: Team with the highest `team_money` at turn 500 wins.

---

## 2. Core Mechanics

### 2.1 Turn Structure

Each turn proceeds as follows:
1. **Environment Tick**: Cooking progress advances, dishes wash automatically if sink is active
2. **Order Expiry Check**: Expired orders deduct penalty from team money
3. **Switch Window Check**: If switch window ended, teleport bots back to home map
4. **Bot Actions**: Each bot can perform **1 move + 1 action**

### 2.2 Movement

- **Distance Metric**: Chebyshev distance (diagonal movement allowed)
- **Max Movement**: 1 tile per turn (any of 8 directions)
- **Collision**: Cannot move onto a tile occupied by another bot
- **Walkable Tiles**: Only `FLOOR` and `SUBMIT` tiles are walkable

```
Valid moves from position X:
  [↖][↑][↗]
  [←][X][→]
  [↙][↓][↘]
```

### 2.3 Actions

Actions must target a tile within **Chebyshev distance 1** (including the bot's own tile). Actions include:
- `pickup()`, `place()`, `trash()`
- `buy()`, `chop()`, `start_cook()`, `take_from_pan()`
- `take_clean_plate()`, `add_food_to_plate()`, `wash_sink()`, `put_dirty_plate_in_sink()`
- `submit()`
- `switch_maps()` (special, does not consume action)

### 2.4 Holding Items

- Each bot can hold **exactly 1 item** at a time
- Items include: `Food`, `Plate`, `Pan`
- Plates can contain multiple foods

---

## 3. Map Structure & Tiles

### 3.1 Map Format

Maps are ASCII text files. Example:
```
################
#...C.....$...b#
#...K.....U.R..#
#...S.....T.B..#
#......b.......#
################

SWITCH: turn=250 duration=100

ORDERS:
start=0  duration=200  required=NOODLES,MEAT  reward=10000 penalty=3
```

### 3.2 Tile Types

| Symbol | Tile Name | Walkable | Placeable | Interactable | Description |
|--------|-----------|----------|-----------|--------------|-------------|
| `.` | FLOOR | ✅ | ❌ | ❌ | Basic walkable tile |
| `#` | WALL | ❌ | ❌ | ❌ | Impassable barrier |
| `C` | COUNTER | ❌ | ✅ | ❌ | Place items, chop food here |
| `K` | COOKER | ❌ | ✅ | ✅ | Pan goes here for cooking |
| `S` | SINK | ❌ | ✅ | ✅ | Dirty plates go here for washing |
| `T` | SINKTABLE | ❌ | ❌ | ✅ | Clean plates appear here after washing |
| `R` | TRASH | ❌ | ✅ | ❌ | Dispose of items (clears plate/pan contents) |
| `U` | SUBMIT | ✅ | ✅ | ✅ | Turn in completed orders here |
| `$` | SHOP | ❌ | ❌ | ✅ | Buy ingredients and equipment |
| `B` | BOX | ❌ | ✅ | ✅ | Stackable storage (multiple identical items) |
| `b` | Bot Spawn | ✅ | ❌ | ❌ | Starting positions for bots |

### 3.3 Important Spatial Rules

- **Interaction Range**: Must be within Chebyshev distance 1 to interact
- **Bots cannot stand on non-walkable tiles**
- **Each map is independent**: RED and BLUE have separate, identical maps

---

## 4. Items & Economy

### 4.1 Food Types

| Food | Cost | Can Chop | Can Cook | Notes |
|------|------|----------|----------|-------|
| EGG | $20 | ❌ | ✅ | Cook only |
| ONIONS | $30 | ✅ | ❌ | Chop only |
| MEAT | $80 | ✅ | ✅ | Must chop then cook |
| NOODLES | $40 | ❌ | ❌ | No processing needed |
| SAUCE | $10 | ❌ | ❌ | No processing needed |

### 4.2 Equipment

| Item | Cost | Description |
|------|------|-------------|
| PLATE | $2 | Required for plating and submitting orders |
| PAN | $4 | Required for cooking on the cooker |

### 4.3 Cooking System

1. **Place Pan on Cooker** (via `place()`)
2. **Add Food to Pan** (via `place()` while holding cookable food)
3. **Automatic Cooking**: Each turn, `cook_progress` increments by 1
4. **Cooked Stage Transitions**:
   - `cook_progress = 0-19`: Stage 0 (raw)
   - `cook_progress = 20`: Stage 1 (cooked) ✅
   - `cook_progress = 40+`: Stage 2 (burned) ❌

> **CRITICAL**: Cooking is automatic and CANNOT be paused. You must retrieve food at exactly the right time.

### 4.4 Plate System

1. **Clean Plates**: Come from `SINKTABLE` (via `take_clean_plate()`)
2. **Adding Food**: Use `add_food_to_plate()` - bot can hold plate and target food, or hold food and target plate
3. **Submitting**: Plate must be clean and contain exact ingredients for an order
4. **After Submission**: Plate becomes dirty and goes to SINK automatically
5. **Washing**: Bot uses `wash_sink()` on SINK tile. Each turn washing is active, `PLATE_WASH_PROGRESS` (2) increments. At 2, one dirty plate becomes clean and appears at SINKTABLE.

---

## 5. Bot API Reference

### 5.1 State Query Methods

```python
controller.get_turn() -> int                    # Current turn number
controller.get_team() -> Team                   # RED or BLUE
controller.get_enemy_team() -> Team             # The opposing team
controller.get_map() -> Map                     # Deep copy of your map
controller.get_orders() -> List[Dict]           # Active orders for your team
controller.get_team_bot_ids() -> List[int]      # Your bot IDs
controller.get_team_money() -> int              # Shared team money
controller.get_bot_state(bot_id) -> Dict        # Position, holding, etc.
controller.get_tile(team, x, y) -> Tile         # Tile at position
```

### 5.2 Movement

```python
controller.can_move(bot_id, dx, dy) -> bool     # Check if move is valid
controller.move(bot_id, dx, dy) -> bool         # Execute move (-1/0/1 for dx/dy)
```

### 5.3 Item Manipulation

```python
controller.pickup(bot_id, target_x, target_y) -> bool       # Pick up item
controller.place(bot_id, target_x, target_y) -> bool        # Place held item
controller.trash(bot_id, target_x, target_y) -> bool        # Trash held item
```

### 5.4 Shop

```python
controller.can_buy(bot_id, item, target_x, target_y) -> bool
controller.buy(bot_id, item, target_x, target_y) -> bool    # item = FoodType.X or ShopCosts.X
```

### 5.5 Food Processing

```python
controller.chop(bot_id, target_x, target_y) -> bool         # Chop food on COUNTER (bot must NOT hold anything)
controller.start_cook(bot_id, target_x, target_y) -> bool   # Put held food into pan on cooker
controller.take_from_pan(bot_id, target_x, target_y) -> bool # Take food from pan
```

### 5.6 Plating

```python
controller.take_clean_plate(bot_id, target_x, target_y) -> bool      # From SINKTABLE
controller.add_food_to_plate(bot_id, target_x, target_y) -> bool     # Add food to plate
```

### 5.7 Sink

```python
controller.put_dirty_plate_in_sink(bot_id, target_x, target_y) -> bool
controller.wash_sink(bot_id, target_x, target_y) -> bool             # Must call each turn to progress
```

### 5.8 Submit

```python
controller.can_submit(bot_id, target_x, target_y) -> bool
controller.submit(bot_id, target_x, target_y) -> bool
```

---

## 6. Order System

### 6.1 Order Structure

```python
{
    "order_id": int,
    "required": List[str],        # e.g., ["NOODLES", "MEAT"]
    "created_turn": int,
    "expires_turn": int,
    "reward": int,                # Money gained on completion
    "penalty": int,               # Money lost on expiry
    "is_active": bool
}
```

### 6.2 Order Matching Rules

For an order requiring `[NOODLES, MEAT]`:
- **NOODLES**: Must be raw (not chopped, not cooked) - because `can_chop=False, can_cook=False`
- **MEAT**: Must be chopped AND cooked (stage 1) - because `can_chop=True, can_cook=True`

**The matching algorithm:**
```python
for each required food_type:
    expected = (food_id, must_be_chopped=can_chop, must_be_cooked=1 if can_cook else 0)
```

### 6.3 Submission Flow

1. Bot holds a **clean, non-empty Plate**
2. Bot is adjacent to **SUBMIT** tile (`U`)
3. Call `controller.submit(bot_id, submit_x, submit_y)`
4. If plate contents match an active order → **Success**:
   - Team gains `reward` money
   - Plate becomes dirty (auto-sent to nearest SINK)
5. If no match → **Failure**: Action wasted, plate still held

---

## 7. Mid-Game Switch Mechanic

### 7.1 Overview

This is a **sabotage mechanic** allowing teams to temporarily invade the enemy's kitchen.

| Parameter | Default Value |
|-----------|---------------|
| Switch Turn | 250 |
| Switch Duration | 100 turns |

### 7.2 Switch Window

- **Active Window**: Turn 250 through Turn 349 (inclusive)
- **During Window**: Either team can call `switch_maps()` **once per game**
- **Effect**: All bots on that team teleport to the enemy's map

### 7.3 API

```python
controller.get_switch_info() -> Dict
# Returns:
# {
#     "turn": int,
#     "switch_turn": int,
#     "switch_duration": int,
#     "window_active": bool,
#     "window_end_turn": int,
#     "my_team_switched": bool,
#     "enemy_team_switched": bool
# }

controller.can_switch_maps() -> bool
controller.switch_maps() -> bool           # Does NOT consume action
```

### 7.4 After Switch Window Ends

At turn 350:
- All switched bots **automatically teleport back** to their home map
- Bots **keep whatever they are holding** when teleported

### 7.5 Sabotage Strategies

On enemy map, you can:
1. **Steal their Pan** - They can't cook without it (costs $4 to replace)
2. **Take food from their pan** and trash it
3. **Block their SUBMIT station** - Stand on/near it
4. **Pick up their plated food** from counters and trash it

### 7.6 Defense Strategies

When enemy is on your map:
1. **Hold important items** - Don't leave plates/food on counters
2. **Guard your cooker** - Stand adjacent to prevent pickup
3. **Continue washing** - The sink is safe (can't be sabotaged)
4. **Rush submissions** - Get food out before it can be stolen

---

## 8. Current Bot Strategy

### 8.1 Architecture

The bot uses a **centralized role-based system** with two roles:

| Role | Bot Index | Responsibilities |
|------|-----------|------------------|
| **Chef** | Bot 0 | Main cooking loop: buy → chop → cook → plate → submit |
| **Support** | Bot 1+ | Wash dishes, stay out of way, assist with defense |

### 8.2 Pre-Computation (Initialization)

On first turn:
1. **Parse map**: Extract locations of all tile types (shops, cookers, sinks, etc.)
2. **Build distance matrix**: BFS from every walkable tile, storing shortest paths
3. **Result**: O(1) distance lookups during gameplay

### 8.3 Chef State Machine

```
State 0:  Init - check if pan exists on cooker
State 1:  Buy pan → place on cooker
State 2:  Buy meat
State 3:  Place meat on counter
State 4:  Chop meat (bot hands must be empty)
State 5:  Pick up chopped meat
State 6:  Place meat in pan (starts cooking automatically)
State 7:  Buy plate
State 8:  Place plate on counter
State 9:  Buy noodles
State 10: Add noodles to plate
State 11: Wait for cooking → take from pan (when cooked_stage == 1)
State 12: Add cooked meat to plate
State 13: Pick up plate
State 14: Submit order
State 15: Handle errors (trash burnt food, restart)
```

### 8.4 Pathfinding

- **Algorithm**: BFS with 8-directional movement
- **Target**: Find tile adjacent to destination (since you can't stand on most interactable tiles)
- **Returns**: First step `(dx, dy)` toward goal

### 8.5 Cooking Tracker

Each turn, the bot scans all cookers and tracks:
```python
CookingTracker:
    location: (x, y)
    food_name: str
    cook_progress: int
    cooked_stage: int
    turns_to_cooked: int   # 20 - cook_progress
    turns_to_burned: int   # 40 - cook_progress
    is_urgent: bool        # True if ≤3 turns from cooked
```

### 8.6 Sabotage Protocol

**Trigger Conditions** (all must be true):
- Switch window is active
- Team has not already switched
- Current turn > (switch_end - 30)

**Actions on Enemy Map**:
1. Navigate to enemy cooker
2. Pickup their pan (with or without food)
3. Move to corner and hold it

### 8.7 Defense Protocol

**Trigger**: `enemy_team_switched == True`

**Actions**:
- If holding a plate with food → move toward SUBMIT and stay near
- Otherwise → guard the cooker by standing adjacent

---

## 9. Known Weaknesses & Open Questions

### 9.1 Current Strategy Weaknesses

1. **Single-threaded Chef**: Only Bot 0 does cooking. Bot 1+ mostly idles.
2. **No parallel order processing**: Cannot work on multiple orders simultaneously.
3. **Fixed ingredient targeting**: Always makes NOODLES + MEAT, ignores other order types.
4. **Suboptimal plate recycling**: Doesn't proactively wash dishes during downtime.
5. **Naive sabotage timing**: Switches late in window, reducing sabotage time.
6. **No collision avoidance between our bots**: Bots may block each other.

### 9.2 Open Strategic Questions

1. **Multi-order parallelism**: Should Bot 1 start a second order while Bot 0 waits for cooking?
2. **Preemptive ingredient prep**: Should we pre-buy/pre-chop ingredients during cooking wait?
3. **Box utilization**: Boxes can store multiple identical items - useful for batching?
4. **Sabotage ROI**: Is sabotage worth losing ~50 turns of cooking on our own map?
5. **Optimal switch timing**: Early switch = more sabotage time. Late switch = harder to counter. What's optimal?
6. **Counter-sabotage**: If enemy switches first, should we also switch to sabotage back?

### 9.3 Potential Improvements

| Improvement | Expected Impact | Complexity |
|-------------|-----------------|------------|
| Parallel cooking (2 chefs) | +50% throughput | Medium |
| Dynamic order selection | Handle varied orders | Medium |
| Pre-prep during cook wait | -5 turns per order | Low |
| Hungarian assignment | Optimal multi-bot allocation | High |
| Time-Space A* | Zero collisions | High |
| Adaptive sabotage | Counter enemy state | Medium |

---

## Appendix A: Complete Order Fulfillment Example

**Order**: `required=["NOODLES", "MEAT"], reward=10000, penalty=3`

**Required Plate Contents**:
- 1x NOODLES (raw, as-is from shop)
- 1x MEAT (chopped=True, cooked_stage=1)

**Action Sequence** (approximately 45 turns):

| Turn | Action | Result |
|------|--------|--------|
| 1-2 | Move to SHOP | Adjacent to shop |
| 3 | buy(PAN) | Holding: Pan |
| 4-5 | Move to COOKER | Adjacent to cooker |
| 6 | place() | Pan on cooker |
| 7-8 | Move to SHOP | Adjacent to shop |
| 9 | buy(MEAT) | Holding: Meat (raw) |
| 10-11 | Move to COUNTER | Adjacent to counter |
| 12 | place() | Meat on counter |
| 13 | chop() | Meat now chopped |
| 14 | pickup() | Holding: Meat (chopped) |
| 15-16 | Move to COOKER | Adjacent to cooker |
| 17 | place() | Meat in pan, cooking starts |
| 18-19 | Move to SHOP | Adjacent to shop |
| 20 | buy(PLATE) | Holding: Plate (empty) |
| 21-22 | Move to COUNTER | Adjacent to counter |
| 23 | place() | Plate on counter |
| 24-25 | Move to SHOP | Adjacent to shop |
| 26 | buy(NOODLES) | Holding: Noodles |
| 27-28 | Move to COUNTER (plate) | Adjacent to plate |
| 29 | add_food_to_plate() | Plate now has noodles |
| 30-31 | Move to COOKER | Adjacent to cooker |
| 32-36 | (Wait for cooking) | cook_progress: 17→20 |
| 37 | take_from_pan() | Holding: Meat (cooked) |
| 38-39 | Move to COUNTER (plate) | Adjacent to plate |
| 40 | add_food_to_plate() | Plate: [noodles, meat] |
| 41 | pickup() | Holding: Plate (full) |
| 42-43 | Move to SUBMIT | Adjacent to submit |
| 44 | submit() | +$10000, order complete |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Chebyshev Distance | Max of horizontal and vertical distance; allows diagonal movement |
| cook_progress | Counter that increments each turn while food is in pan on cooker |
| cooked_stage | 0=raw, 1=cooked (good), 2=burned (bad) |
| Switch Window | Time period when teams can teleport to enemy map |
| SINKTABLE | Where clean plates appear after washing |
| Hungarian Algorithm | Optimal solution to bipartite matching (bots to tasks) |
| Time-Space A* | Pathfinding that reserves (x,y,t) tuples to prevent collisions |

---

*This document is designed to be self-contained. Any LLM reading this should have complete understanding of game mechanics to propose strategy improvements.*
