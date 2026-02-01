# AWAP 2026: Carnegie Cookoff - Complete Game Mechanics Report

## Table of Contents
1. [Game Overview](#1-game-overview)
2. [Core Mechanics](#2-core-mechanics)
3. [Map Structure & Tiles](#3-map-structure--tiles)
4. [Items & Economy](#4-items--economy)
5. [Bot API Reference](#5-bot-api-reference)
6. [Order System](#6-order-system)
7. [Mid-Game Switch Mechanic](#7-mid-game-switch-mechanic)

---

## 1. Game Overview

**Carnegie Cookoff** is a turn-based competitive cooking game where two teams (RED and BLUE) race to fulfill food orders and maximize their money score.

| Parameter | Value |
|-----------|-------|
| Total Turns | 500 |
| Teams | 2 (RED vs BLUE) |
| Bots per Team | 2 (configurable) |
| Starting Money | Configurable via `GameConstants` or `GameState` init (default varies) |
| Passive Income | +$1 per team per turn |

**Win Condition**: Team with the highest `team_money` at turn 500 wins.

---

## 2. Core Mechanics

### 2.1 Turn Structure

Each turn proceeds as follows:
1. **Environment Tick**: Cooking progress advances, dishes wash automatically if sink is active
2. **Order Expiry Check**: Expired orders deduct penalty from team money
3. **Switch Window Check**: If switch window ended, teleport bots back to home map
4. **Bot Actions**: Each bot can perform **1 move** AND **1 action** (in that order, or just action)

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

Maps are ASCII text files defining the grid layout and order waves. Note that starting money is NOT configured in the map file.

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

### 3.3 Tile State Attributes

Access via `controller.get_tile(team, x, y)`:

| Tile | Attribute | Description |
|------|-----------|-------------|
| COOKER | `item` | Pan object (has `.food` attribute) |
| COOKER | `cook_progress` | Int, ticks food has been cooking |
| SINK | `num_dirty_plates` | Int, dirty plates in sink |
| SINK | `curr_dirty_plate_progress` | Int, washing progress (0-1) |
| SINKTABLE | `num_clean_plates` | Int, clean plates available |
| COUNTER | `item` | Item on counter (Food/Plate/Pan or None) |
| BOX | `item` | Item type stored |
| BOX | `count` | Number of items in box |

### 3.4 Important Spatial Rules

- **Interaction Range**: Must be within Chebyshev distance 1 to interact
- **Bots cannot stand on non-walkable tiles**
- **Each map is independent**: RED and BLUE have separate, identical maps (until Switch)

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

**Important Attribute**: Access chopping status via `item.chopped` (boolean), NOT `is_chopped`.

### 4.2 Buying Rules
- **No Debt**: You cannot buy an item if your team's money is less than the item's cost.
- **Negative Balance**: If your balance is negative, you cannot buy anything (assuming costs are positive).

### 4.3 Equipment

| Item | Cost | Enum | Description |
|------|------|------|-------------|
| PLATE | $2 | `ShopCosts.PLATE` | Required for plating/submitting |
| PAN | $4 | `ShopCosts.PAN` | Required for cooking |

### 4.4 Cooking System

1. **Place Pan on Cooker** (via `place()`)
2. **Add Food to Pan** (via `place()` while holding cookable food)
3. **Automatic Cooking**: Each turn, `cook_progress` increments by 1
4. **Cooked Stage Transitions**:
   - `cook_progress = 0-19`: Stage 0 (raw)
   - `cook_progress = 20`: Stage 1 (cooked) ✅
   - `cook_progress = 40+`: Stage 2 (burned) ❌

> **CRITICAL**: Cooking is automatic and CANNOT be paused. You must retrieve food at exactly the right time.

### 4.5 Plate System

1. **Clean Plates**: Come from `SINKTABLE` (via `take_clean_plate()`). Can also buy new plates.
2. **Adding Food**: Use `add_food_to_plate()` - bot can hold plate and target food, or hold food and target plate.
3. **Submitting**: Plate must be clean and contain exact ingredients for an order.
4. **After Submission**: Plate becomes dirty.
5. **Washing**: Bot uses `wash_sink()` on SINK tile. Each turn washing is active, `PLATE_WASH_PROGRESS` increments. When full, a dirty plate becomes clean at SINKTABLE.

---

## 5. Bot API Reference

**NOTE**: Many `controller` methods now require the `team` argument (e.g., `controller.get_team()`).

### 5.1 State Query Methods

```python
controller.get_turn() -> int                    # Current turn number
controller.get_team() -> Team                   # RED or BLUE
controller.get_enemy_team() -> Team             # The opposing team

# UPDATED: These require the team argument
controller.get_map(team) -> Map                 # Deep copy of map for specified team
controller.get_orders(team) -> List[Order]      # Active orders for specified team
controller.get_team_bot_ids(team) -> List[int]  # Bot IDs [0,1] for yours, [2,3] for enemy

controller.get_team_money(team) -> int          # Team money
controller.get_tile(team, x, y) -> Tile         # Tile at position

# Bot state returns dict:
controller.get_bot_state(bot_id) -> {
    'x': int,
    'y': int, 
    'holding': dict | None,  # See below
    'team': Team
}
# holding dict structure (if not None):
# {'type': 'Food', 'food_name': str, 'chopped': bool, 'cooked_stage': int}
# {'type': 'Plate', 'dirty': bool, 'food': [...]}
# {'type': 'Pan', 'food': dict | None}
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
# FAILS if team_money < cost
```

### 5.5 Food Processing

```python
controller.chop(bot_id, target_x, target_y) -> bool         # Chop food on COUNTER
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
- **NOODLES**: Must be raw (not chopped, not cooked)
- **MEAT**: Must be chopped (`item.chopped=True`) AND cooked (`cooked_stage=1`)

---

## 7. Mid-Game Switch Mechanic

### 7.1 Overview

A **sabotage mechanic** allowing teams to temporarily invade the enemy's kitchen.

### 7.2 Switch Window

- **Active Window**: Defined in map file (e.g., Turn 250 to 350)
- **Action**: Either team can call `switch_maps()` **once** during the window.
- **Effect**: All bots on that team teleport to the enemy's map.

### 7.3 API

```python
controller.get_switch_info() -> {
    'switch_turn': int,          # Turn when window opens
    'switch_duration': int,      # How long window lasts
    'has_switched': bool,        # Your team already switched?
    'enemy_has_switched': bool   # Enemy already switched?
}
controller.can_switch_maps() -> bool
controller.switch_maps() -> bool           # Does NOT consume action
```

### 7.4 After Switch Window Ends

- All switched bots **automatically teleport back** to their home map.
- Bots **keep whatever they are holding** when teleported.
