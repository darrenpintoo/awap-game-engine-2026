"""
Do Nothing Bot - Baseline Control Strategy
============================================

STRATEGY:
- Does absolutely NOTHING
- Never buys ingredients, plates, or pans
- Bots just stand still (or optionally wiggle randomly)
- Tests the "what if we just don't play" baseline

PURPOSE:
This bot serves as a critical baseline test:
1. Any competent bot MUST beat this consistently
2. Exposes maps/orders where penalties don't punish inaction
3. Tests if sabotage-focused bots waste resources attacking nothing
4. Reveals if any bot has negative expected value (loses to doing nothing)

KEY DECISION LOGIC:
1. Do nothing
2. See step 1

WEAKNESSES THIS EXPLOITS:
- Sabotage bots waste time switching to an empty map
- Over-engineered bots might have bugs that lose to inaction
- Maps with low penalties might not punish this enough

PERFORMANCE PROFILE:
- EXCELS ON: Maps with very low penalties, broken scoring systems
- STRUGGLES ON: Everything (by design - this SHOULD lose)

If your champion bot ever loses to this, something is very wrong.
"""

try:
    from robot_controller import RobotController
except ImportError:
    pass


class BotPlayer:
    """The ultimate lazy bot - does absolutely nothing."""
    
    def __init__(self, map_copy):
        self.map = map_copy
        # That's it. We don't need anything else.
    
    def play_turn(self, controller: RobotController):
        """
        The most elegant strategy: do nothing.
        
        No pathfinding needed.
        No order selection needed.
        No state machine needed.
        No bugs possible (almost).
        
        Just... exist.
        """
        pass  # The entire strategy in one keyword
