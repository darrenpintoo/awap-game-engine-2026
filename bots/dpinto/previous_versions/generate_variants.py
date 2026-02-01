import os

BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"
BASE_BOT = os.path.join(BASE_DIR, "bots/dpinto/ultimate_champion_bot.py")
VARIANT_DIR = os.path.join(BASE_DIR, "bots/dpinto/champion_variants/")

if not os.path.exists(VARIANT_DIR):
    os.makedirs(VARIANT_DIR)

with open(BASE_BOT, 'r') as f:
    content = f.read()

variants = {
    # SCORING TWEEKS
    "v01_greedy_score": {
        "target": "self.score = self.profit / max(self.turns_needed, 1)",
        "replacement": "self.score = (self.profit ** 1.3) / max(self.turns_needed, 1)"
    },
    "v02_efficient_score": {
        "target": "self.score = self.profit / max(self.turns_needed, 1)",
        "replacement": "self.score = self.profit / (max(self.turns_needed, 1) ** 1.2)"
    },
    "v03_simple_order_focus": {
        "target": "self.score += (5 - len(self.required)) * 2.0",
        "replacement": "self.score += (5 - len(self.required)) * 10.0"
    },
    "v04_high_value_focus": {
        "target": "self.score = self.profit / max(self.turns_needed, 1)",
        "replacement": "self.score = self.profit # Ignore time, just money"
    },
    
    # SABOTAGE TIMING
    "v05_early_sabotage": {
        "target": "if turn >= 200:",
        "replacement": "if turn >= 50:"
    },
    "v06_late_sabotage": {
        "target": "if turn >= 200:",
        "replacement": "if turn >= 300:"
    },
    "v07_aggressive_gap": {
        "target": "if gap > 50:",
        "replacement": "if gap > 0: # Switch if losing at all"
    },
    "v08_desperate_only": {
        "target": "if gap > 50:",
        "replacement": "if gap > 200: # Only switch if getting crushed"
    },
    "v09_no_sabotage": {
        "target": "should_sabotage = False\n        # Adaptive Sabotage",
        "replacement": "return # No sabotage\n        # Adaptive Sabotage"
    },
    "v10_always_sabotage": {
        "target": "if can_switch and not self.has_switched:",
        "replacement": "if can_switch and not self.has_switched and turn > 20: # Go early and always"
    },

    # DEFENSE TWEAKS
    "v11_paranoid_defense": {
        "target": "if self._is_enemy_near(controller, team, (kx, ky), dist=2):",
        "replacement": "if self._is_enemy_near(controller, team, (kx, ky), dist=5):"
    },
    "v12_lazy_defense": {
        "target": "if self._is_enemy_near(controller, team, (kx, ky), dist=2):",
        "replacement": "if self._is_enemy_near(controller, team, (kx, ky), dist=1):"
    },
    "v13_active_hiding": {
         "target": "and (pan_state is None or pan_state == 1):",
         "replacement": ": # Hide everything regardless of state"
    },

    # COORDINATION TWEAKS
    "v14_helper_cooks": {
        "target": "if not INGREDIENT_INFO.get(ing, {}).get('cook'):",
        "replacement": "if True: # Helper can do anything"
    },
    "v15_helper_tight_trigger": {
        "target": "len(self.counters) >= 3:",
        "replacement": "len(self.counters) >= 5:"
    },
    "v16_helper_loose_trigger": {
        "target": "len(self.counters) >= 3:",
        "replacement": "len(self.counters) >= 2:"
    },

    # MISC TWEAKS
    "v17_box_hoarder": {
        "target": "if self.plate_storage_box:",
        "replacement": "if True and self.plate_storage_box: # Force box usage logic if possible"
    },
    "v18_no_recycling": {
         "target": "if sink_table and self._count_clean_plates(controller, team) > 0:",
         "replacement": "if False:"
    },
    "v19_fast_replan": {
        "target": "if current_turn % 30 == 0:",
        "replacement": "if current_turn % 10 == 0:"
    },
    
    # HYBRID (Best of Round 1 Theory)
    "v20_hybrid_stable": {
        "target": "should_sabotage = False\n        # Adaptive Sabotage",
        "replacement": """
        # Hybrid: No Sabotage + Simple Bonus + Tight Helper
        # 1. No Sabotage
        return
        
        """
    }
}

# Apply simple replacement for v20 separately as it's complex
# Actually, let's just do single replacements for simplicity in this script
# For v20, we will manually inject the logic via string replacement logic below

for name, change in variants.items():
    v_content = content.replace(change['target'], change['replacement'])
    
    # Special handling for v20 multiple changes
    if name == "v20_hybrid_stable":
        # Add simple bonus
        v_content = v_content.replace(
            "self.score += (5 - len(self.required)) * 2.0",
            "self.score += (5 - len(self.required)) * 5.0"
        )
        # Tighten helper trigger
        v_content = v_content.replace(
            "len(self.counters) >= 3:",
            "len(self.counters) >= 5:"
        )

    # Add variant name to logs
    v_content = v_content.replace('def log(msg):', f'def log(msg):\n    msg = "[{name}] " + str(msg)')
    
    with open(os.path.join(VARIANT_DIR, name + ".py"), 'w') as f:
        f.write(v_content)
    print(f"Generated {name}.py")
