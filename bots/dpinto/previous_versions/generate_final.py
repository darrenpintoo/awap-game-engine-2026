import os

BASE_DIR = "/Users/darrenpinto/Documents/Hackathons/AWAP2026/awap-game-engine-2026-public"
BASE_BOT = os.path.join(BASE_DIR, "bots/dpinto/ultimate_champion_bot.py")
VARIANT_DIR = os.path.join(BASE_DIR, "bots/dpinto/champion_variants/")

with open(BASE_BOT, 'r') as f:
    content = f.read()

# v_final: efficiency focus
# 1. Disable Sabotage (v4/v9/v09 success)
# 2. Disable Recycling (v18/v7 success)
# 3. Enable Box Hoarding (v8/v17 success)

content = content.replace(
    "should_sabotage = False\n        # Adaptive Sabotage",
    "return # v_final: No sabotage\n        # Adaptive Sabotage"
)

content = content.replace(
    "if sink_table and self._count_clean_plates(controller, team) > 0:",
    "if False: # v_final: No recycling"
)

content = content.replace(
    "if self.plate_storage_box:",
    "if True and self.plate_storage_box:"
)

content = content.replace('def log(msg):', 'def log(msg):\n    msg = "[v_final] " + str(msg)')

with open(os.path.join(VARIANT_DIR, "v_final_efficiency.py"), 'w') as f:
    f.write(content)
print("Generated v_final_efficiency.py")
