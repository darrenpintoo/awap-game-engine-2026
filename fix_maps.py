import os
from pathlib import Path

MAPS_DIR = Path("maps/test-maps-revised-for-fairness")

def fix_map(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    fixed = False
    for line in lines:
        # If line starts with "# " (hash space), convert to "// "
        if line.startswith("# ") or line.startswith("#\t"):
            new_lines.append("//" + line[1:])
            fixed = True
        else:
            new_lines.append(line)
            
    if fixed:
        print(f"Fixing {path}")
        with open(path, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    for f in MAPS_DIR.glob("*.txt"):
        fix_map(f)
