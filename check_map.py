from game_constants import TileType
from robot_controller import RobotController
import pickle

def check_connectivity(map_obj):
    w, h = map_obj.width, map_obj.height
    walkable = []
    for x in range(w):
        for y in range(h):
            if map_obj.tiles[x][y].is_walkable:
                walkable.append((x,y))
    
    components = []
    visited = set()
    for start in walkable:
        if start in visited: continue
        comp = []
        q = [start]
        visited.add(start)
        while q:
            curr = q.pop(0)
            comp.append(curr)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = curr[0]+dx, curr[1]+dy
                if 0 <= nx < w and 0 <= ny < h and map_obj.tiles[nx][ny].is_walkable and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        components.append(comp)
    
    print(f"Total components: {len(components)}")
    for i, c in enumerate(components):
        print(f"Component {i}: size {len(c)}")
        # Check adjacent facilities
        facs = []
        for x, y in c:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                fx, fy = x+dx, y+dy
                if 0 <= fx < w and 0 <= fy < h:
                    name = map_obj.tiles[fx][fy].tile_name
                    if name != 'EMPTY' and name != 'COUNTER' and (fx, fy) not in c:
                        facs.append(name)
        print(f"  Reach: {set(facs)}")

# We can't easily run this without a map object.
# I'll just add it to JuniorChampion.py __init__ for one run.
