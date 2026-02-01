
import os

def write_map(filename, layout_lines, orders):
    # Ensure consistent width
    width = len(layout_lines[0])
    for i, line in enumerate(layout_lines):
        if len(line) != width:
            print(f"Error in {filename} row {i}: expected {width}, got {len(line)}")
            # Pad with dots or trim
            if len(line) < width:
                layout_lines[i] = line + "." * (width - len(line))
            else:
                layout_lines[i] = line[:width]
            
            # Ensure borders are correct if it's a border row
            if i == 0 or i == len(layout_lines) - 1:
                layout_lines[i] = "#" * width
            else:
                # Ensure start/end are # unless it's a side entrance (none here)
                pass

    with open(filename, "w") as f:
        for line in layout_lines:
            f.write(line + "\n")
        f.write("\nORDERS:\n")
        f.write(orders)
    print(f"Wrote {filename} ({len(layout_lines)}x{width})")

def create_warehouse():
    # 40x40 Open Warehouse
    rows = []
    # Row 0: Top Border
    rows.append("#" * 40)
    
    # Row 1: Spawns and counters
    # #b......C........C...................b#
    # We build rows programmatically to be safe
    line = "#b" + "." * 6 + "C" + "." * 6 + "C" + "." * 16 + "b#" 
    # adjust length to 40
    # Current: 1+1+6+1+6+1+16+1+1 = 34. Too short.
    # Need 40. 
    # 1(#) + 1(b) + 36(middle) + 1(b) + 1(#) = 40
    padding = "." * 36
    # Let's just make a grid
    grid = [["." for _ in range(40)] for _ in range(40)]
    
    # Borders
    for x in range(40):
        grid[0][x] = "#"
        grid[39][x] = "#"
    for y in range(40):
        grid[y][0] = "#"
        grid[y][39] = "#"
        
    # Spawns (Top Left b, Top Right b, Bottom Left b, Bottom Right b)
    # Actually usually it's symmetric. Map convention: (0,0) is bottom left.
    # Text file: Row 0 is top.
    # Let's put b at (1,1) in grid coords (top-left in file) -> Real map (1, 38)
    # The parsing:
    # row 0 is y=height-1. row height-1 is y=0.
    
    # Top-left and bottom-right? Or Left vs Right side?
    # Standard maps usually have Left vs Right or diagonally opposed.
    # Let's do b at (1,1) and b at (38,1) -> Top Left and Top Right?
    # Let's follow the previous attempt: #b... ...b#
    
    grid[1][1] = "b"
    grid[1][38] = "b"
    grid[38][1] = "b" 
    grid[38][38] = "b"
    
    # Columns of Counters
    for y in range(2, 6):
        grid[y][8] = "C"
        grid[y][16] = "C"
        grid[y][32] = "C"
    
    # Bottom Counters (reflected)
    for y in range(34, 38):
        grid[y][8] = "C"
        grid[y][16] = "C"
        grid[y][32] = "C"

    # Shops
    for y in range(8, 11):
        grid[y][4] = "$"
        grid[y][36] = "$"

    # Cookers
    for y in range(12, 16):
        grid[y][8] = "K"
        grid[y][32] = "K"

    # Sinks
    for y in range(12, 16):
        grid[y][10] = "S"
        grid[y][30] = "S"
        
    # Central Box structure
    for x in range(15, 25):
        grid[18][x] = "#"
        grid[22][x] = "#"
    for y in range(18, 23):
        grid[y][15] = "#"
        grid[y][25] = "#"
        
    # Submit inside box
    for y in range(19, 22):
        grid[y][20] = "U"
        
    # Serialize
    lines = ["".join(row) for row in grid]
    
    orders = """start=0    duration=100 required=NOODLES           reward=100  penalty=10
start=0    duration=100 required=MEAT              reward=150  penalty=20
start=10   duration=150 required=NOODLES,MEAT      reward=300  penalty=50
start=20   duration=150 required=NOODLES,ONIONS    reward=250  penalty=40
start=30   duration=200 required=MEAT,EGG,ONIONS   reward=500  penalty=100
start=50   duration=100 required=NOODLES           reward=100  penalty=10
start=60   duration=100 required=MEAT              reward=150  penalty=20
start=70   duration=150 required=NOODLES,MEAT      reward=300  penalty=50
start=80   duration=150 required=NOODLES,ONIONS    reward=250  penalty=40
start=100  duration=200 required=MEAT,EGG,ONIONS   reward=500  penalty=100"""

    write_map("maps/dpinto/mega_warehouse.txt", lines, orders)

def create_maze():
    # 40x40 Maze
    grid = [["." for _ in range(40)] for _ in range(40)]
    
    # Borders
    for x in range(40):
        grid[0][x] = "#"
        grid[39][x] = "#"
    for y in range(40):
        grid[y][0] = "#"
        grid[y][39] = "#"
        
    # Spawns
    grid[1][1] = "b"
    grid[38][38] = "b"
    
    # Walls - Concentric-ish squares
    # Outer ring
    for x in range(4, 36):
        grid[4][x] = "#"
        grid[35][x] = "#"
    for y in range(4, 36):
        grid[y][4] = "#"
        grid[y][35] = "#"
        
    # Inner block
    for x in range(10, 30):
        for y in range(10, 30):
            if (x+y) % 2 == 0:
                grid[y][x] = "#"

    # Facilities in corners
    # TL
    grid[2][2] = "C"
    grid[2][3] = "C"
    
    # TR
    grid[2][37] = "K"
    grid[3][37] = "S"
    
    # BL
    grid[37][2] = "$"
    grid[36][2] = "U"
    
    # BR
    grid[37][37] = "T"
    
    # Serialize
    lines = ["".join(row) for row in grid]
    
    orders = """start=0    duration=150 required=NOODLES           reward=100  penalty=10
start=0    duration=150 required=MEAT              reward=150  penalty=20
start=20   duration=200 required=NOODLES,MEAT      reward=300  penalty=50
start=30   duration=200 required=EGG,ONIONS        reward=200  penalty=40
start=50   duration=250 required=MEAT,EGG,ONIONS   reward=500  penalty=100
start=100  duration=150 required=NOODLES           reward=100  penalty=10
start=110  duration=150 required=MEAT              reward=150  penalty=20
start=120  duration=200 required=NOODLES,MEAT      reward=300  penalty=50
start=130  duration=200 required=EGG,ONIONS        reward=200  penalty=40
start=150  duration=250 required=MEAT,EGG,ONIONS   reward=500  penalty=100"""

    write_map("maps/dpinto/mega_maze.txt", lines, orders)

if __name__ == "__main__":
    create_warehouse()
    create_maze()
