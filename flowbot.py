import Quartz, AppKit
import os, sys
import time
from mss import mss
from PIL import Image
import pyautogui as pag
from typing import TypeVar, Any, List, Dict, Tuple
from flowsolver_sat import Puzzle, PuzzleRect

WindowLocation = Tuple[int, int, int, int] # x, y, width, height
T = TypeVar('T')
Grid = List[List[T]] # 2D array
RGBt = Tuple[int, int, int] # red green blue
Coord = Tuple[int, int] # 2D coordinate

class FlowBot:
    def __init__(self, verbose=False) -> None:
        self.window_location = self.get_window("Flow", verbose=verbose)
        self.X, self.Y, self.W, self.H = self.window_location
        self.TOP_MARGIN = 145 # pixels from the top of the window title bar to the top of the game grid
        self.cell_size = self.get_puzzle_dims()

    def get_window(target_window: str, verbose=False) -> WindowLocation | None:
        '''Find the game window, bring it to focus, and return its location & dimensions.'''
        # move to focus
        apps = AppKit.NSWorkspace.sharedWorkspace().runningApplications()
        for app in apps:
            if app.localizedName() == target_window:
                app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)

        # list windows
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListExcludeDesktopElements | Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID
        )
        # get dimensions
        window_bounds = None
        for window in windows:
            window_owner = window[Quartz.kCGWindowOwnerName]
            window_name = window.get(Quartz.kCGWindowName, '<no name>')
            if window_owner == target_window:
                window_bounds = window['kCGWindowBounds']
                X,Y = int(window_bounds['X']), int(window_bounds['Y'])
                W,H = int(window_bounds['Width']), int(window_bounds['Height'])
                if verbose: print(f"Window found; {window_owner} - '{window_name}'\n{X=} {Y=} {W=} {H=}")
                return X,Y,W,H
        
        print(f"Window not found: {target_window}")

    def capture_window(window_location: WindowLocation, save_png=False) -> Image:
        '''Screenshot the game window for later use.'''
        X,Y,W,H = window_location
        with mss() as sct:
            sct_img = sct.grab({
                "top": Y + TOP_MARGIN,
                "left": X,
                "width": W,
                "height": W
            })
            img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            if save_png: img.save(os.path.join(os.getcwd(), 'puzzle.png'))
        return img

    def cell_to_screen(self, r: int, c: int) -> Coord:
        '''Convert cell coordinate to screen coordinates'''
        x = c * self.cell_size + self.cell_size//2 + self.X
        y = r * self.cell_size + self.cell_size//2 + self.Y + self.TOP_MARGIN
        return (x, y)
    
    def get_puzzle_dims(self, puzzle_img: Image) -> Tuple[int, int, int]:
        '''Get number of rows, number of columns, and cell size.'''
        # TODO

    def drag_cursor_cells(self, cells: List[Coord], duration=0, verbose=False):
        '''Drag the cursor along the given cells.'''
        r0, c0 = cells[0]
        pag.moveTo(self.cell_to_screen(r0,c0), _pause=False)
        if verbose: print(f"{cells[0]} | {self.cell_to_screen(r0,c0)}")

        for r,c in cells[1:]:
            pag.dragTo(self.cell_to_screen(r,c), duration=duration, button='left', _pause=False)
            if verbose: print(f" -> ({r}, {c}) | {self.cell_to_screen(r,c)}")

    def drag_cursor_coords(coords: List[Coord], duration=0, verbose=False):
        '''Drag the cursor along the given coordinates.'''
        pag.moveTo(coords[0], _pause=False)
        if verbose: print(f"{coords[0]}")

        for x,y in coords[1:]:
            pag.dragTo(coords[x][y], duration=duration, button='left', _pause=False)
            if verbose: print(f" -> {coords[x][y]}")


TIME = 30 # seconds
GRID_SIZE = 9 # cells per axis in the grid; assumed to be a square grid in time trials

DIRECTIONS = [
    ('L', 0, -1),
    ('R', 0, 1),
    ('U', -1, 0),
    ('D', 1, 0)
]



def identify_color(rgb: RGBt) -> str:
    '''RGBA to a letter representing the color, assumed to be one listed below.
    White is the "last" color to show up (in 9x9 time trials).'''
    R,G,B = rgb
    colors = {
        "R": (234,  51,  35), # red
        "G": ( 61, 138,  38), # green
        "B": ( 20,  40, 245), # blue
        "Y": (232, 223,  73), # yellow
        "O": (235, 142,  52), # orange
        "C": (117, 251, 253), # cyan
        "M": (234,  53, 193), # magenta
        "m": (151,  52,  48), # maroon
        "P": (116,  20, 123), # purple
        "W": (255, 255, 255)  # white
    }
    for color, (r,g,b) in colors.items():
        if abs(R-r) <= 5 and abs(G-g) <= 5 and abs(B-b) <= 5:
            return color
    return " "

def get_grid(img: Image.Image, grid_size=GRID_SIZE, verbose=False) -> Grid:
    '''Find the grid of colors.'''
    img_width, _ = img.size
    cell_size = img_width // grid_size # assume grid fills width
    grid_rgbs = [[None]*grid_size for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            cx = c * cell_size + cell_size//2
            cy = r * cell_size + cell_size//2
            pixel_color = img.getpixel((cx, cy))
            grid_rgbs[r][c] = pixel_color

    grid_colors = [[identify_color(pixel) for pixel in row] for row in grid_rgbs]

    if verbose:
        print("*"*10 + " RGB " + "*"*10)
        for row in grid_rgbs:
            print(row)
        
        print("*"*10 + " Colors " + "*"*10)
        for row in grid_colors:
            print(''.join(row))

    return grid_colors

def grid_to_txt(grid: Grid, txt_path: str) -> None:
    '''Save a grid as a .txt file.'''
    with open(txt_path, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')


def find_path(cgrid: Grid, source: Coord, color: int, verbose=False) -> List[Coord]:
    '''Find the path of grid coordinates from given color source to the sink.'''
    path = []
    r, c = source
    path_end = False
    while not path_end:
        path.append((r,c))
        cgrid[r][c] = -1

        path_end = True
        for nr, nc in valid_neighbors(r,c):
            if cgrid[nr][nc] == color:
                r,c = nr,nc
                path_end = False
                break
    
    if verbose: print(f"{color}: {path}")
    return path

def coord_to_dirs(coords: List[Coord], verbose=False) -> List[str]:
    '''Convert grid coordinates to directions (U,D,L,R) relative to the source.'''
    directions = []
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        
        for direction, dx, dy in DIRECTIONS:
            if (x1+dx, y1+dy) == (x2, y2):
                directions.append(direction)
                break
    
    if verbose: print(directions)
    return directions

def merge_dirs(start: Coord, directions: List[str], verbose=False) -> List[Coord]:
    '''Merge a direction path into a minimal number of grid coordinates in the path.'''
    if len(directions) > 1: directions.pop() # skip sink
    merged_dirs = []
    curr_dir = directions[0]
    steps = 1

    for dir in directions[1:]:
        if dir == curr_dir:
            steps += 1
        else:
            merged_dirs.append((curr_dir, steps))
            curr_dir = dir
            steps = 1

    merged_dirs.append((curr_dir, steps))

    coordinates = [start]
    x, y = start

    for dir, steps in merged_dirs:
        for dir_char, dx, dy in DIRECTIONS:
            if dir == dir_char:
                x += steps * dx
                y += steps * dy
                coordinates.append((x, y))
                break

    if verbose: print(coordinates)
    return coordinates


if __name__ == '__main__':
    LIMIT = 99 # limit on number of puzzles solved, only use for debugging purposes
    VERBOSE = False

    window_location = get_window("Flow", verbose=VERBOSE)
    if not window_location: sys.exit()
    txt_path = os.path.join(os.getcwd(), 'puzzle.txt')

    time.sleep(2) # time to move to start
    pag.click(button='left', _pause=False)
    time.sleep(0.5)

    start = time.time()
    i = 0

    while time.time() - start < TIME + 1 and i < LIMIT:
        i += 1
        puzzle_start = time.time()
        print(f"iter {i}", end='')

        img = capture_window(window_location, save=VERBOSE)

        grid_colors = get_grid(img, verbose=VERBOSE)
        grid_to_txt(grid_colors, txt_path)

        print(f" scan: {time.time() - puzzle_start :.2f}", end='')
        tcheck = time.time()

        soln = solve_single(txt_path)
        print(f" solve: {time.time() - tcheck :.2f}", end='')
        tcheck = time.time()

        sources = find_sources(soln, verbose=VERBOSE)
        cgrid = [[clr for clr,dir in row] for row in soln]

        n_colors = max(sources.keys()) + 1
        paths = [find_path(cgrid, sources[i], i, verbose=VERBOSE) for i in range(n_colors)]
        dirs = [coord_to_dirs(i, verbose=VERBOSE) for i in paths]

        for color in range(n_colors):
            clr_path = merge_dirs(sources[color], dirs[color], verbose=VERBOSE)
            drag_cursor_cells(clr_path, window_location, verbose=VERBOSE)

        print(f" action: {time.time() - tcheck :.2f}", end='')
        print(f" total: {time.time() - puzzle_start :.2f}")
        time.sleep(0.52) # time for transition between puzzles
    print(f"total time: {time.time() - start :.2f}")