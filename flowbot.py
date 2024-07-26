import Quartz, AppKit, screeninfo
import os, sys
import time
from mss import mss
from PIL import Image
import cv2
import math
import numpy as np
import pyautogui as pag
from typing import Any, List, Dict, Tuple
from flowsolver_sat import PuzzleRect
from utilities import *

class FlowBot:
    BASE_TOP_MARGIN = 28 # height of the title bar
    def __init__(self, verbose=False) -> None:
        self.window_location = FlowBot.get_window("Flow", verbose=verbose)
        self.X, self.Y, self.W, self.H = self.window_location
        self.monitor_w, self.monitor_h = FlowBot.get_monitor(verbose=verbose)

    def solve_puzzle(self, verbose=False, show_ts=True) -> None:
        ts = Timestamp()

        self.puzzle_img = FlowBot.screen_capture(self.window_location, save_png='puzzle.png' if verbose else False)
        ts.add('screenshot')

        self.left_margin, self.top_margin, self.grid_width, self.grid_height, self.cell_size =\
            self.get_puzzle_dims(self.puzzle_img, verbose=verbose)
        ts.add('get puzzle dimensions')
        
        self.puzzle_img = self.trim_puzzle_img(self.puzzle_img)
        if verbose: self.puzzle_img.save(os.path.join(os.getcwd(), 'trimmed_puzzle.png'))

        self.grid_colors = self.get_grid(self.puzzle_img, verbose=verbose)
        self.puzzle = PuzzleRect(self.grid_colors)
        terminals = self.puzzle.terminals
        ts.add('read and parse puzzle')

        soln_grid = self.puzzle.solve_puzzle()
        ts.add('solve puzzle')

        paths = [self.find_path(soln_grid, terminals[i], i, verbose=verbose) for i in self.puzzle.iter_colors()]
        dirs = [None] + [self.coord_to_dirs(i, verbose=verbose) for i in paths] # offset by 1 to match color indices
        ts.add('compute mouse path')

        for color in self.puzzle.iter_colors():
            clr_path = self.merge_dirs(terminals[color], dirs[color], verbose=verbose)
            self.drag_cursor_cells(clr_path, duration=0, pause=False, verbose=verbose)
        ts.add('drag mouse')

        if show_ts: print(ts.to_str())

    def find_img(self, screen: np.ndarray, img_file: str, screen_is_gray=True, verbose=False) -> WindowLocation | None:
        '''Find `img` on `screen`.  Match confidence coefficient is expected to be > 0.9
        (in practice, it is almost always close to perfect, 1.000).'''

        if not screen_is_gray: screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) # faster matching
        img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        matches = cv2.matchTemplate(screen, img_gray, cv2.TM_CCOEFF_NORMED)
        _, match_val, _, match_tl = cv2.minMaxLoc(matches)
        if match_val < 0.9: print(f"image [{img_file}] was not found."); return None
        
        match_x, match_y = match_tl
        match_h, match_w = img_gray.shape
        if verbose: print(f"image [{img_file}] was found at {match_tl} with w={match_w}, h={match_h} and confidence={match_val:.3f}")
        return (match_x, match_y, match_w, match_h)
    
    @staticmethod
    def _cv2_rect_from_loc(img: np.ndarray, loc: WindowLocation, color: RGBColor, stroke=1) -> None:
        '''Wrapper for drawing a rectangle on a cv2 canvas given a `WindowLocation` instead of 2 points.'''
        x,y,w,h = loc
        cv2.rectangle(img, (x,y), (x+w,y+h), color, stroke)

    def find_lines(self, img: Image.Image | np.ndarray,  verbose=False, show_img_process=False) -> Tuple[List[Line], List[Line]]:
        '''Find puzzle border and cell lines.'''

        if isinstance(img, Image.Image): img = np.array(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(img_gray, threshold1=50, threshold2=100)
        lines = cv2.HoughLinesP(img_edges, 1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=5)

        # find puzzle grid boundaries based on screen elements
        flows_counter_loc = self.find_img(img_gray, f'./assets/flows_counter_{self.monitor_w}x{self.monitor_h}.png', verbose=verbose)
        flows_counter_bottom = flows_counter_loc[1] + flows_counter_loc[3]

        hint_lines_loc = self.find_img(img_gray, f'./assets/hint_lines_{self.monitor_w}x{self.monitor_h}.png', verbose=verbose)
        hint_lines_top = hint_lines_loc[1]

        if show_img_process: cv2.imshow('edges', img_edges); cv2.waitKey(0)
    
        hlines = []
        vlines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if flows_counter_bottom < y1 and y2 < hint_lines_top: # filter within boundaries
                if abs(line_angle) < 1:
                    hlines.append(line[0])
                elif abs(line_angle) > 89:
                    vlines.append(line[0])

        def filter_duplicate_lines(lines: List[Line], direction: str, threshold=20):
            compare_idx, shift_idx = (0,1) if direction == 'v' else (1,0)

            # for vertical lines, sort left -> right; for horizontal lines, sort by top -> bottom
            lines.sort(key=lambda line: (line[compare_idx]))

            filtered_lines = []
            i = 0
            while i < len(lines):
                current_line = list(lines[i])
                j = i + 1
                # group lines within threshold, merge into a larger line
                while j < len(lines) and abs(lines[j][compare_idx] - current_line[compare_idx]) <= threshold:
                    current_line[shift_idx] = min(current_line[shift_idx], lines[j][shift_idx])
                    current_line[shift_idx + 2] = max(current_line[shift_idx + 2], lines[j][shift_idx + 2])
                    j += 1
                filtered_lines.append(current_line)
                i = j
            return filtered_lines

        hlines = filter_duplicate_lines(hlines, 'h')
        vlines = filter_duplicate_lines(vlines, 'v')

        if show_img_process:
            img_lines = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            for line in hlines: # red horizontal
                x1, y1, x2, y2 = line
                cv2.line(img_lines, (x1,y1), (x2,y2), (0,0,255), 2)
            for line in vlines: # green vertical
                x1, y1, x2, y2 = line
                cv2.line(img_lines, (x1,y1), (x2,y2), (0,255,0), 2)
            # blue recognized images
            FlowBot._cv2_rect_from_loc(img_lines, flows_counter_loc, (255,0, 0))
            FlowBot._cv2_rect_from_loc(img_lines, hint_lines_loc, (255,0, 0))
        
            cv2.imshow('lines', img_lines); cv2.waitKey(0)

        return hlines, vlines

    @staticmethod
    def grid_to_txt(grid: Grid, txt_path: str) -> None:
        '''Save a grid as a .txt file.'''
        with open(txt_path, 'w') as f:
            for row in grid:
                f.write(''.join(row) + '\n')

    @staticmethod
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
                Y += FlowBot.BASE_TOP_MARGIN
                H -= FlowBot.BASE_TOP_MARGIN
                if verbose: print(f"Window found; {window_owner} - '{window_name}': {X=} {Y=} {W=} {H=}")
                return X,Y,W,H
        
        print(f"Window not found: {target_window}")

    @staticmethod
    def get_monitor(verbose=False) -> Tuple[int, int]:
        '''Get the primary monitor resolution.'''
        monitor = screeninfo.get_monitors()[0]
        if verbose: print(f"monitor {monitor}")
        return monitor.width, monitor.height

    @staticmethod
    def screen_capture(bbox: WindowLocation, save_png: bool | str = False) -> Image.Image:
        '''Save a region of the screen as an image.'''
        X,Y,W,H = bbox
        with mss() as sct:
            sct_img = sct.grab({"left": X, "top": Y, "width": W, "height": H})
            img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
            if save_png: img.save(os.path.join(os.getcwd(), save_png))
        return img

    def cell_to_screen(self, r: int, c: int) -> Coord:
        '''Convert cell coordinate to screen coordinates'''
        x = c * self.cell_size + self.cell_size//2 + self.X
        y = r * self.cell_size + self.cell_size//2 + self.Y + self.top_margin
        return (x, y)
    
    def trim_puzzle_img(self, img: Image.Image) -> Image.Image:
        '''Trim a puzzle image to specified puzzle dimensions.
        Also resize to 1/2 because mac screenshots are higher resolution than the monitor.'''
        img_resized = img.resize((img.width//2, img.height//2))
        return img_resized.crop((
            self.left_margin,
            self.top_margin,
            self.left_margin + self.grid_width * self.cell_size,
            self.top_margin + self.grid_height * self.cell_size
        ))

    def get_grid(self, img: Image.Image, verbose=False) -> Grid:
        '''Find the grid of colors.'''

        grid_centers = [[None]*self.grid_height for _ in range(self.grid_width)]
        grid_rgbs = [[None]*self.grid_height for _ in range(self.grid_width)]
        grid_colors = [[None]*self.grid_height for _ in range(self.grid_width)]
        color_map = {}

        def color_distance(c1, c2):
            return sum(abs(a - b) for a, b in zip(c1, c2)) // 3

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                cx = c * self.cell_size + self.cell_size // 2
                cy = r * self.cell_size + self.cell_size // 2
                pixel_color = img.getpixel((cx, cy))
                grid_rgbs[r][c] = pixel_color
                grid_centers[r][c] = (cx,cy)

                # check for existing color similarity
                found = False
                if color_distance(pixel_color, (0,0,0)) <= 32: # black, empty
                    grid_colors[r][c] = 0
                    found = True
                for existing_color, color_id in color_map.items():
                    if color_distance(pixel_color, existing_color) <= 5: # other terminal was found
                        grid_colors[r][c] = color_id
                        found = True
                        break

                if not found:
                    new_color_id = len(color_map)+1
                    color_map[pixel_color] = grid_colors[r][c] = new_color_id

        if verbose:
            print("*"*10 + " Cell Centers " + "*"*10)
            for row in grid_centers:
                print(row)
            
            print("*"*10 + " RGB " + "*"*10)
            for row in grid_rgbs:
                print(row)
            
            print("*"*10 + " Colors " + "*"*10)
            for row in grid_colors:
                print(' '.join(('.' if c == 0 else str(c)) for c in row))

        return grid_colors

    def get_puzzle_dims(self, puzzle_img: Image.Image, verbose=False) -> Tuple[int, int, int, int, int]:
        '''Get left grid margin, top grid margin, maximum horizontal cells (grid width),
        maximum vertical cells (grid height), and cell size.'''
        
        # divide by 2 because mac screenshots are higher resolution than the screen
        hlines, vlines = self.find_lines(puzzle_img, verbose=verbose, show_img_process=verbose)
        margin_left, margin_top, margin_right = vlines[0][0]//2, hlines[0][1]//2, vlines[-1][0]//2
        grid_width, grid_height = len(vlines)-1, len(hlines)-1
        cell_size = (margin_right - margin_left) // grid_width
        
        if verbose: print(f"{margin_left=} {margin_top=} {grid_width=} {grid_height=} {cell_size=}")
        return margin_left, margin_top, grid_width, grid_height, cell_size 

    def find_path(self, color_grid: Grid, source: Coord, color: int, verbose=False) -> List[Coord]:
        '''Find the path of grid coordinates from given color source to the sink.'''
        path = []
        r, c = source
        path_end = False
        while not path_end:
            path.append((r,c))
            color_grid[r][c] = -1

            path_end = True
            for nr, nc in self.puzzle.neighbors(r,c):
                if color_grid[nr][nc] == color:
                    r,c = nr,nc
                    path_end = False
                    break
        
        if verbose: print(f"{color}: {path}")
        return path

    def coord_to_dirs(self, grid_coords: List[Coord], verbose=False) -> List[str]:
        '''Convert grid coordinates to directions (U,D,L,R) relative to the source.'''
        directions = []
        for i in range(len(grid_coords) - 1):
            x1, y1 = grid_coords[i]
            x2, y2 = grid_coords[i+1]
            
            for direction, dx, dy in self.puzzle.DIRECTIONS:
                if (x1+dx, y1+dy) == (x2, y2):
                    directions.append(direction)
                    break
        
        if verbose: print(directions)
        return directions
    
    def merge_dirs(self, start: Coord, directions: List[str], verbose=False) -> List[Coord]:
        '''Merge a direction path into a minimal number of grid coordinates in the path.'''
        if len(directions) > 1: directions.pop() # skip sink, it is automatically connected to the penultimate pipe
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
            for dir_char, dx, dy in self.puzzle.DIRECTIONS:
                if dir == dir_char:
                    x += steps * dx
                    y += steps * dy
                    coordinates.append((x, y))
                    break

        if verbose: print(coordinates)
        return coordinates


    def drag_cursor_cells(self, cells: List[Coord], duration=0, pause=False, verbose=False):
        '''Drag the cursor along the given cells.'''
        r0, c0 = cells[0]
        pag.moveTo(self.cell_to_screen(r0,c0), _pause=pause)
        if verbose: print(f"{cells[0]} | {self.cell_to_screen(r0,c0)}")

        for r,c in cells[1:]:
            pag.dragTo(self.cell_to_screen(r,c), duration=duration, button='left', _pause=pause)
            if verbose: print(f" -> ({r}, {c}) | {self.cell_to_screen(r,c)}")

    def drag_cursor_coords(screen_coords: List[Coord], duration=0, pause=False, verbose=False):
        '''Drag the cursor along the given coordinates.'''
        pag.moveTo(screen_coords[0], _pause=pause)
        if verbose: print(f"{screen_coords[0]}")

        for x,y in screen_coords[1:]:
            pag.dragTo(screen_coords[x][y], duration=duration, button='left', _pause=pause)
            if verbose: print(f" -> {screen_coords[x][y]}")



if __name__ == '__main__':
    bot = FlowBot(verbose=False)
    bot.solve_puzzle(verbose=False)