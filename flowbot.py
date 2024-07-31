import Quartz, AppKit, screeninfo
import os, sys
import objc
import time
from mss import mss
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pyautogui as pag
from typing import Any, List, Dict, Tuple, Callable
from flowsolver_sat import PuzzleRect
from utilities import *

class FlowBot:
    BASE_TOP_MARGIN = 28 # height of the title bar
    def __init__(self, verbose=False) -> None:
        self.window_loc = FlowBot.get_window("Flow", verbose=verbose)
        self.X, self.Y, self.W, self.H = self.window_loc
        self.monitor_w, self.monitor_h = FlowBot.get_monitor(verbose=verbose)

    def solve_puzzle(self, verbose=False, show_imgs=False, show_ts=True) -> None:
        ts = Timestamp()
        pag.moveTo(1,1, duration=0, _pause=False)

        self.puzzle_img = FlowBot.screen_capture(self.window_loc, save_name='puzzle.png' if verbose else False)
        if show_imgs: imgshow('puzzle', self.puzzle_img)
        ts.add('screenshot')

        puzzle_img_gray = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2GRAY)
        self.set_puzzle_dims(puzzle_img_gray, verbose=verbose, show_imgs=show_imgs)
        ts.add('get puzzle dimensions')

        self.puzzle_img = self.resize_puzzle_img(self.puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.puzzle_img = self.crop_puzzle_img(self.puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.grid_colors = self.get_grid(self.puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.puzzle = PuzzleRect(self.grid_colors)
        terminals = self.puzzle.terminals
        ts.add('read and parse puzzle')

        soln_grid = self.puzzle.solve_puzzle()
        ts.add('solve puzzle')

        paths = [self.find_path(soln_grid, terminals[i], i, verbose=verbose) for i in self.puzzle.iter_colors()]
        dirs = [None] + [self.coord_to_dirs(i, verbose=verbose) for i in paths] # offset by 1 to match color indices
        ts.add('compute mouse path')

        if show_imgs: FlowBot.focus_window("Flow"); pag.click(self.window_loc[0], self.window_loc[1])
        for color in self.puzzle.iter_colors():
            clr_path = self.merge_dirs(terminals[color], dirs[color], verbose=verbose)
            self.drag_cursor_cells(clr_path, duration=0, pause=False, verbose=verbose)
        ts.add('drag mouse')

        if show_ts: print(ts.to_str())

    def find_img(self, screen: np.ndarray, img_file: str, search_region: WindowLocation,
                 screen_is_gray=True, verbose=False) -> WindowLocation | None:
        '''Find `img_file` on `screen` within the search region specified.
        Match confidence coefficient is expected to be >= 0.9 (in practice, it is almost always close to perfect, 1.000).'''

        tdelta = time.time()

        if not screen_is_gray: screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) # faster matching
        img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        search_x, search_y, search_w, search_h = search_region
        matches = cv2.matchTemplate(screen[search_y:search_y+search_h, search_x:search_x+search_w], img_gray, cv2.TM_CCOEFF_NORMED)
        _, match_val, _, match_tl = cv2.minMaxLoc(matches)
        if match_val < 0.9: print(f"image [{img_file}] was not found, best match {match_val} ({time.time()-tdelta:.4f} s)."); return None
        
        match_x, match_y = match_tl[0] + search_x, match_tl[1] + search_y
        match_h, match_w = img_gray.shape
        if verbose: print(f"image [{img_file}] was found at {match_tl} with w={match_w}, h={match_h}" +
                          f" and confidence={match_val:.3f} in ({time.time()-tdelta:.4f} s)")
        return (match_x, match_y, match_w, match_h)
    
    @staticmethod
    def _cv2_rect_from_loc(img: np.ndarray, loc: WindowLocation, color: RGBColor, stroke=1) -> None:
        '''Wrapper for drawing a rectangle on a cv2 canvas given a `WindowLocation` instead of 2 points.'''
        x,y,w,h = loc
        cv2.rectangle(img, (x,y), (x+w,y+h), color, stroke)

    @staticmethod
    def _cv2_morph(img: np.ndarray, op: Callable, kernel_size: int, kernel_shape=cv2.MORPH_RECT, n_iter=1) -> np.ndarray:
        '''Execute a morphological operation `op` on `img` with the specified `kernel_size` (default square).'''
        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size,kernel_size))
        return op(img, kernel, iterations=n_iter)

    @staticmethod
    def _find_largest_contour(contours: List[np.ndarray]) -> np.ndarray:
        '''Find the contour with maximum area.'''
        largest = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                largest = contour
                max_area = area
        return largest

    @staticmethod
    def grid_to_txt(grid: Grid, txt_path: str) -> None:
        '''Save a grid as a .txt file.'''
        with open(txt_path, 'w') as f:
            for row in grid:
                f.write(''.join(row) + '\n')

    @staticmethod
    def focus_window(target_window: str) -> None:
        '''Bring window to focus.'''
        apps = AppKit.NSWorkspace.sharedWorkspace().runningApplications()
        for app in apps:
            if app.localizedName() == target_window:
                app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)

    @staticmethod
    def get_window(target_window: str, verbose=False) -> WindowLocation | None:
        '''Find the game window, bring it to focus, and return its location & dimensions.'''
        
        FlowBot.focus_window(target_window) # move to focus

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
    def screen_capture(bbox: WindowLocation, save_name: bool | str = False) -> np.ndarray:
        '''Save a region of the screen as an image.'''
        X,Y,W,H = bbox
        with mss() as sct:
            sct_img = sct.grab({"left": X, "top": Y, "width": W, "height": H})
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            if save_name: cv2.imwrite(os.path.join(os.getcwd(), save_name), img)
        return img

    def cell_to_screen(self, r: int, c: int) -> Coord:
        '''Convert cell coordinate to screen coordinates'''
        x = int(c * self.cell_size + self.cell_size/2 + self.X + self.margin_left)
        y = int(r * self.cell_size + self.cell_size/2 + self.Y + self.margin_top)
        return (x, y)
    
    def crop_to_board_region(self, img_gray: np.ndarray, verbose=False, show_imgs=False) -> Tuple[np.ndarray, WindowLocation]:
        '''Crop a puzzle image based on where the grid is found.  Requires a grayscale image input.
        Returns the cropped board region as a binary thresholded image,
        and the location of the board relative to the puzzle image.
        '''

        if verbose: print_break("Crop to board region")

        k, c = 3, 5
        img_threshold = img_gray.copy()
        img_threshold = cv2.adaptiveThreshold(img_threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, k, c)
        if show_imgs: imgshow(f'Threshold; kernel size:{k} c:{c}', img_threshold)

        # dilate to merge duplicate lines
        img_edges = img_threshold.copy()
        img_edges = FlowBot._cv2_morph(img_edges, cv2.dilate, 5) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Dilate', img_edges)

        # find contours
        contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if show_imgs:
            img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_contours, contours, -1, (0,0,255), 3)
            imgshow(f'Contours: all', img_contours)

        # find largest contour, the board
        largest = FlowBot._find_largest_contour(contours)
        x, y, w, h = cv2.boundingRect(largest)
        if verbose: print(f'Board BBox: {x=} {y=} {w=} {h=}')
        if show_imgs:
            cv2.drawContours(img_contours, largest, -1, (0,255,255), 25)
            imgshow(f'Contours: largest', img_contours)
            FlowBot._cv2_rect_from_loc(img_contours, (x,y,w,h), (0,255,0), stroke=3)
            imgshow(f'Board BBox', img_contours)

        img_cropped = img_threshold[y : y+h+1, x : x+w+1]
        if verbose: print(f"Cropped {h}x{w}; Original {img_gray.shape}")
        if show_imgs: imgshow('Cropped', img_cropped)

        return img_cropped, (x,y,w,h)
    
    def get_grid_size(self, img_cropped: np.ndarray, verbose=False, show_imgs=False) -> Tuple[int, int, float]:
        '''Find grid width, height, cell size.  Assume thresholded binary input.'''

        print_break("Get Grid Size")

        # dilate to merge duplicate lines
        img_edges = FlowBot._cv2_morph(img_cropped, cv2.dilate, 9) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Dilate', img_edges)
        img_edges = FlowBot._cv2_morph(img_edges, cv2.erode, 11) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Dilate', img_edges)

        img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(np.int32(img_edges), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        if show_imgs:
            cv2.drawContours(img_contours, contours, -1, (0,255,0), 1)
            imgshow(f'Contours: all', img_contours)

        # filter square polygon contours
        print(f"Unfiltered contours: {len(contours)}")
        square_contours = []
        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_poly) == 4 and cv2.contourArea(contour) > 100: square_contours.append(approx_poly)
        contours = square_contours

        # group contours
        threshold_percentage = 0.15
        contours_in_group = [False] * len(contours)
        similar_contours = []
        for i, contour in enumerate(contours):
            if not contours_in_group[i]:
                area = cv2.contourArea(contour)
                similar_group = [i]
                for j, other_contour in enumerate(contours):
                    if i != j and not contours_in_group[j]:
                        other_area = cv2.contourArea(other_contour)
                        if abs(area - other_area) / area < threshold_percentage:
                            similar_group.append(j)
                            contours_in_group[j] = True
                similar_contours.append(similar_group)
                contours_in_group[i] = True

        # draw similar contours in different colors
        if show_imgs: 
            for group in similar_contours:
                color = np.random.randint(128, 255, size=3).tolist()
                img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
                for idx in group:
                    cv2.drawContours(img_contours, contours, idx, color, 5)
                imgshow(f'Contours: groups', img_contours)
        
        print(f"Filtered contours: {len(contours)}")
        for group in similar_contours:
            print(len(group), cv2.contourArea(contours[group[0]]))

        # find largest group of similar contours
        cell_contours_idx = np.argmax([len(g) for g in similar_contours])
        print(cell_contours_idx, len(similar_contours[cell_contours_idx]))
        cell_contours = [contours[i] for i in similar_contours[cell_contours_idx]]

        # find left, right, top, bottom bounds of the cell contours
        x_min = min([cv2.boundingRect(contour)[0] for contour in cell_contours])
        y_min = min([cv2.boundingRect(contour)[1] for contour in cell_contours])
        x_max = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in cell_contours])
        y_max = max([cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in cell_contours])

        # calculate the number of cells in each row and column, and the cell side length
        cell_width, cell_height = cv2.boundingRect(cell_contours[0])[2:4]

        grid_width = (x_max - x_min) // cell_width
        grid_height = (y_max - y_min) // cell_height
        
        cell_size = ((x_max-x_min)/grid_width + (y_max-y_min)/grid_height)/2 / 2

        return grid_width, grid_height, cell_size

    def crop_puzzle_img(self, img: np.ndarray, verbose=False, show_imgs=False) -> np.ndarray:
        '''Crop a puzzle image to board dimensions.'''

        img_cropped = img[self.margin_top : self.margin_bottom, self.margin_left : self.margin_right]
        if verbose: print(f"cropped {img_cropped.shape}; original {img.shape}")
        if show_imgs: imgshow('cropped', img_cropped)
        return img_cropped

    def resize_puzzle_img(self, img: np.ndarray, verbose=False, show_imgs=False) -> np.ndarray:
        '''Resize a puzzle image by a factor of 1/2, since macOS screenshots are double the resolution of the display.'''

        h,w,*_ = img.shape
        img_resized = cv2.resize(img, (w//2,h//2), interpolation=cv2.INTER_LANCZOS4)
        if verbose: print(f"resized ({w//2},{h//2}); original ({w},{h})")
        if show_imgs: imgshow('resized', img_resized)
        return img_resized

    def get_grid(self, img: np.ndarray, verbose=False, show_imgs=False) -> Grid:
        '''Find the grid of colors.'''

        grid_centers = [[None]*self.grid_width for _ in range(self.grid_height)]
        grid_rgbs = [[None]*self.grid_width for _ in range(self.grid_height)]
        grid_colors = [[None]*self.grid_width for _ in range(self.grid_height)]
        color_map = {}
        pixel_samples = img.copy()

        def color_distance(c1, c2): # avg manhattan distance between colors
            return sum(abs(a - b) for a, b in zip(c1, c2)) // 3

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                cx = int(c * self.cell_size + self.cell_size / 2)
                cy = int(r * self.cell_size + self.cell_size / 2)
                pixel_color = tuple(img[cy, cx].tolist())
                grid_rgbs[r][c] = pixel_color
                grid_centers[r][c] = (cx,cy)
                if show_imgs: cv2.circle(pixel_samples, (cx,cy), radius=3, color=(255,255,255))

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

        if show_imgs: imgshow('Pixel samples', pixel_samples)
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

    def set_puzzle_dims(self, puzzle_img: np.ndarray, verbose=False, show_imgs=False):
        '''Get left, right, top, and bottom grid margins, maximum horizontal cells (grid width),
        maximum vertical cells (grid height), and cell size.  All units in display pixels.'''

        img_cropped, (x,y,w,h) = self.crop_to_board_region(puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.margin_left, self.margin_right = x//2+2, (x+w)//2-2
        self.margin_top, self.margin_bottom = y//2+2, (y+h)//2-2
        self.grid_width, self.grid_height, self.cell_size =\
            self.get_grid_size(img_cropped, verbose=verbose, show_imgs=show_imgs)
        
        if verbose: print(f"{self.margin_left=} {self.margin_right=} {self.margin_top=} {self.margin_bottom=}\n" +
                          f"{self.grid_width=} {self.grid_height=} {self.cell_size=:.3f}")

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
    bot.solve_puzzle(verbose=False, show_imgs=False)