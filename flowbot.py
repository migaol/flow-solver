import Quartz, AppKit, screeninfo
import os, sys
from mss import mss
import cv2
import numpy as np
import pyautogui as pag
from enum import Enum
from typing import List, Tuple, Callable
from flowsolver_sat import PuzzleRect
from utilities import *
from pathfinder import Pathfinder

class TTDuration(Enum):
    '''Valid time trial durations.'''
    _30SEC = 30
    _60SEC = 60
    _1MIN = 60
    _2MIN = 2*60
    _4MIN = 4*60

TEPSILON = 1e-4
class WaitTime(Enum):
    '''Wait duration constants.'''
    PUZZLE_LOAD_SERIES_INIT = 0.5 + TEPSILON
    PUZZLE_LOAD_SERIES = 0.52
    PUZZLE_LOAD_TT = 0.50 + TEPSILON
    LARGE_PUZZLE_LOAD_TT = 0.505 + TEPSILON
    NEXT_LEVEL_BUTTON = 0.1 + TEPSILON

#TODO add puzzle logging
# switch mouse path method for denser grids
class FlowBot:
    BASE_TOP_MARGIN = 28 # height of the title bar
    def __init__(self, verbose=False) -> None:
        self.window_bbox = FlowBot.get_window("Flow", verbose=verbose)
        if not self.window_bbox: sys.exit()
        self.monitor_w, self.monitor_h = FlowBot.get_monitor(verbose=verbose)
        keyboard.hook(exit_on_keypress('esc'))

    def wait_key_click(self, puzzle_load_time: WaitTime, move_to_window=True) -> None:
        '''Wait for a key press, then click (and pause for the puzzle to load).
        Should be used to position the mouse before starting.'''
        if move_to_window: pag.moveTo(*self.window_bbox.center(), duration=0, _pause=False)
        keyboard.read_event()
        pag.click(duration=0, _pause=False)
        pag.sleep(puzzle_load_time.value)

    def click_next_level(window_bbox: XYWH) -> None:
        '''Click on the 'next level' button.  Alignment is always in the center in the grid window,
        assuming that ads are disabled (by turning off internet).'''
        pag.sleep(WaitTime.NEXT_LEVEL_BUTTON.value)
        pag.click(window_bbox.center(), duration=0, _pause=False)
        pag.sleep(WaitTime.PUZZLE_LOAD_SERIES.value)

    def solve_series(self, verbose=False, show_imgs=False, show_ts=True) -> None:
        '''Solve several puzzles in a row, proceeding to the next puzzle upon completing one.'''

        self.wait_key_click(WaitTime.PUZZLE_LOAD_SERIES_INIT)

        ts_agg = []
        it = 0
        while True:
            it += 1
            if verbose or show_ts: print_break(f"Iter {it}")
            puzzle_ts = self.solve_puzzle(verbose=verbose, show_imgs=show_imgs, show_ts=show_ts)
            ts_agg.append(puzzle_ts)

            FlowBot.click_next_level(self.window_bbox)

    def solve_time_trial(self, duration=TTDuration._30SEC, verbose=False, show_ts=True) -> None:
        '''Solve subsequent puzzles in a time trial.
        In the interest of time efficiency, verbose is False for most methods, and showing images is disabled.'''
        
        try: TTDuration(duration)
        except ValueError: raise ValueError(f"Invalid duration:[{duration.value}]")
        duration = duration.value
        if verbose: print(f"Started time trial for duration {duration} sec.")

        self.wait_key_click(WaitTime.PUZZLE_LOAD_TT)
        pag.moveTo(1,1, duration=0, _pause=False) # move mouse away from screenshot area
        ts_agg = Timestamp()

        self.puzzle_img = FlowBot.screen_capture(self.window_bbox, save_name='puzzle.png' if verbose else False)
        ts_agg.add('Initial screenshot')

        puzzle_img_gray = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2GRAY)
        self.set_puzzle_dims(puzzle_img_gray, verbose=verbose)
        puzzle_bbox = self.margins.to_xywh()
        puzzle_bbox.shift(self.window_bbox.x, self.window_bbox.y)
        ts_agg.add('Get puzzle dimensions')
        if show_ts: print(ts_agg.to_str())

        it = 0
        while ts_agg.get_elapsed() <= duration:
            ts_puzzle = Timestamp()
            pag.moveTo(1,1, duration=0, _pause=False) # move mouse away from screenshot area
            it += 1
            if verbose or show_ts: print_break(f"Iter {it}")

            if it > 1:
                self.puzzle_img = FlowBot.screen_capture(puzzle_bbox, save_name='puzzle.png' if verbose else False)
                ts_puzzle.add('Screenshot')

            self.puzzle_img = FlowBot.resize_half(self.puzzle_img)
            if it == 1: self.puzzle_img = FlowBot.crop_puzzle_img(self.puzzle_img, self.margins)
            self.grid_colors = FlowBot.get_grid(self.puzzle_img, (self.grid_width, self.grid_height, self.cell_size))
            self.puzzle = PuzzleRect(self.grid_colors)
            terminals = self.puzzle.terminals
            ts_puzzle.add('Read and parse puzzle')

            soln_grid = self.puzzle.solve_puzzle()
            ts_puzzle.add('Solve puzzle')

            paths = [self.find_path(soln_grid, terminals[i], i, verbose=verbose) for i in self.puzzle.iter_colors()]
            path_coords = [None] + [self.merge_path_coords(p, verbose=verbose) for p in paths]
            ts_puzzle.add('Compute mouse path')

            for color in self.puzzle.iter_colors():
                clr_path = path_coords[color]
                self.drag_cursor_coords(clr_path, duration=0, pause=False, verbose=verbose)
            ts_puzzle.add('Drag mouse')

            if show_ts:
                print(ts_puzzle.to_str())
                print(f"Total elapsed: {ts_agg.get_elapsed():.4f}")

            puzzle_wait = WaitTime.PUZZLE_LOAD_TT if self.puzzle.n_cells < 40 else WaitTime.LARGE_PUZZLE_LOAD_TT
            pag.sleep(puzzle_wait.value)


    def solve_puzzle(self, verbose=False, show_imgs=False, show_ts=True) -> Timestamp:
        '''Solve a single puzzle.'''

        ts = Timestamp()
        pag.moveTo(1,1, duration=0, _pause=False) # move mouse away from screenshot area

        self.puzzle_img = FlowBot.screen_capture(self.window_bbox, save_name='puzzle.png' if verbose else False)
        if show_imgs: imgshow('Puzzle', self.puzzle_img)
        ts.add('Screenshot')

        puzzle_img_gray = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2GRAY)
        if show_imgs: imgshow('Gray', puzzle_img_gray)
        self.set_puzzle_dims(puzzle_img_gray, verbose=verbose, show_imgs=show_imgs)
        ts.add('Get puzzle dimensions')

        self.puzzle_img = FlowBot.resize_half(self.puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.puzzle_img = FlowBot.crop_puzzle_img(self.puzzle_img, self.margins, verbose=verbose, show_imgs=show_imgs)
        self.grid_colors = FlowBot.get_grid(self.puzzle_img, (self.grid_width, self.grid_height, self.cell_size),
                                            verbose=verbose, show_imgs=show_imgs)
        self.puzzle = PuzzleRect(self.grid_colors)
        terminals = self.puzzle.terminals
        ts.add('Read and parse puzzle')

        soln_grid = self.puzzle.solve_puzzle(verbose=verbose)
        ts.add('Solve puzzle')

        paths = [self.find_path(soln_grid, terminals[i], i, verbose=verbose) for i in self.puzzle.iter_colors()]
        # dirs = [None] + [self.coord_to_dirs(i, verbose=verbose) for i in paths] # offset by 1 to match color indices
        path_coords = [None] + [self.merge_path_coords(p, verbose=verbose) for p in paths]
        if show_imgs: self.draw_paths_coords(self.puzzle_img, path_coords)
        ts.add('Compute mouse path')

        if show_imgs: FlowBot.focus_window("Flow"); pag.click(self.window_bbox.x, self.window_bbox.y) # refocus on game
        for color in self.puzzle.iter_colors():
            clr_path = path_coords[color]
            self.drag_cursor_coords(clr_path, duration=0, pause=False, verbose=verbose)
        ts.add('Drag mouse')

        if show_ts: print(ts.to_str())
        return ts
    
    @staticmethod
    def _cv2_rect_from_loc(img: np.ndarray, loc: XYWH, color: PxColor, stroke=1) -> None:
        '''Wrapper for drawing a rectangle on a cv2 canvas given a `XYWH` bbox instead of 2 points.'''
        cv2.rectangle(img, (loc.x,loc.y), (loc.x+loc.w,loc.y+loc.h), color, stroke)

    @staticmethod
    def _cv2_morph(img: np.ndarray, op: Callable, kernel_size: int, kernel_shape=cv2.MORPH_RECT, n_iter=1) -> np.ndarray:
        '''Execute a morphological operation `op` on `img` with the specified `kernel_size` (square by default).'''
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
    def get_window(target_window: str, verbose=False) -> XYWH | None:
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
                bbox = XYWH(X,Y,W,H)
                if verbose: print(f"Window found; {window_owner} - '{window_name}': {bbox}")
                return bbox
        
        print(f"Window not found: {target_window}")

    @staticmethod
    def get_monitor(verbose=False) -> Tuple[int, int]:
        '''Get the primary monitor resolution.'''
        monitor = screeninfo.get_monitors()[0]
        if verbose: print(f"monitor {monitor}")
        return monitor.width, monitor.height

    @staticmethod
    def screen_capture(bbox: XYWH, save_name: bool | str = False) -> np.ndarray:
        '''Save a region of the screen as an image.'''
        with mss() as sct:
            sct_img = sct.grab({"left": bbox.x, "top": bbox.y, "width": bbox.w, "height": bbox.h})
            img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            if save_name: cv2.imwrite(os.path.join(os.getcwd(), save_name), img)
        return img


    def set_puzzle_dims(self, puzzle_img: np.ndarray, verbose=False, show_imgs=False):
        '''Get left, right, top, and bottom grid margins, maximum horizontal cells (grid width),
        maximum vertical cells (grid height), and cell size.  All units in display pixels.'''

        img_cropped, bbox = FlowBot.crop_to_board_region(puzzle_img, verbose=verbose, show_imgs=show_imgs)
        self.margins = bbox.to_lrtb(operation = lambda x : x//2)
        self.grid_width, self.grid_height, self.cell_size =\
            FlowBot.get_grid_size(img_cropped, verbose=verbose, show_imgs=show_imgs)
        self.grid_tl = (self.window_bbox.x + self.margins.l, self.window_bbox.y + self.margins.t)
        
        if verbose:
            print_break("Puzzle Dimensions")
            print(f"{self.margins}\n{self.grid_width=} {self.grid_height=} {self.cell_size=:.3f}")

    @staticmethod
    def crop_to_board_region(img_gray: np.ndarray, verbose=False, show_imgs=False) -> Tuple[np.ndarray, XYWH]:
        '''Crop a puzzle image based on where the grid is found.  Requires a grayscale image input.
        Returns the cropped board region as a binary thresholded image,
        and the location of the board relative to the puzzle image. '''

        if verbose: print_break("Crop to board region")

        thresh_k,thresh_c = 3,5
        img_threshold = img_gray.copy()
        img_threshold = cv2.adaptiveThreshold(img_threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh_k, thresh_c)
        if show_imgs: imgshow(f'Threshold; kernel size:{thresh_k} c:{thresh_c}', img_threshold)

        # dilate to merge duplicate lines
        img_edges = img_threshold.copy()
        img_edges = FlowBot._cv2_morph(img_edges, cv2.dilate, 5) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Dilate', img_edges)

        # find contours
        contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if show_imgs:
            img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_contours, contours, -1, Colors.RED.value, 3)
            imgshow(f'Contours: all', img_contours)

        # find largest contour, the board
        BBOX_SHRINK_PX = 4
        largest = FlowBot._find_largest_contour(contours)
        board_bbox = XYWH(*cv2.boundingRect(largest))
        board_bbox.shrink(BBOX_SHRINK_PX)
        if verbose: print(f'Board BBox: {board_bbox}')
        if show_imgs:
            cv2.drawContours(img_contours, largest, -1, Colors.YELLOW.value, 25)
            imgshow(f'Contours: largest (corners)', img_contours)
            FlowBot._cv2_rect_from_loc(img_contours, board_bbox, Colors.GREEN.value, stroke=3)
            imgshow(f'Board BBox', img_contours)

        img_cropped = img_threshold[board_bbox.y : board_bbox.y+board_bbox.h+1, board_bbox.x : board_bbox.x+board_bbox.w+1]
        if verbose: print(f"Cropped {board_bbox.h}x{board_bbox.w}; Original {img_gray.shape}")
        if show_imgs: imgshow('Cropped', img_cropped)

        return img_cropped, board_bbox
    
    @staticmethod
    def get_grid_size(img_cropped: np.ndarray, verbose=False, show_imgs=False) -> Tuple[int, int, float]:
        '''Find grid width, height, cell size.  Assume thresholded binary image input.'''

        if verbose: print_break("Get Grid Size")

        # dilate to merge duplicate lines
        img_edges = FlowBot._cv2_morph(img_cropped, cv2.dilate, 9) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Dilate', img_edges)
        img_edges = FlowBot._cv2_morph(img_edges, cv2.erode, 11) # on different resolutions, kernel size may need to change
        if show_imgs: imgshow('Erode', img_edges)

        # find contours
        contours, _ = cv2.findContours(np.int32(img_edges), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        if show_imgs:
            img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_contours, contours, -1, Colors.GREEN.value, 1)
            imgshow(f'Board contours: all', img_contours)
        if verbose: print(f"Unfiltered contours: {len(contours)}")

        # filter square polygon contours
        MIN_CONTOUR_AREA = 100 # minimum contour area to be considered; filters noise contours
        APPROX_EPS_COEFF = 0.05 # epsilon coefficient for cv2.approxPolyDP(); converts noisy cells into squares
        square_contours = []
        for contour in contours:
            epsilon = APPROX_EPS_COEFF * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_poly) == 4 and cv2.contourArea(contour) > MIN_CONTOUR_AREA: square_contours.append(approx_poly)
        contours = square_contours

        # group similar-sized contours
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        SIMILARITY_THRESH_PCT = 0.15 # percent difference to be considered a similar-sized contour
        contours_in_group = [False] * len(contours)
        similar_contours = []
        for i, contour in enumerate(contours):
            if contours_in_group[i]: continue
            similar_group = [i]
            for j, _ in enumerate(contours):
                if i == j or contours_in_group[j]: continue
                if pct_diff(contour_areas[i], contour_areas[j]) < SIMILARITY_THRESH_PCT:
                    similar_group.append(j)
                    contours_in_group[j] = True
            similar_contours.append(similar_group)
            contours_in_group[i] = True

        # show similar contours
        if verbose:
            print(f"Filtered contours: {len(contours)}")
            for i,group in enumerate(similar_contours):
                print(f"Group {i} - size:{len(group)} area:{cv2.contourArea(contours[group[0]])}")
        if show_imgs: 
            for i,group in enumerate(similar_contours):
                color = np.random.randint(128, 255, size=3).tolist()
                img_contours = cv2.cvtColor(img_edges.copy(), cv2.COLOR_GRAY2BGR)
                for c in group:
                    cv2.drawContours(img_contours, contours, c, color, 5)
                imgshow(f'Contours: groups ({i})', img_contours)

        # find largest group of similar contours
        cell_contours_idx = np.argmax([len(g) for g in similar_contours])
        cell_contours = [contours[i] for i in similar_contours[cell_contours_idx]]
        if verbose: print(f"Largest contour group idx:{cell_contours_idx}; size:{len(similar_contours[cell_contours_idx])}")

        # find left, right, top, bottom bounds of the cell contours
        cell_bboxes = [cv2.boundingRect(contour) for contour in cell_contours]
        x_min = min([bbox[0] for bbox in cell_bboxes])
        y_min = min([bbox[1] for bbox in cell_bboxes])
        x_max = max([bbox[0] + bbox[2] for bbox in cell_bboxes])
        y_max = max([bbox[1] + bbox[3] for bbox in cell_bboxes])

        # calculate the number of cells in each row and column
        cell_width, cell_height = cell_bboxes[0][2:4] # arbitrarily use bottom right box
        grid_width = (x_max - x_min) // cell_width
        grid_height = (y_max - y_min) // cell_height
        # average unrounded grid width & height, then divide by 2 because mac screenshots are higher resolution
        cell_size = ((x_max-x_min)/grid_width + (y_max-y_min)/grid_height)/2 / 2

        return grid_width, grid_height, cell_size

    @staticmethod
    def crop_puzzle_img(img: np.ndarray, lrtb: LRTB, verbose=False, show_imgs=False) -> np.ndarray:
        '''Crop a puzzle image to board dimensions.'''

        img_cropped = img[lrtb.t : lrtb.b, lrtb.l : lrtb.r]
        if verbose: print(f"Cropped {img_cropped.shape}; Original {img.shape}")
        if show_imgs: imgshow('Cropped', img_cropped)
        return img_cropped

    @staticmethod
    def resize_half(img: np.ndarray, verbose=False, show_imgs=False) -> np.ndarray:
        '''Resize a puzzle image by a factor of 1/2, since macOS screenshots are double the resolution of the display.'''

        h,w,*_ = img.shape
        img_resized = cv2.resize(img, (w//2,h//2), interpolation=cv2.INTER_LANCZOS4)
        if verbose: print(f"Resized ({w//2},{h//2}); Original ({w},{h})")
        if show_imgs: imgshow('Resized (1/2)', img_resized)
        return img_resized

    @staticmethod
    def get_grid(img: np.ndarray, grid_dims: Tuple[int, int, float], verbose=False, show_imgs=False) -> Grid:
        '''Find the grid of colors.'''

        grid_width, grid_height, cell_size = grid_dims

        if verbose: grid_centers = [[None]*grid_width for _ in range(grid_height)]
        grid_rgbs = [[None]*grid_width for _ in range(grid_height)]
        grid_colors = [[None]*grid_width for _ in range(grid_height)]
        color_map = {}
        pixel_samples = img.copy()

        for r in range(grid_height):
            for c in range(grid_width):
                cx = int(c * cell_size + cell_size / 2)
                cy = int(r * cell_size + cell_size / 2)
                pixel_color = tuple(img[cy, cx].tolist())
                grid_rgbs[r][c] = pixel_color
                if verbose: grid_centers[r][c] = (cx,cy)
                if show_imgs: cv2.circle(pixel_samples, (cx,cy), 5, Colors.WHITE.value, 3)

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
    
    
    def merge_path_coords(self, path: List[Coord], verbose=False) -> List[Coord]:
        '''Merge a direction path into a minimal number of screen coordinates in the path.'''
        pf = Pathfinder(path, self.cell_size, pct_size=0.6)
        path_coords = pf.find_path()
        path_coords = [p.inverted() for p in path_coords]
        if verbose: print(path_coords)
        return path_coords


    def draw_paths_cells(self, img: np.ndarray, paths: List[List[Coord]], img_copy=True) -> None:
        if img_copy: img = img.copy()
        for path in paths:
            if not path: continue
            r0,c0 = path[0]
            x,y = self.cell_to_screen(r0,c0,rel_grid=True)
            color = tuple(img[y,x].tolist())
            for r1,c1 in path[1:]:
                cv2.line(img, self.cell_to_screen(r0,c0,rel_grid=True), self.cell_to_screen(r1,c1,rel_grid=True), color, 5)
                r0,c0 = r1,c1
        imgshow('Path', img)

    def draw_paths_coords(self, img: np.ndarray, paths: List[List[Point]], rel_board=True, img_copy=True) -> None:
        if img_copy: img = img.copy()
        for path in paths:
            if not path: continue
            if not rel_board: path = [p - self.grid_tl for p in path]
            x0,y0 = path[0].unpack()
            for p1 in path[1:]:
                x1,y1 = p1.unpack()
                cv2.line(img, (x0,y0), (x1,y1), (255,255,255), 3)
                cv2.circle(img, (x1,y1), 5, (128,128,128), 5)
                x0,y0 = x1,y1
        imgshow('Path', img)

    def cell_to_square(self, r: int, c: int, rel_grid=False, diam_pct=0.85) -> LRTB:
        '''Convert cell coordinate to screen square, with side length `diam_pct` of `cell_size`.
        A smaller side length avoids pixel errors as part of rounding.'''
        cx,cy = self.cell_to_screen(r,c, rel_grid=rel_grid)
        half_side = int((self.cell_size * diam_pct)//2)
        return LRTB(cx-half_side, cx+half_side, cy-half_side, cy+half_side)

    def cell_to_screen(self, r: int | Coord, c: int = None, rel_grid=False) -> Coord:
        '''Convert cell coordinate to screen coordinates.
        If `rel_grid` is True, screen coordinates are relative to the top left of the grid,
        otherwise they are relative to the top left of the screen.'''
        if c is None: r,c = r # for single param inputs, assume it is a coord tuple
        x = int(c * self.cell_size + self.cell_size/2)
        y = int(r * self.cell_size + self.cell_size/2)
        if not rel_grid:
            x += self.window_bbox.x + self.margins.l
            y += self.window_bbox.y + self.margins.t
        return x,y
    
    def drag_cursor_cells(self, cells: List[Coord], duration=0, pause=False, verbose=False) -> None:
        '''Drag the cursor along the given cells.'''

        r0,c0 = cells[0]
        pag.moveTo(self.cell_to_screen(r0,c0), duration=duration, _pause=pause)
        if verbose: print(f"{cells[0]} | {self.cell_to_screen(r0,c0)}")

        for r,c in cells[1:]:
            pag.dragTo(self.cell_to_screen(r,c), button='left', duration=duration, _pause=pause)
            if verbose: print(f" -> ({r}, {c}) | {self.cell_to_screen(r,c)}")

    def drag_cursor_coords(self, coords: List[Point], rel_screen=False, duration=0, pause=False, verbose=False) -> None:
        '''Drag the cursor along the given coordinates.'''

        if not rel_screen: coords = [c + self.grid_tl for c in coords]

        pag.moveTo(coords[0].unpack(), duration=duration, _pause=pause)
        if verbose: print(f"{coords[0]}")

        for p in coords[1:]:
            pag.dragTo(p.unpack(), button='left', duration=duration, _pause=pause)
            if verbose: print(f" -> {p}")



if __name__ == '__main__':
    bot = FlowBot(verbose=False)
    bot.solve_puzzle(verbose=False, show_imgs=False)
    # bot.solve_series(verbose=False, show_imgs=False, show_ts=True)
    # bot.solve_time_trial(duration=TTDuration._30SEC, verbose=False, show_ts=True)