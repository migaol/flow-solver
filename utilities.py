import os
import time
import numpy as np
import cv2
from enum import Enum
import keyboard
from typing import TypeVar, Any, Union, List, Tuple, Callable

T = TypeVar('T')
Grid = List[List[T]] # 2D array
Clause = List[int] # List of disjunction of literals
ColorID = int # int representation of a color
PuzzleState = Grid | Any # Any puzzle state representation

PxColor = Tuple[int, int, int] # RGB or BGR
Coord = Tuple[int, int] # 2D coordinate
Line = Tuple[int, int, int, int] # x1, y1, x2, y2

class Colors(Enum):
    # BGR colors
    BLACK =      (  0,   0,   0)
    WHITE =      (255, 255, 255)
    RED =        (  0,   0, 255)
    GREEN =      (  0, 255,   0)
    BLUE =       (255,   0,   0)
    YELLOW =     (  0, 255, 255)
    CYAN =       (255, 255,   0)
    MAGENTA =    (255,   0, 255)
    GRAY =       (128, 128, 128)
    DARK_GRAY =  ( 64,  64,  64)
    LIGHT_GRAY = (192, 192, 192)
    ORANGE =     (  0, 165, 255)
    PURPLE =     (128,   0, 128)

class LRTB: # left, right, top, bottom (extrema)
    def __init__(self, l: int, r: int, t: int, b: int) -> None:
        self.l, self.r, self.t, self.b = l,r,t,b

    def __repr__(self) -> str:
        return f"left:{self.l} right:{self.r} top:{self.t} bottom:{self.b}"

    def to_xywh(self) -> 'XYWH':
        return XYWH(self.l, self.t, self.r - self.l, self.b - self.t)

class XYWH: # x, y, width, height (bounding box)
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x, self.y, self.w, self.h = x,y,w,h

    def __repr__(self) -> str:
        return f"x:{self.x} y:{self.y} width:{self.w} height:{self.h}"

    def unpack(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_lrtb(self, operation: Callable[[int], int] = lambda x: x) -> LRTB:
        return LRTB(operation(self.x), operation(self.x+self.w), operation(self.y), operation(self.y+self.h))

    def center(self) -> Coord:
        return (self.x + self.w//2, self.y + self.h//2)

    def shift(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy

    def shrink(self, px: int) -> None:
        self.x += px
        self.y += px
        self.w -= 2*px
        self.h -= 2*px

class Point:
    def __init__(self, x: int | Coord, y: int = None) -> None:
        if y is None: x,y = x # for single param inputs, assume it is a coord tuple
        self.x, self.y = x,y

    def __repr__(self) -> str:
        return f"Point({self.x},{self.y})"
    
    def __add__(self, other: Union['Point', tuple]) -> 'Point':
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple) and len(other) == 2:
            return Point(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __sub__(self, other: Union['Point', tuple]) -> 'Point':
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        elif isinstance(other, tuple) and len(other) == 2:
            return Point(self.x - other[0], self.y - other[1])
        return NotImplemented
    
    def __mul__(self, scalar: float) -> 'Point':
        if isinstance(scalar, float):
            return Point(int(self.x * scalar), int(self.y * scalar))
        return NotImplemented
    
    def __getitem__(self, idx: int) -> int:
        if idx == 0: return self.x
        elif idx == 1: return self.y
        raise IndexError()
    
    def unpack(self) -> Coord:
        return int(self.x), int(self.y)

    def in_square(self, square: LRTB) -> bool:
        return square.l <= self.x <= square.r and square.t <= self.y <= square.b
    
    def is_collinear(self, p: 'Point', q: 'Point') -> bool:
        return (min(p.x,q.x) <= self.x <= max(p.x,q.x)) and (min(p.y,q.y) <= self.y <= max(p.y,q.y))

def lines_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> int:
        '''Check whether 3 points are CW, CCW, or COL'''
        val = ((q.y-p.y) * (r.x-q.x)) - ((q.x-p.x) * (r.y-q.y))
        if val > 0: return 1 # clockwise
        elif val < 0: return -1 # counter-clockwise
        elif val == 0: return 0 # collinear

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4: return True # general case
    if o1 == 0 and p2.is_collinear(p1, q1): return True
    if o2 == 0 and q2.is_collinear(p1, q1): return True
    if o3 == 0 and p1.is_collinear(p2, q2): return True
    if o4 == 0 and q1.is_collinear(p2, q2): return True
    return False

def line_intersects_square(p: Point, q: Point, square: LRTB) -> bool:
    '''Check if line segment pq intersects a square.'''
    if p.in_square(square) or q.in_square(square): return True # endpoints in square

    square_edges = [
        (Point(square.l, square.t), Point(square.l, square.b)), # left
        (Point(square.l, square.b), Point(square.r, square.b)), # bottom
        (Point(square.r, square.b), Point(square.r, square.t)), # right
        (Point(square.r, square.t), Point(square.l, square.t)) # top
    ]

    for edge in square_edges:
        if lines_intersect(p, q, edge[0], edge[1]): return True

    return False


class Timestamp:
    '''Record labeled timestamps.'''
    def __init__(self) -> None:
        self.ts = []
        self.start = time.time()
        self.prev = self.start

    def add(self, name: str) -> None:
        '''Add a timestamp with the given name.'''
        now = time.time()
        self.ts.append((name, now - self.prev))
        self.prev = now

    def to_str(self) -> str:
        max_name_length = max(len(name) for name,_ in self.ts)
        total_time = sum(t for _,t in self.ts)

        formatted_ts = [f"{name.rjust(max_name_length)}: {t:.4f} s ({t/total_time:.2%})" for name,t in self.ts]
        formatted_ts.append(f"{'total'.rjust(max_name_length)}: {total_time:.4f} s")
        return '\n'.join(formatted_ts)
    
    def get_elapsed(self) -> float:
        return time.time() - self.start

def exit_on_keypress(keyname: str) -> Callable:
    def callback(event):
        if event.name == keyname:
            print("exited by key press")
            os._exit(1)
    return callback

def pct_diff(a: float, b: float) -> float:
    '''Percent distance.'''
    return abs(a-b)/b

def color_distance(c1: PxColor, c2: PxColor) -> int:
    '''Averaged manhattan distance between colors.'''
    return sum(abs(a-b) for a,b in zip(c1,c2)) // 3

def print_break(s: str) -> None:
    print(f"{'*'*5} {s} {'*'*5}")

def imgshow(winname: str, mat: np.ndarray) -> None:
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()