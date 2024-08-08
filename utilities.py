'''General utility classes and functions.'''

import time
import numpy as np
import cv2
from enum import Enum
from typing import TypeVar, Any, Union, List, Tuple, Callable

T = TypeVar('T')
Grid = List[List[T]] # 2D array
Clause = List[int] # List of disjunction of literals
ColorID = int # int representation of a color
PuzzleState = Union[Grid, Any] # Any puzzle state representation

Numeric = Union[int, float] # any numeric value
PxColor = Tuple[int, int, int] # RGB or BGR
Coord = Tuple[Numeric, Numeric] # 2D coordinate
Line = Tuple[Numeric, Numeric, Numeric, Numeric] # x1, y1, x2, y2

EPSILON = 1e-5

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

class LRTB:
    '''Store left, right, top, and bottom.  Functions as a set of extrema.'''
    def __init__(self, l: int, r: int, t: int, b: int) -> None:
        self.l, self.r, self.t, self.b = l,r,t,b

    def __repr__(self) -> str:
        return f"left:{self.l} right:{self.r} top:{self.t} bottom:{self.b}"

    def to_xywh(self) -> 'XYWH':
        return XYWH(self.l, self.t, self.r - self.l, self.b - self.t)

class XYWH:
    '''Store top left corner (x,y), width, and height.  Functions as a bounding box.'''
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

class Vect2:
    '''2D vector.'''
    def __init__(self, x: Numeric, y: Numeric) -> None:
        self.x, self.y = x,y

    @classmethod
    def coord(cls, tuple: Coord) -> 'Vect2':
        return cls(tuple[0], tuple[1])

    def __repr__(self) -> str:
        return f"({self.x:.4f},{self.y:.4f})"
    
    def __add__(self, other: Union['Vect2', Coord]) -> 'Vect2':
        if isinstance(other, Vect2):
            return Vect2(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple) and len(other) == 2:
            return Vect2(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __sub__(self, other: Union['Vect2', Coord]) -> 'Vect2':
        if isinstance(other, Vect2):
            return Vect2(self.x - other.x, self.y - other.y)
        elif isinstance(other, tuple) and len(other) == 2:
            return Vect2(self.x - other[0], self.y - other[1])
        return NotImplemented
    
    def __mul__(self, scalar: Numeric) -> 'Vect2':
        if isinstance(scalar, Numeric): return Vect2(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __getitem__(self, idx: int) -> int:
        if idx == 0: return int(self.x)
        elif idx == 1: return int(self.y)
        raise IndexError()
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vect2): return NotImplemented
        return abs(self.x - other.x) < EPSILON and abs(self.y - other.y) < EPSILON
    
    def inverted(self) -> 'Vect2':
        return Vect2(self.y, self.x)
    
    def unpack(self) -> Coord:
        return int(self.x), int(self.y)

    def in_square(self, square: LRTB) -> bool:
        return square.l <= self.x <= square.r and square.t <= self.y <= square.b
    
    def is_endpoint(self, p: 'Vect2', q: 'Vect2') -> bool:
        return self == p or self == q
    
    def is_collinear(self, p: 'Vect2', q: 'Vect2') -> bool:
        return (min(p.x,q.x) <= self.x <= max(p.x,q.x)) and (min(p.y,q.y) <= self.y <= max(p.y,q.y))
    
    def is_strictly_collinear(self, p: 'Vect2', q: 'Vect2') -> bool:
        return self.is_collinear(p,q) and not self.is_endpoint(p,q)
    
    def parallel_axis(self, other: 'Vect2') -> bool:
        return self.x == other.x or self.y == other.y
    
    def _to_manim(self) -> np.ndarray:
        return np.array([self.x, self.y, 0])

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

    def get_t(self, name: str = None) -> float:
        '''Get timestamp entry.  If no name is specified, then get the sum of all existing entries.'''
        if not name: return sum(t for _,t in self.ts)
        for n,t in self.ts:
            if n == name: return t

    def to_str(self) -> str:
        max_name_length = max(len(name) for name,_ in self.ts)
        total_time = sum(t for _,t in self.ts)

        formatted_ts = [f"{name.rjust(max_name_length)}: {t:.4f} s ({t/total_time:.2%})" for name,t in self.ts]
        formatted_ts.append(f"{'total'.rjust(max_name_length)}: {total_time:.4f} s")
        return '\n'.join(formatted_ts)
    
    def get_elapsed(self) -> float:
        return time.time() - self.start

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
    # cv2.imwrite(os.path.join('puzzle_processing', f'{winname}.png'), mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()