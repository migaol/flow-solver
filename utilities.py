import time
import numpy as np
import cv2
from typing import TypeVar, Any, List, Tuple, Callable

T = TypeVar('T')
Grid = List[List[T]] # 2D array
Clause = List[int] # List of disjunction of literals
ColorID = int # int representation of a color
PuzzleState = Grid | Any # Any puzzle state representation

class LRTB: # left, right, top, bottom (extrema)
    def __init__(self, l: int, r: int, t: int, b: int) -> None:
        self.l, self.r, self.t, self.b = l,r,t,b
    def __repr__(self) -> str:
        return f"left:{self.l} right:{self.r} top:{self.t} bottom:{self.b}"
class XYWH: # x, y, width, height (bounding box)
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x, self.y, self.w, self.h = x,y,w,h
    def __repr__(self) -> str:
        return f"x:{self.x} y:{self.y} width:{self.w} height:{self.h}"
    def unpack(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h
    def to_lrtb(self, operation: Callable[[int], int] = lambda x: x) -> LRTB:
        return LRTB(operation(self.x), operation(self.x+self.w), operation(self.y), operation(self.y+self.h))
RGBColor = Tuple[int, int, int] # red, green, blue
Coord = Tuple[int, int] # 2D coordinate
Line = Tuple[int, int, int, int] # x1, y1, x2, y2

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

def pct_diff(a: float, b: float) -> float:
    '''Percent distance.'''
    return abs(a-b)/b

def color_distance(c1: RGBColor, c2: RGBColor) -> int:
    '''Averaged manhattan distance between colors.'''
    return sum(abs(a-b) for a,b in zip(c1,c2)) // 3

def print_break(s: str) -> None:
    print(f"{'*'*5} {s} {'*'*5}")

def imgshow(winname: str, mat: np.ndarray) -> None:
    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()