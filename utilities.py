import time
from typing import TypeVar, Any, List, Tuple

T = TypeVar('T')
Grid = List[List[T]] # 2D array
Clause = List[int] # List of disjunction of literals
ColorID = int # int representation of a color
PuzzleState = Grid | Any # Any puzzle state representation

WindowLocation = Tuple[int, int, int, int] # x, y, width, height
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