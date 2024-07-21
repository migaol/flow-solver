'''
Solve flow puzzles by reducing to the boolean satisfiability problem.
This uses the Glucose 4.2.1 SAT solver in the pysat library.
More info on the Glucose solver: https://github.com/audemard/glucose
'''

import sys
from pysat.solvers import Glucose42
from typing import TypeVar, Any, List, Dict, Tuple, Generator
from abc import ABC, abstractmethod
import itertools

T = TypeVar('T')
Grid = List[List[T]]
Clause = List[int]

class Puzzle(ABC):
    '''Puzzle information and SAT reducer.'''
    state: Any
    n_colors: int
    def __init__(self, type: str) -> None:
        self.type = type

    @staticmethod
    def no_two(vars: List[int]) -> List[Clause]:
        '''Generate CNF clauses which dictate that no two (at most one) of the given variables are true.'''
        return ([-a, -b] for a, b in itertools.combinations(vars, 2))
    
    @staticmethod
    def exactly_one(vars: List[int]) -> List[Clause]:
        '''Generate CNF clauses which dictates that exactly one of the given variables are true.'''
        clauses = [vars] # at least 1 is true
        clauses.extend(Puzzle.no_two(vars)) # at most 1 is true / no 2 are true
        return clauses
    
    @staticmethod
    def exactly_two(vars: List[int]) -> List[Clause]:
        '''Generate CNF clauses which dictate that exactly two of the given variables are true.'''
        assert len(vars) > 1

        # at least 1 is true / not exactly 0 are true
        clauses = [vars]

        # not exactly 1 is true
        for i in range(len(vars)):
            # create clauses where each variable is the only negated literal once
            clause = [-vars[i]] + [vars[j] for j in range(len(vars)) if j != i]
            clauses.append(clause)
        if len(vars) == 2: return clauses
        
        # at most 2 are true / no 3 are true
        for combo in itertools.combinations(vars, 3):
            clauses.append([-x for x in combo])

        return clauses

    @abstractmethod
    def to_str(self, puzzle_state: Any) -> str:
        '''Represent the puzzle state as a string.'''

    @abstractmethod
    def var_vertex(self, *args, **kwargs) -> int:
        '''Create a uniquely identifiable SAT variable, as an int,
        which represents a cell being a certain color.'''
    
    @abstractmethod
    def var_edge(self, *args, **kwargs) -> int:
        '''Create a uniquely identifiable SAT variable, as an int,
        which represents an edge being a certain color (or none).'''

    @abstractmethod
    def create_vertex_clauses(self, *args, **kwargs) -> List[Clause]:
        '''Create clauses for vertices/cells.  These clauses generally represent:
        - terminal cells are their original color and at most this one color
        - terminal cells have exactly one neighbor with the same color
        - empty/pipe cells are exactly one color
        '''

    @abstractmethod
    def create_edge_clauses(self, *args, **kwargs) -> List[Clause]:
        '''Create clauses for edges, which represent relations between neighbor vertices/cells.
        These clauses generally represent:
        - empty/pipe cells which are connected have the same color
        - empty/pipe cells neighbor exactly two other cells of the same color'''

    @staticmethod
    def solve_sat(clauses: List[Clause]) -> List[int]:
        '''Solve a CNF formula by feeding into a SAT solver.'''
        solver = Glucose42()
        for clause in clauses:
            solver.add_clause(clause)

        soln = solver.get_model() if solver.solve() else None
        solver.delete()

        return soln

    @abstractmethod
    def parse_solution(self) -> Any:
        '''Convert SAT solution variables into a readable form.'''

    @abstractmethod
    def solve_puzzle(self) -> None:
        '''Solve the puzzle and decode the solution.'''

    @abstractmethod
    def _verify_clauses(self, print_clauses=False) -> None:
        '''Print the clauses and verify there are the correct number.'''

class PuzzleRect(Puzzle):
    U = 0b0001
    D = 0b0010
    L = 0b0100
    R = 0b1000
    directions = [
        (U, -1, 0),
        (D, 1, 0),
        (L, 0, -1),
        (R, 0, 1)
    ]
    def __init__(self, puzzle_grid: Grid) -> None:
        super().__init__('rect')
        self.state: Grid = puzzle_grid
        self.n_colors = max(max(row) for row in puzzle_grid)
        self.rows = len(puzzle_grid)
        self.cols = len(puzzle_grid[0])
        self.n_cells = self.rows * self.cols
        self.edge_var_offset = self.n_cells * (self.n_colors+1) + 1 # self.n_colors+1 since colors are 1-indexed

    def __repr__(self) -> str:
        return self.to_str(self.state)
    
    def to_str(self, puzzle_state: Grid) -> str:
        max_length = max(len(str(cell)) for row in puzzle_state for cell in row)
        return '\n'.join(' '.join(f"{cell:0{max_length}d}" if cell > 0 else "-"*max_length
                                  for cell in row) for row in puzzle_state)

    def var_vertex(self, r: int, c: int, clr: int) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents
        "the cell/vertex at row `r` and column `c` is `clr`".'''
        return (clr * self.cols + c) * self.rows + r

    def var_edge(self, r1: int, c1: int, r2: int, c2: int, clr: int) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents
        "the edge between the cell/vertex at row `r1` and column `c1` and the cell/vertex at (`r2`, `c2`) is `clr`".'''

        # enforce deterministic edge variable regardless of order of the 2 vertices.
        # (r2,c2) will always be down/right of (r1,c1)
        if r1 * self.cols + c1 > r2 * self.cols + c2: r1, c1, r2, c2 = r2, c2, r1, c1

        edge_index = 2 * (r1 * self.cols + c1) # identify with the up/left endpoint cell/vertex
        if c2-c1 == 1: edge_index += 1 # up-down edge

        return self.edge_var_offset + edge_index * (self.n_colors+1) + clr
    
    def _parse_var_vertex(self, var: int, as_str=False) -> Tuple[int, int, int, bool] | str:
        '''Find the row, column, color, and parity corresponding to a numeric vertex variable.'''

        if var <= 0 or var >= self.edge_var_offset: raise ValueError("Variable does not correspond to a vertex.")

        r = var % self.rows
        temp = var // self.rows
        c = temp % self.cols
        clr = temp // self.cols

        if not as_str: return r, c, clr, var > 0
        return f"({r},{c}):{clr}" if var > 0 else f"NOT({r},{c}):{clr}"
    
    def _parse_var_edge(self, var: int, as_str=False) -> Tuple[int, int, int, int, int, bool] | str:
        '''Find the row and column of endpoint vertices, color, and parity corresponding to a numeric edge variable.'''
        
        if var < self.edge_var_offset: raise ValueError("Variable does not correspond to an edge.")
        
        var -= self.edge_var_offset
        edge_index = var // (self.n_colors+1)
        clr = var % (self.n_colors+1)
        endpoint = edge_index // 2
        is_horizontal = edge_index % 2 == 1
        
        r1 = endpoint // self.cols
        c1 = endpoint % self.cols
        
        if is_horizontal: r2, c2 = r1, c1 + 1
        else:             r2, c2 = r1 + 1, c1
        
        if not as_str: return r1, c1, r2, c2, clr, var + self.edge_var_offset > 0
        
        return f"({r1},{c1})-({r2},{c2}):{clr}" if var + self.edge_var_offset > 0 else f"NOT({r1},{c1})-({r2},{c2}):{clr}"

    def iter_puzzle(self) -> Generator[Tuple[int, int, int], None, None]:
        '''Iterate through row index, column index, and cell.'''
        for r, row in enumerate(self.state):
            for c, cell in enumerate(row):
                yield r, c, cell

    def iter_colors(self) -> Generator[int, None, None]:
        '''Iterate over all unique numeric color identifiers in the puzzle.'''
        return range(1, self.n_colors+1)

    def valid_cell(self, r: int, c: int) -> bool:
        '''Check whether the cell at (row, col) exists in the puzzle.'''
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int, int]]:
        '''Return a list of valid neighbors of the given cell coordinates.
        Each neighbor is represented as: (direction bit, row, column)'''
        return [(dir, r+dr, c+dc) for (dir, dr, dc) in self.directions if self.valid_cell(r+dr, c+dc)]

    def create_vertex_clauses(self, print_clauses=False) -> List[Clause]:
        clauses = []
        for r, c, cell in self.iter_puzzle():
            new_clauses = []
            if cell > 0: # terminal cell
                # cell is this color
                new_clauses.append([self.var_vertex(r, c, cell)])
                # cell is at most one color
                new_clauses.extend([-self.var_vertex(r, c, clr)] for clr in self.iter_colors() if clr != cell)

                # exactly one neighbor cell is this color
                neighbors_this_color = [self.var_vertex(nr, nc, cell) for dir, nr, nc in self.neighbors(r, c)]
                new_clauses.extend(Puzzle.exactly_one(neighbors_this_color))
            
            else: # empty cell, future pipe cell
                # cell is exactly one color
                cell_is_color = [self.var_vertex(r, c, clr) for clr in self.iter_colors()]
                new_clauses.extend(Puzzle.exactly_one(cell_is_color))

            clauses.extend(new_clauses)
            if print_clauses:
                print(f"cell ({r},{c}) clauses:")
                for clause in new_clauses:
                    print(clause, '; '.join(self._parse_var_vertex(var, as_str=True) for var in clause))
        
        return clauses
    
    def create_edge_clauses(self, print_clauses=False) -> List[Clause]:
        pass
    
    def parse_solution(self, soln: List[int]) -> Grid | None:
        if not soln:
            print("No solution")
            return None
        
        soln.insert(0, 0) # insert at index 0 so that SAT variables can index directly into their parity in the solution
        
        puzzle_soln = [[0]*self.cols for _ in range(self.rows)]
        for r, c, _ in self.iter_puzzle():
            for clr in self.iter_colors():
                if soln[self.var_vertex(r, c, clr)] > 0:
                    puzzle_soln[r][c] = clr
        
        return puzzle_soln

    def solve_puzzle(self, print_soln=False) -> None:
        clauses = self.create_vertex_clauses(print_clauses=False)

        soln = Puzzle.solve_sat(clauses)
        puzzle_soln = self.parse_solution(soln)
        if print_soln: print(self.to_str(puzzle_soln))

    def _verify_clauses(self, print_clauses=False) -> None:
        vset = set()
        for r, c, _ in self.iter_puzzle():
            for clr in self.iter_colors():
                vvar = self.var_vertex(r, c, clr)
                vset.add(vvar)
                if print_clauses: print(r, c, clr, vvar, self._parse_var_vertex(vvar, as_str=True))
        print(f"vertex size:{len(vset)} / expected:{self.n_cells * self.n_colors}")

        eset = set()
        for r, c, _ in self.iter_puzzle():
            for _, nr, nc in self.neighbors(r, c):
                for clr in self.iter_colors():
                    evar = self.var_edge(r,c,nr,nc,clr)
                    if evar in eset: continue
                    eset.add(evar)
                    if print_clauses: print(r, c, nr, nc, clr, evar, self._parse_var_edge(evar, as_str=True))
        print(f"edge size:{len(eset)} / expected:{((self.rows-1)*self.cols + (self.cols-1)*self.rows) * self.n_colors}")







def read_puzzle(puzzle_txt: str, ptype='rect') -> Puzzle:
    '''Convert a puzzle from a .txt file to a numerical representation.
    Unique numbers represent the same color; 0 represents an open cell.
    Assumes puzzle validity.  Returns the puzzle as an object.'''

    symbols = {}
    puzzle: Puzzle

    with open(puzzle_txt, 'r') as file:
        if ptype == 'rect':
            puzzle_grid = []
            for row in file.readlines():
                puzzle_row = []
                for char in row.removesuffix('\n'):
                    if char in ['.',' ']: # open cell
                        puzzle_row.append(0)
                    else:
                        if char not in symbols:
                            symbols[char] = len(symbols) + 1
                        puzzle_row.append(symbols[char])
                puzzle_grid.append(puzzle_row)

            puzzle = PuzzleRect(puzzle_grid)
        
        return puzzle
    
puzzle = read_puzzle('puzzle.txt')

print(puzzle)
puzzle.solve_puzzle()
puzzle._verify_clauses(print_clauses=True)

# print(Puzzle.exactly_two([1,2]))
# print(Puzzle.exactly_two([1,2,3]))