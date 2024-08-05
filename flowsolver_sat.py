'''
Solve flow puzzles by reducing to the boolean satisfiability problem.
This uses the Glucose 4.2.1 SAT solver in the pysat library.
More info on the Glucose solver: https://github.com/audemard/glucose
'''

import sys
from utilities import *
from pysat.solvers import Glucose42
from typing import Any, List, Dict, Tuple, Generator
from abc import ABC, abstractmethod
import itertools

class Puzzle(ABC):
    '''Puzzle information and SAT reducer.'''
    state: PuzzleState
    n_colors: int
    terminals: Dict[ColorID, Coord]
    def __init__(self, puzzle_source: str | PuzzleState) -> None:
        if isinstance(puzzle_source, str):
            self.state = self.from_txt(puzzle_source)
        else:
            self.state = puzzle_source
            self.find_terminals()

    @staticmethod
    def cell_is_empty(char: str) -> bool:
        '''Check whether a char represents an empty cell.'''
        return char in ['.',' ']

    @staticmethod
    def no_two(vars: List[int]) -> List[Clause]:
        '''Generate CNF clauses which dictate that no two (at most one) of the given variables are true.'''
        assert len(vars) > 1
        return ([-a, -b] for a, b in itertools.combinations(vars, 2))
    
    @staticmethod
    def exactly_one(vars: List[int]) -> List[Clause]:
        '''Generate CNF clauses which dictates that exactly one of the given variables are true.'''
        assert len(vars) > 0
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

    @staticmethod
    def same_parity(vars: List[int], exceptions: List[int] = []) -> List[Clause]:
        '''Generate CNF clauses which dictate that the variables in `vars` must all have the same parity,
        except when any of the literals in `exceptions` are true (variables can be true or false in exception conditions).'''
        assert len(vars) > 1
        clauses = []
        for (var1, var2) in itertools.combinations(vars, 2):
            clauses.append([-var1, var2] + exceptions) # var1      OR NOT var2  OR exceptions
            clauses.append([var1, -var2] + exceptions) # NOT var1  OR var2      OR exceptions
        return clauses

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
    def to_str(self, puzzle_state: Any) -> str:
        '''Represent the puzzle state as a string.'''

    @abstractmethod
    def from_txt(self, puzzle_txt: str) -> PuzzleState:
        '''Convert a puzzle from a .txt file to a numerical representation.
        Unique numbers represent the same color; 0 represents an open cell.
        Additionally finds an arbitrary/terminal for each color.
        Assumes puzzle validity.  Returns the puzzle as an object.'''

    @abstractmethod
    def find_terminals(self, *args, **kwargs) -> None:
        '''Find terminal cells from a grid input.'''

    @abstractmethod
    def var_vertex(self, *args, **kwargs) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents a cell being a certain color.'''
    
    @abstractmethod
    def var_edge(self, *args, **kwargs) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents an edge existing (or not).'''

    @abstractmethod
    def create_clauses(self, *args, **kwargs) -> List[Clause]:
        '''Create clauses for vertices and edges.  These clauses generally represent:
        - terminal cells are their original color, and at most this one color
        - terminal cells have exactly one neighbor with the same color
        - empty/pipe cells are exactly one color
        - empty/pipe cells which are connected have the same color
        - empty/pipe cells neighbor exactly two other cells of the same color
        '''

    @abstractmethod
    def parse_solution(self) -> Any:
        '''Convert SAT solution variables into a readable form.'''

    @abstractmethod
    def solve_puzzle(self, print_soln=False, verbose=False) -> Grid:
        '''Solve the puzzle and decode the solution.  Return the solution grid.'''

    @abstractmethod
    def _verify_satvars(self, print_clauses=False) -> None:
        '''Print the SAT variables and verify there are the correct number.'''

class PuzzleRect(Puzzle):
    '''Puzzle class implementation for standard rectangular puzzles.'''
    DIRECTIONS = [
        ('U', -1, 0),
        ('D', 1, 0),
        ('L', 0, -1),
        ('R', 0, 1)
    ]
    def __init__(self, source: str | Grid) -> None:
        super().__init__(source)
        self.n_colors = max(max(row) for row in self.state)
        self.rows = len(self.state)
        self.cols = len(self.state[0])
        self.n_cells = self.rows * self.cols
        self.edge_var_offset = self.n_cells * (self.n_colors+1) + 1 # self.n_colors+1 since colors are 1-indexed

    def __repr__(self) -> str:
        return self.to_str(self.state)
    
    def to_str(self, puzzle_state: Grid, alpha=False) -> str:
        if alpha: # breaks with more than 26 colors, which never happens in the game
            return '\n'.join(' '.join(f"{chr(ord('a') + cell-1)}" if cell > 0 else "-"
                                      for cell in row) for row in puzzle_state)

        max_length = max(len(str(cell)) for row in puzzle_state for cell in row)
        return '\n'.join(' '.join(f"{cell:0{max_length}d}" if cell > 0 else "-"*max_length
                                  for cell in row) for row in puzzle_state)
    
    def from_txt(self, puzzle_txt: str) -> PuzzleState:
        symbols = {}
        self.terminals = {}
        puzzle_grid: Grid = []

        with open(puzzle_txt, 'r') as file:
            for r,row in enumerate(file.readlines()):
                puzzle_row = []
                for c,char in enumerate(row.removesuffix('\n')):
                    if Puzzle.cell_is_empty(char): # open cell
                        puzzle_row.append(0)
                    else:
                        if char not in symbols:
                            symbols[char] = len(symbols) + 1
                            self.terminals[symbols[char]] = (r,c)
                        puzzle_row.append(symbols[char])
                puzzle_grid.append(puzzle_row)
        
        return puzzle_grid

    def find_terminals(self) -> None:
        self.terminals = {}
        for r,row in enumerate(self.state):
            for c,cell in enumerate(row):
                if not Puzzle.cell_is_empty(cell) and cell not in self.terminals:
                    self.terminals[cell] = (r,c)

    def var_vertex(self, r: int, c: int, clr: int) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents
        "the cell/vertex at row `r` and column `c` is `clr`".'''
        return (clr * self.cols + c) * self.rows + r

    def var_edge(self, r1: int, c1: int, r2: int, c2: int) -> int:
        '''Create a uniquely identifiable SAT variable, as an int, which represents
        "there exists an edge between the cell/vertex at row `r1` and column `c1`
        and the cell/vertex at row `r2` and column `c2`".'''

        # enforce deterministic edge variable regardless of order of the 2 vertices.
        # (r2,c2) will always be down/right of (r1,c1)
        if r1 * self.cols + c1 > r2 * self.cols + c2: r1, c1, r2, c2 = r2, c2, r1, c1

        edge_index = 2 * (r1 * self.cols + c1) # identify with the up/left endpoint cell/vertex
        if c2-c1 == 1: edge_index += 1 # up-down/vertical edge

        return self.edge_var_offset + edge_index
    
    def _parse_var_vertex(self, var: int, as_str=False) -> Tuple[int, int, int, bool] | str:
        '''Find the row, column, color, and parity corresponding to a numeric vertex variable.'''

        if var <= 0 or var >= self.edge_var_offset: raise ValueError("Variable does not correspond to a vertex.")

        r = var % self.rows
        temp = var // self.rows
        c = temp % self.cols
        clr = temp // self.cols

        if not as_str: return r, c, clr, var > 0
        return f"({r},{c}):{clr}" if var > 0 else f"NOT({r},{c}):{clr}"
    
    def _parse_var_edge(self, var: int, as_str=False) -> Tuple[int, int, int, int, bool] | str:
        '''Find the row and column of endpoint vertices, and parity corresponding to a numeric edge variable.'''
        
        if var < self.edge_var_offset: raise ValueError("Variable does not correspond to an edge.")
        
        edge_index = var - self.edge_var_offset
        endpoint = edge_index // 2
        is_horizontal = edge_index % 2 == 1
        
        r1 = endpoint // self.cols
        c1 = endpoint % self.cols
        
        if is_horizontal: r2, c2 = r1, c1 + 1
        else:             r2, c2 = r1 + 1, c1
        
        if not as_str: return r1, c1, r2, c2, var + self.edge_var_offset > 0
        
        return f"({r1},{c1})-({r2},{c2})" if var + self.edge_var_offset > 0 else f"NOT({r1},{c1})-({r2},{c2})"

    def _parse_var(self, var: int, as_str=False) -> Tuple | str:
        '''Parse a vertex or edge variable.'''
        parse_func = self._parse_var_vertex if var < self.edge_var_offset else self._parse_var_edge
        return parse_func(var, as_str=as_str)

    def iter_vertex(self) -> Generator[Tuple[int, int, int], None, None]:
        '''Iterate through row index, column index, and cell.'''
        for r, row in enumerate(self.state):
            for c, cell in enumerate(row):
                yield r, c, cell

    def iter_edge(self) -> Generator[Tuple[int, int, int, int], None, None]:
        '''Iterate through (r1,c1), (r2,c2) vertex combinations which represent edges.'''
        # horizontal
        for r, row in enumerate(self.state):
            for c, _ in enumerate(row[:-1]):
                yield r, c, r, c+1
        # vertical
        for r, row in enumerate(self.state[:-1]):
            for c, _ in enumerate(row):
                yield r, c, r+1, c

    def iter_colors(self) -> Generator[int, None, None]:
        '''Iterate over all unique numeric color identifiers in the puzzle.'''
        return range(1, self.n_colors+1)

    def valid_cell(self, r: int, c: int) -> bool:
        '''Check whether the cell at (row, col) exists in the puzzle.'''
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r: int, c: int) -> List[Coord]:
        '''Return a list of valid neighbors of the given cell coordinates.'''
        return [(r+dr, c+dc) for dir, dr, dc in self.DIRECTIONS if self.valid_cell(r+dr, c+dc)]

    def create_clauses(self, print_clauses=False) -> List[Clause]:
        clauses = []

        # vertex clauses
        for r, c, cell in self.iter_vertex():
            new_clauses = []

            if cell > 0: # terminal vertex
                # vertex is this color
                new_clauses.append([self.var_vertex(r,c,cell)])
                # vertex is at most one color
                new_clauses.extend([-self.var_vertex(r,c,clr)] for clr in self.iter_colors() if clr != cell)

                # exactly one incident edge exists
                incident_edges = [self.var_edge(r,c,nr,nc) for nr,nc in self.neighbors(r,c)]
                new_clauses.extend(Puzzle.exactly_one(incident_edges))
            
            else: # empty vertex, future pipe vertex
                # vertex is exactly one color
                cell_is_color = [self.var_vertex(r,c,clr) for clr in self.iter_colors()]
                new_clauses.extend(Puzzle.exactly_one(cell_is_color))

                # exactly two incident edges exist
                incident_edges = [self.var_edge(r,c,nr,nc) for nr,nc in self.neighbors(r,c)]
                new_clauses.extend(Puzzle.exactly_two(incident_edges))

            clauses.extend(new_clauses)
            if print_clauses:
                print(f"vertex ({r},{c}) clauses:")
                for clause in new_clauses:
                    print(clause, '; '.join(self._parse_var(var, as_str=True) for var in clause))

        # edge clauses
        for r1,c1,r2,c2 in self.iter_edge():
            new_clauses = []

            # some incident edge exists iff the cell and neighboring cell are the same color
            edge_var = self.var_edge(r1,c1,r2,c2)
            for clr in self.iter_colors():
                cell_is_color = self.var_vertex(r1,c1,clr)
                neighbor_is_color = self.var_vertex(r2,c2,clr)

                # cell and neighbor must be the same color, except if the edge does not exist
                new_clauses.extend(self.same_parity([cell_is_color, neighbor_is_color],
                                                    exceptions=[-edge_var]))
                # if the edge does not exist, then cell and neighbor must be different colors
                new_clauses.append([edge_var, -cell_is_color, -neighbor_is_color])

            clauses.extend(new_clauses)
            if print_clauses:
                print(f"edge ({r1},{c1})-({r2},{c2}) clauses:")
                for clause in new_clauses:
                    print(clause, '; '.join(self._parse_var(var, as_str=True) for var in clause))
        
        return clauses
    
    def parse_solution(self, soln: List[int]) -> Grid | None:
        if not soln: print("No solution"); return None
        
        soln.insert(0, 0) # insert at index 0 so that SAT variables can index directly into their parity in the solution
        
        puzzle_soln = [[0]*self.cols for _ in range(self.rows)]
        for r, c, _ in self.iter_vertex():
            for clr in self.iter_colors():
                if soln[self.var_vertex(r, c, clr)] > 0:
                    puzzle_soln[r][c] = clr
        
        return puzzle_soln

    def solve_puzzle(self, verbose=False, print_soln=False) -> Grid:
        ts = Timestamp()

        clauses = self.create_clauses(print_clauses=False)
        ts.add('create clauses')

        soln = Puzzle.solve_sat(clauses)
        ts.add('solve clauses')

        puzzle_soln = self.parse_solution(soln)
        ts.add('parse solution')

        if soln and print_soln: print(self.to_str(puzzle_soln, alpha=True))
        if verbose:
            print(f"number of clauses: {len(clauses):,}")
            print(ts.to_str())

        return puzzle_soln

    def _verify_satvars(self, print_clauses=False) -> None:
        vset = set()
        for r, c, _ in self.iter_vertex():
            for clr in self.iter_colors():
                vvar = self.var_vertex(r,c,clr)
                vset.add(vvar)
                if print_clauses: print(r, c, clr, vvar, self._parse_var_vertex(vvar, as_str=True))
        print(f"vertex size:{len(vset)} / expected:{self.n_cells * self.n_colors}")

        eset = set()
        for r1,c1,r2,c2 in self.iter_edge():
            evar = self.var_edge(r1,c1,r2,c2)
            if evar in eset: continue
            eset.add(evar)
            if print_clauses: print(r1, c1, r2, c2, evar, self._parse_var_edge(evar, as_str=True))
        print(f"edge size:{len(eset)} / expected:{(self.rows-1)*self.cols + (self.cols-1)*self.rows}")
