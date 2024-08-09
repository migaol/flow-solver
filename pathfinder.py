from utilities import *
from typing import Set
from manim import *
from utilities import Vect2
import lines
from collections import deque

class Pathfinder:
    def __init__(self, path: List[Coord], cell_size: float, pct_size: float) -> None:
        self.path = [Vect2(x,y) for x,y in path]
        self.cell_size = cell_size
        self.pct_size = pct_size
        self.create_vertices(self.path, cell_size, pct_size)

    def find_corners(self, path: List[Vect2]) -> List[int]:
        corners = []
        prev_dir = path[1] - path[0]
        for i in range(1, len(path) - 1):
            curr_dir = path[i+1] - path[i]
            if curr_dir != prev_dir: corners.append((i, prev_dir, curr_dir))
            prev_dir = curr_dir
        
        return corners

    def create_vertices(self, path: List[Vect2], cell_size: float, pct_size: float):
        V: Set[Vect2] = set()
        O: Set[Vect2] = set()
        half_size = cell_size/2
        half_cell = (half_size,half_size)

        e_radius = half_size * pct_size
        v_radius = half_size * (pct_size - 0.01)

        N = 5
        spacing = 2*v_radius / (N-1)
        ds = [-1,-0.5,0,0.5,1]
        dxs = [Vect2(x,0) for x in ds]
        dys = [Vect2(0,y) for y in ds]
        def get_cell_vertices25(cell: Vect2) -> List[Vect2]:
            v = []
            center = cell*cell_size + half_cell
            for dy in dys:
                for dx in dxs:
                    v.append(center + dy*v_radius + dx*v_radius)
            return v
        
        get_cell_method = get_cell_vertices25

        # source vertices/edges
        src_dir: Vect2 = path[1] - path[0]
        src_o1 = path[0]*cell_size + half_cell - src_dir*e_radius - src_dir.inverted()*e_radius
        src_o2 = path[0]*cell_size + half_cell - src_dir*e_radius + src_dir.inverted()*e_radius
        S = get_cell_method(path[0])
        V.update(S)
        O.add((src_o1, src_o2)) # src cap

        # intermediate edges
        prev1, prev2 = src_o1, src_o2
        for i, prev_dir, curr_dir in self.find_corners(path):
            curr: Vect2 = path[i]

            curr_i: Vect2 = curr*cell_size + half_cell - prev_dir*e_radius + curr_dir*e_radius # inner
            curr_o: Vect2 = curr*cell_size + half_cell + prev_dir*e_radius - curr_dir*e_radius # outer
            if curr_i.parallel_axis(prev1):   O.update([(prev1, curr_i), (prev2, curr_o)])
            elif curr_i.parallel_axis(prev2): O.update([(prev2, curr_i), (prev1, curr_o)])
            
            prev1, prev2 = curr_i, curr_o

        T = []
        for i,(curr,prev) in enumerate(zip(path[1:], path[:-1])):
            path_dir: Vect2 = curr - prev
            vs = [curr*cell_size + half_cell - path_dir*half_size + path_dir.inverted()*v_radius*d for d in ds]
            V.update(vs) # border cells
            if i == len(path)-2:
                T += vs
                for j in range(-N//2,N+1):
                    new_vs = [v + path_dir*spacing*j for v in vs]
                    V.update(new_vs)
                    T += new_vs
                for j in range(-N,-N//2):
                    new_vs = [v + path_dir*spacing*j for v in vs]
                    V.update(new_vs)
            elif i == len(path)-3:
                pass
            else:
                for j in range(1,N+1):
                    new_vs = [v + path_dir*spacing*j for v in vs]
                    V.update(new_vs)
        
        # target edges
        tgt_dir: Vect2 = path[-2] - path[-1]
        tgt_o1 = path[-1]*cell_size + half_cell - tgt_dir*e_radius - tgt_dir.inverted()*e_radius
        tgt_o2 = path[-1]*cell_size + half_cell - tgt_dir*e_radius + tgt_dir.inverted()*e_radius

        O.add((tgt_o1, tgt_o2)) # tgt cap
        if tgt_o1.parallel_axis(prev1):   O.update([(prev1, tgt_o1), (prev2, tgt_o2)])
        elif tgt_o1.parallel_axis(prev2): O.update([(prev2, tgt_o1), (prev1, tgt_o2)])

        self.V, self.O = V, O
        self.S, self.T = S, T

    def get_neighbors(self, v: Vect2, visited: Set[Vect2] = {}) -> List[Vect2]:
        neighbors = []
        for u in self.V:
            # check visited here before border intersection because the latter is expensive
            if u == v or u in self.S or u in visited: continue
            if not lines.intersects_polygon(v, u, list(self.O)): neighbors.append(u)
            # don't need to cache; each (u,v) line-border intersection is calculated at most once
        return neighbors

    def find_path(self) -> List[Vect2]:
        start_points = self.S
        end_points = self.T
        visited = set()
        q = deque([(Vect2, [Vect2]) for Vect2 in start_points])

        while q:
            current_point, path = q.popleft()

            if current_point in end_points: return path

            for neighbor in self.get_neighbors(current_point, visited):
                visited.add(neighbor)
                q.append((neighbor, path + [neighbor]))

        return []


class VizPathfinder(Scene):
    def construct(self,
                  path: List[Vect2] = [(0,0),(1,0),(1,1),(1,2),(1,3),(1,4)],
                #   path: List[Vect2] = [(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(2,2),(1,2),(1,3),(0,3),(0,4)],
                  rows: int = 6, cols: int = 6,
                  cell_size: float = 50,
                  pct_size: float = 0.67):
        self.camera.frame_width, self.camera.frame_height = cols*cell_size, rows*cell_size

        pf = Pathfinder(path, cell_size, pct_size)
        vertices = list(pf.V)
        outline = list(pf.O)
        soln = pf.find_path()
        import timeit
        time_taken = timeit.timeit(lambda: pf.find_path(), number=1)
        soln = pf.find_path()
        print(f"Time taken to find path: {time_taken:.6f} seconds")
        print(soln)

        half_cell = cell_size/2
        def tlo(Vect2: np.ndarray):
            x,y,z = Vect2[:3]
            return np.array([x - self.camera.frame_width/2, -y + self.camera.frame_height/2, z])

        # temp
        temp = VGroup()
        temp_lines = []
        temp_lines = [(np.array([x0,y0,0]),np.array([x1,y1,0])) for x0,y0,x1,y1 in temp_lines]
        for start, end in temp_lines:
            temp.add(Line(tlo(start), tlo(end), color=RED, stroke_width=100))

        grid_lines = VGroup()
        for r in range(rows + 1): # horizontal
            start = np.array([0, r * cell_size, 0])
            end = np.array([cols * cell_size, r * cell_size, 0])
            grid_lines.add(Line(tlo(start), tlo(end), color=DARK_GRAY, stroke_width=100, stroke_opacity=0.5))
        for c in range(cols + 1): # vertical
            start = np.array([c * cell_size, 0, 0])
            end = np.array([c * cell_size, rows * cell_size, 0])
            grid_lines.add(Line(tlo(start), tlo(end), color=DARK_GRAY, stroke_width=100, stroke_opacity=0.5))

        # Highlight the first and last squares
        start_square = Square(side_length=cell_size, fill_color=GREEN, fill_opacity=0.25)
        end_square = Square(side_length=cell_size, fill_color=RED, fill_opacity=0.25)
        start_square.move_to(tlo(np.array([path[0][0] * cell_size + half_cell, path[0][1] * cell_size + half_cell, 0])))
        end_square.move_to(tlo(np.array([path[-1][0] * cell_size + half_cell, path[-1][1] * cell_size + half_cell, 0])))

        # path line
        path_points = [np.array([x * cell_size + half_cell, y * cell_size + half_cell, 0]) for x, y in path]
        path_lines = VGroup()
        for start, end in zip(path_points[:-1], path_points[1:]):
            path_lines.add(Line(tlo(start), tlo(end), color=WHITE, stroke_width=100))

        # soln
        soln_points = [np.array([x,y,0]) for x, y in soln]
        soln_lines = VGroup()
        for start, end in zip(soln_points[:-1], soln_points[1:]):
            soln_lines.add(Line(tlo(start), tlo(end), color=ORANGE, stroke_width=100))

        # outline
        outline_lines = VGroup()
        for outline_line in outline:
            p0, p1 = outline_line
            x0,y0 = p0
            x1,y1 = p1
            outline_lines.add(Line(tlo(np.array([x0, y0, 0])), tlo(np.array([x1, y1, 0])), color=GRAY, stroke_width=50))

        # vertices
        vertex_dots = VGroup()
        for vertex in vertices:
            if vertex in pf.S: clr = GREEN
            elif vertex in pf.T: clr = RED
            else: clr = BLUE
            x, y = vertex
            vertex_dots.add(Dot(tlo(np.array([x, y, 0])), radius=1, color=clr))

        self.play(
            # Create(temp),
            Create(grid_lines),
            Create(start_square),Create(end_square),
            Create(path_lines),
            Create(vertex_dots),
            Create(outline_lines),
            Create(soln_lines),
        )
        self.wait(2)

if __name__ == '__main__':
    path = [(2,0), (1, 0), (1, 1), (1, 2), (1,3), (0, 3), (0,4), (0,5), (1,5), (2,5), (2,6)]
    cell_size = 10.0
    pct_size = 0.9

    pf = Pathfinder(path, cell_size, pct_size)
    soln = pf.find_path()
    print(soln)

'''
manim -s -r 1200,1200 pathfinder.py VizPathfinder
'''