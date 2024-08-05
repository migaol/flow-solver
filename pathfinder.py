from utilities import *
import matplotlib.pyplot as plt
from typing import Set
from manim import *
from utilities import Point as Pt
from collections import deque

class Pathfinder:
    def __init__(self, path: List[Coord], cell_size: float, pct_size: float) -> None:
        self.path = [Pt(x,y) for x,y in path]
        self.cell_size = cell_size
        self.pct_size = pct_size
        self.create_vertices(self.path, cell_size, pct_size)

    def find_corners(self, path: List[Pt]) -> List[int]:
        corners = []
        prev_dir = path[1] - path[0]
        for i in range(1, len(path) - 1):
            curr_dir = path[i+1] - path[i]
            if curr_dir != prev_dir: corners.append((i, prev_dir, curr_dir))
            prev_dir = curr_dir
        
        return corners

    def create_vertices(self, path: List[Pt], cell_size: float, pct_size: float):
        V: Set[Pt] = set()
        O: Set[Pt] = set()
        half_size = cell_size/2
        half_cell = (half_size,half_size)

        e_radius = half_size * pct_size
        v_radius = half_size * (pct_size - 0.1)
        d_radius = v_radius - e_radius

        src_mid = path[0]*cell_size + half_cell
        tgt_mid = path[-2]*cell_size + half_cell
        for cell in path[1:-1]: # cell centers
            V.add(cell*cell_size + half_cell)

        # source vertices/edges
        src_dir: Pt = path[1] - path[0]
        src_o1 = path[0]*cell_size + half_cell - src_dir*e_radius - src_dir.inverted()*e_radius
        src_o2 = path[0]*cell_size + half_cell - src_dir*e_radius + src_dir.inverted()*e_radius
        src_i1 = path[0]*cell_size + half_cell + src_dir*e_radius - src_dir.inverted()*e_radius
        src_i2 = path[0]*cell_size + half_cell + src_dir*e_radius + src_dir.inverted()*e_radius
        S = [
            src_o1 - src_dir*d_radius - src_dir.inverted()*d_radius,
            src_o2 - src_dir*d_radius + src_dir.inverted()*d_radius,
            src_i1 + src_dir*d_radius - src_dir.inverted()*d_radius,
            src_i2 + src_dir*d_radius + src_dir.inverted()*d_radius,
            src_mid
        ]
        V.update(S)
        O.add((src_o1, src_o2)) # src cap

        # intermediate vertices/edges
        prev1, prev2 = src_o1, src_o2
        for i, prev_dir, curr_dir in self.find_corners(path):
            curr: Pt = path[i]
            curr_i: Pt = curr*cell_size + half_cell - prev_dir*e_radius + curr_dir*e_radius # inner
            curr_o: Pt = curr*cell_size + half_cell + prev_dir*e_radius - curr_dir*e_radius # outer

            if curr_i.parallel_axis(prev1):   O.update([(prev1, curr_i), (prev2, curr_o)])
            elif curr_i.parallel_axis(prev2): O.update([(prev2, curr_i), (prev1, curr_o)])
            
            prev1, prev2 = curr_i, curr_o
            V.update([
                curr_i - prev_dir*d_radius + curr_dir*d_radius,
                curr_o + prev_dir*d_radius - curr_dir*d_radius
            ])
        
        # target vertices/edges
        tgt_dir: Pt = path[-2] - path[-1]
        tgt_o1 = path[-1]*cell_size + half_cell - tgt_dir*e_radius - tgt_dir.inverted()*e_radius
        tgt_o2 = path[-1]*cell_size + half_cell - tgt_dir*e_radius + tgt_dir.inverted()*e_radius
        T = [
            path[-2]*cell_size + half_cell - tgt_dir*v_radius - tgt_dir.inverted()*v_radius,
            path[-2]*cell_size + half_cell - tgt_dir*v_radius + tgt_dir.inverted()*v_radius,
            path[-2]*cell_size + half_cell - tgt_dir.inverted()*v_radius,
            path[-2]*cell_size + half_cell + tgt_dir.inverted()*v_radius,
            tgt_mid
        ]
        V.update(T)

        O.add((tgt_o1, tgt_o2)) # tgt cap
        if tgt_o1.parallel_axis(prev1):   O.update([(prev1, tgt_o1), (prev2, tgt_o2)])
        elif tgt_o1.parallel_axis(prev2): O.update([(prev2, tgt_o1), (prev1, tgt_o2)])

        self.V, self.O = V, O
        self.S, self.T = S, T

    def get_neighbors(self, v: Pt) -> List[Pt]:
        neighbors = []
        for u in self.V:
            if u == v or u in self.S: continue
            if not line_intersects_polygon(v, u, list(self.O)): neighbors.append(u)
        # print(f"neighbors {v} : {neighbors}")
        return neighbors

    def find_path(self) -> List[Pt]:
        start_points = self.S
        end_points = self.T
        visited = set()
        q = deque([(pt, [pt]) for pt in start_points])

        while q:
            current_point, path = q.popleft()

            if current_point in end_points: return path

            for neighbor in self.get_neighbors(current_point):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))

        return []


class VizPathfinder(Scene):
    def construct(self, path: List[Pt] = [(2,0),(1,0),(1,1),(1,2),(1,3)],
                  rows: int = 5, cols: int = 5, cell_size: float = 50, pct_size: float = 0.8):
        self.camera.frame_width, self.camera.frame_height = cols*cell_size, rows*cell_size

        pf = Pathfinder(path, cell_size, pct_size)
        vertices = list(pf.V)
        outline = list(pf.O)
        soln = pf.find_path()
        print(soln)

        half_cell = cell_size/2
        def tlo(pt: np.ndarray):
            x,y,z = pt[:3]
            return np.array([x - self.camera.frame_width/2, -y + self.camera.frame_height/2, z])
            # return np.array([x,y,z])

        # temp
        temp = VGroup()
        temp_lines = []
        temp_lines = [(np.array([x0,y0,0]),np.array([x1,y1,0])) for x0,y0,x1,y1 in temp_lines]
        for start, end in temp_lines:
            temp.add(Line(tlo(start), tlo(end), color=RED, stroke_width=100))

        # path
        path_points = [np.array([x * cell_size + half_cell, y * cell_size + half_cell, 0]) for x, y in path]
        path_lines = VGroup()
        for start, end in zip(path_points[:-1], path_points[1:]):
            path_lines.add(Line(tlo(start), tlo(end), color=RED, stroke_width=100))

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
            outline_lines.add(Line(tlo(np.array([x0, y0, 0])), tlo(np.array([x1, y1, 0])), color=GREEN, stroke_width=25))

        # vertices
        vertex_dots = VGroup()
        for vertex in vertices:
            x, y = vertex
            vertex_dots.add(Dot(tlo(np.array([x, y, 0])), radius=1, color=BLUE))

        self.play(
            # Create(temp),
            Create(path_lines),
            Create(vertex_dots),
            Create(outline_lines),
            Create(soln_lines)
        )
        self.wait(2)

if __name__ == '__main__':
    path = [(2,0), (1, 0), (1, 1), (1, 2), (1,3), (0, 3), (0,4), (0,5), (1,5), (2,5), (2,6)]
    cell_size = 10.0
    pct_size = 0.9

    pf = Pathfinder(path, cell_size, pct_size)
    # print(pf.V)
    # print(pf.O)
    soln = pf.find_path()
    print(soln)

'''
manim -s -r 1200,1200 pathfinder.py VizPathfinder
'''