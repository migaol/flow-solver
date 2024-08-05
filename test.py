import math
from collections import deque
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def do_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def is_visible(p1, p2, obstacles):
    for obstacle in obstacles:
        for i in range(len(obstacle)):
            A, B = obstacle[i], obstacle[(i + 1) % len(obstacle)]
            if do_intersect(p1, p2, A, B):
                return False
    return True

def build_visibility_graph(points, obstacles):
    graph = {p: [] for p in points}
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j and is_visible(p1, p2, obstacles):
                graph[p1].append(p2)
    return graph

def bfs_min_segments(graph, start, target):
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # If no path is found

def plot_solution(start, target, obstacles, visibility_graph, path):
    plt.figure(figsize=(7, 7))
    
    # Plot obstacles
    for obstacle in obstacles:
        x = [p.x for p in obstacle] + [obstacle[0].x]
        y = [p.y for p in obstacle] + [obstacle[0].y]
        plt.plot(x, y, 'k-')
    
    # Plot visibility graph
    for p1, neighbors in visibility_graph.items():
        for p2 in neighbors:
            plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g--', alpha=0.5)
    
    # Plot path
    if path:
        path_x = [p.x for p in path]
        path_y = [p.y for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Shortest Path')
    
    # Plot start and target points
    plt.plot(start.x, start.y, 'ro', label='Start')
    plt.plot(target.x, target.y, 'bo', label='Target')
    
    plt.legend()
    plt.xlim((0,10))
    plt.ylim((0,10))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Visibility Graph and Shortest Path')
    plt.grid(True)
    plt.show()

# Example usage
start = Point(5.5,3.5)
target = Point(10, 8)
obstacles = [
    [Point(x,y) for x,y in [(1,0), (1,1), (2,1), (2,2), (3,2), (3,3), (4,3), (4,0)]],
    [Point(x,y) for x,y in [(4,4), (6,4), (6,9), (4,9)]]
]

points = [start, target] + [v for obstacle in obstacles for v in obstacle]
visibility_graph = build_visibility_graph(points, obstacles)
path = bfs_min_segments(visibility_graph, start, target)

plot_solution(start, target, obstacles, visibility_graph, path)
