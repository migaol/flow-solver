'''Line-specific utility functions.'''

from utilities import *

def intersect(p1: Vect2, q1: Vect2, p2: Vect2, q2: Vect2) -> bool:
    def orientation(p: Vect2, q: Vect2, r: Vect2) -> int:
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

def intersects_square(p: Vect2, q: Vect2, square: LRTB) -> bool:
    '''Check if line segment pq intersects a square.'''
    if p.in_square(square) or q.in_square(square): return True # endpoints in square

    square_edges = [
        (Vect2(square.l, square.t), Vect2(square.l, square.b)), # left
        (Vect2(square.l, square.b), Vect2(square.r, square.b)), # bottom
        (Vect2(square.r, square.b), Vect2(square.r, square.t)), # right
        (Vect2(square.r, square.t), Vect2(square.l, square.t))  # top
    ]

    for edge in square_edges:
        if intersect(p, q, edge[0], edge[1]): return True

    return False

def intersects_polygon(p: Vect2, q: Vect2, polygon: List[Vect2]):
    '''Check if line segment pq intersects a (possibly concave) polygon.'''
    for edge in polygon:
        p1,q1 = edge
        if intersect(p, q, p1, q1): return True
    return False