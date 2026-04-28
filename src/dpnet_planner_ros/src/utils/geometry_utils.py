import numpy as np


def create_rectangle_vertex(length, width):
    """Create rectangular vertices centered at origin for given dimensions."""
    start_x = -length / 2
    start_y = -width / 2

    point0 = np.array([[start_x], [start_y]])
    point1 = np.array([[start_x + length], [start_y]])
    point2 = np.array([[start_x + length], [start_y + width]])
    point3 = np.array([[start_x], [start_y + width]])

    return np.hstack((point0, point1, point2, point3))


def generate_inequalities(vertex):
    """Generate inequality constraints from rectangle vertices."""
    G = np.zeros((4, 2))
    h = np.zeros((4, 1))

    for i in range(4):
        if i + 1 < 4:
            pre_point = vertex[:, i]
            next_point = vertex[:, i + 1]
        else:
            pre_point = vertex[:, i]
            next_point = vertex[:, 0]

        diff = next_point - pre_point
        a = diff[1]
        b = -diff[0]
        c = a * pre_point[0] + b * pre_point[1]

        G[i, 0] = a
        G[i, 1] = b
        h[i, 0] = c

    return G, h
