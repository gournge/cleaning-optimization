import numpy as np

def line_equation(pos1: np.array, pos2: np.array):
    """
        Yields a, b, c in the equation 
        >>> ax + by + c = 0
    """

    x1, y1 = pos1
    x2, y2 = pos2

    if x1 == x2: return -1, 0, x1
    if y1 == y2: return 0, -1, y1

    # y = px + q

    p = (x1 - x2) / (y1 - y2) 
    q = y1 - p * x1

    return p, -1, q


def to_left(coeffs: tuple[float, float, float], pos: np.array):
    """
        Returns true if the position is to left of the line (or to bottom when b=0)
        >>> ax + by + c = 0
    """
    a, b, c = coeffs
    x, y = pos
    p = -c/b
    q = -c/a

    if a == 0: return y < p
    if b == 0: return x < q

    return y < p * x + q

def rotate(pos: np.array, theta: float):
    """
        Returns pos rotated by theta.\n
        Positive theta means counterclockwise.
    """

    # rotation matrix

    rot = np.array([ [np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]  ] )

    return np.dot(rot, pos)

    