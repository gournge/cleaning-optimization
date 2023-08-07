import numpy as np
from tensorflow import Tensor
from tensorflow import convert_to_tensor


def preprocess(room: np.array, mounds) -> Tensor:
    """Converts two `np.array`s of size NxN to a representation through a tensor object of dimensions either `(1, N, N, 3)` (first dimension is for batch size) 
    
    Arguments:
    ---------
    - `np.array` room with walls of value 2 and dirt of value between 0 and 1
    - `np.array` array of 2d positions in the room where mounds are

    Returns:
    --------
    - `tf.Tensor` : [x][y][0] is for dirt, [x][y][1] is for walls, [x][y][2] is for mounds.

    """


    d = room.shape[0]
    assert d == room.shape[1], "Preprocesses only square rooms"

    out = np.zeros((1, d, d, 3))

    for r, row in enumerate(room):
        for c, el in enumerate(row):
            
            if el == 2: 
                out[0][r][c][1] = 1.
            else:
                out[0][r][c][0] = el

    for mound in mounds:
        x, y = mound
        out[0][x][y][0] = 0.
        out[0][x][y][1] = 0.
        out[0][x][y][2] = 1. 

    return convert_to_tensor(out)

def update_broom(room_size: int, action: int, broom):
    """

    Args: 
    ----
    - `room_size`
    - `action` 
        - 0 - 3 is for beg of movement
        - 4 - 7 is for end of movement

    - `broom` np.array of 4 integers

    Returns: 
    ----
    - updated broom if action is not 8

    Order of action is top/down/left/right 

    """

    # beg of movement
    if action == 0:
        broom[1] += 1
    elif action == 1:
        broom[1] -= 1
    elif action == 2:
        broom[0] -= 1
    elif action == 3:
        broom[0] += 1

    # end of movement
    if action == 4:
        broom[3] += 1
    elif action == 5:
        broom[3] -= 1
    elif action == 6:
        broom[2] -= 1
    elif action == 7:
        broom[2] += 1

    return np.clip(broom, 0, room_size-1)

def amount_of_dirt(room: np.array, mounds):
    """
    Arguments:
    ---------
    - room with dirt
    - mounds plan
    
    Returns:
    -------
    - float
    
    """

    out = 0
    d = room.shape[0]

    for x in range(d):
        for y in range(d):

            if (x, y) in mounds: continue
            out += room[x, y]

    return out
            


def discard_to_left(points, line, bottom = True):
    """
        Inputs:
        - list of 2d np.array points
        - coeffs of a line in form of (a, b, c)
        - (optional) whether to discard points under the line (in case its horizontal)
        
        Returns:
        - all points except those that are to the left of the given line.
    """

    horizontal = (line[0] == 0)
    vertical = (line[1] == 0)
    val = -line[2]/line[1] if not vertical else None

    def discard(p):

        if vertical:
            return to_left(line, p)

        if horizontal:
            if bottom:
                return p[1] < val
            return p[1] > val
        
        return to_left(line, p)

    return [p for p in points if not discard(p)]

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

    p = (y1 - y2) / (x1 - x2) 
    q = y1 - p * x1

    return p, -1, q


def to_left(coeffs: tuple[float, float, float], pos: np.array):
    """
        Returns true if the position is to left of the line (or to bottom when b=0)
        >>> ax + by + c = 0
    """
    a, b, c = coeffs
    x, y = pos

    if a == 0: return y < -c/b
    if b == 0: return x < -c/a

    p = -a/b
    q = -c/b

    if p > 0: 
        return y > p * x + q
    return y < p * x + q


def to_left_points(pos1: np.array, pos2: np.array, pos: np.array):

    return to_left( line_equation(pos1, pos2), pos )

def cells_between(pos1: np.array, pos2: np.array, line_eq = None):
    """
        you can provide a, b, c from 
        >>> ax + by + c = 0

    """

    cells = []

    a, b, c = 0, 0, 0

    if line_eq == None:
        a, b, c = line_equation(pos1, pos2)

    x1, y1 = pos1
    x2, y2 = pos2

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    print(x_min, x_max)
    print(y_min, y_max)

    # horizontal
    if a == 0: 
        for x in np.arange(x_min, x_max+1):
            cell = np.array([int(x), int(-c/b)])
            cells.append(cell)
    
    # vertical
    if b == 0:
        for y in np.arange(y_min, y_max+1):
            cell = np.array([int(-c/a), int(y)])
            cells.append(cell)

    # not steep
    if x_max - x_min > y_max - y_min:
        for x in np.arange(x_min, x_max+1):
            y_new = (a*x + c)/(-b) 
            cell = np.array([int(x), int(y_new)])
            cells.append(cell)
        return cells
        
    # steep
    for y in np.arange(y_min, y_max+1):
        x_new = (b*y + c)/(-a) 
        cell = np.array([int(x_new), int(y)])
        cells.append(cell)
    return cells

def dist_point_to_line(pos: np.array, beg1: np.array, beg2: np.array):
    """
        Distance from pos to the line passing through beg1, beg2
    """
    # beg1 != beg2 guarranteed 
    return np.abs( np.cross(beg2 - beg1, pos - beg1) ) / np.linalg.norm(beg2 - beg1) 


def rotate(pos: np.array, theta: float):
    """
        Returns pos rotated by theta.\n
        Positive theta means counterclockwise.
    """

    # rotation matrix

    rot = np.array([ [np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]  ] )

    return np.dot(rot, pos)

def intersection_between_lines(line1, line2):
    """
        assume they are not parallel
    """

    a = np.array( [ [line1[0], line1[1]], [line2[0], line2[1]] ] )
    b = np.array(   [line1[2], line2[2]])
    b *= -1 

    return np.linalg.solve( a, b )

def perpendicular_through(line, point):
    """
        input a line and a point

        return a line perpendicular to this line through this point
    
    """

    a, b, c = line

    # horizontal
    if a == 0:
        # vertical
        return 1, 0, -point[0] 

    # vertical
    if b == 0:
        # horizontal
        return 0, 1, -point[1]
    
    # y = -a/b x - c/b

    # y1 = b/a x1 - c/b 
    
    # c/b = b/a x1 - y1 

    m = b/a 

    return -m, 1, m * point[0] - point[1]

def parallel_through(line, point):

    a, b, c = line
    
    # horizontal
    if a == 0:
        # horizontal
        return 1, 0, -point[1] 

    # vertical
    if b == 0:
        # vertical
        return 0, 1, -point[0]
    
    # y = -a/b x - c / b

    # y1 = -a/b x1 - c/b 
    
    # c/b = -a/b x1 - y1 

    m = a/b

    return m, 1, -m * point[0] - point[1]

def create_list_of_sides(corners):

    sides = [ (corners[0], corners[1]),
              (corners[2], corners[3]),
              (corners[0], corners[2]),
              (corners[1], corners[3]) ]
    
    return sides

# # y = 2x - 3
# print(perpendicular_through((2, -1, -3), (0, 0))) # should return y =  -1/2 x : 1  2 0
# print(parallel_through     ((2, -1, -3), (0, 0))) # should return y =     2 x : 2 -1 0

# print(intersection_between_lines( (1, 1, 1), (3, 2, 4) ))
