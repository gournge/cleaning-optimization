import numpy as np
import warnings

def plan_mounds_naive(room: np.array, mounds_number: int) :
    """Arranges mounds naively assuming that the densest dirt areas are optimal
    
    Arguments:
    ---------
    - 2d `np.array`
    - number of mounds to plan

    Returns:
    -------
    - list of 2d positions of mounds
    """

    warnings.warn("Planning mounds is random instead of naive", NotImplemented)

    out = []
    d = room.shape[0]
    while mounds_number > 0:
        
        x, y = np.random.randint(d, size=2)

        if room[x, y] == 2: continue
        
        out.append((x, y))
        mounds_number -= 1
            
    return out
