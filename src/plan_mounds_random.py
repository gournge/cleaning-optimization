import numpy as np
import warnings

def plan_mounds_random(room: np.array, mounds_number: int) :
    """Arranges mounds randomly 
    Arguments:
    ---------
    - 2d `np.array`
    - number of mounds to plan

    Returns:
    -------
    - list of 2d positions of mounds
    """

    out = []
    d = room.shape[0]
    while mounds_number > 0:
        
        x, y = np.random.randint(d, size=2)

        if room[x, y] == 2: continue
        
        out.append((x, y))
        mounds_number -= 1
            
    return out
