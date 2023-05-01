import numpy as np

# we consider a tile under this level of dirt to be clean 
CLEAN_THRESHOLD = 0.1

def move_broom(room, pos1, pos2):

    return room

def is_clean(room):

    for r in room: 
        for el in r:
            # discard walls
            if el == 1: continue

            if el > CLEAN_THRESHOLD:
                return False
            
    return True
    