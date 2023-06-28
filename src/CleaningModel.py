import numpy as np

import RoomMechanics as RM

class Clean:

    def __init__(self, room: np.array, mounds: np.array):

        # list of positions
        self.mounds = mounds
         
        self.room_environment = RM.RoomMechanics(room, self.mounds)

    