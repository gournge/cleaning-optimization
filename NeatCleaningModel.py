import numpy as np

import RoomMechanics as RM

class Clean:

    def __init__(self, room: np.array, mounds: np.array):

        self.room = room

        # list of positions
        self.mounds = mounds

        self.room_mechanics = RM.RoomMechanics(self.room)

        self.room_mechanics.move_broom((5, 5), (7, 2))

        self.room_mechanics.show_room()
