import numpy as np

import RoomMechanics as RM

class Clean:

    def __init__(self, room: np.array, mounds: np.array):

        self.mRoom = room

        # list of positions
        self.mMounds = mounds

        self.mRoomMechanics = RM.RoomMechanics(room)

        self.mRoomMechanics.show_room()
