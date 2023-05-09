import os
import csv
import numpy as np 

import NeatCleaningModel as Clean
import PlanMoundsModel as Mounds

def main():

    room = np.genfromtxt("premade-rooms/room1.csv", delimiter = ',')

    # list of positions
    mounds = Mounds.PlanMounds(room)

    model = Clean.Clean(room, mounds)

    # moves = []

    # while not Mech.is_clean(room):

    #     room, move = model.find_move(room)

    #     moves.append(move)

    # print(moves)


if __name__ == '__main__':

    main()
