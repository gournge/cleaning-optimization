import os
import csv
import numpy as np 

import CleaningModel 
import PlanMoundsModel
import RoomGeneration 

def main():

    room = np.genfromtxt("premade-rooms/room2.csv", delimiter = ',')
    # room = RoomGeneration.generate_room((64, 64))
    # room = np.random.rand(35, 35)

    # list of positions
    mounds = PlanMoundsModel.PlanMounds(room)

    model = CleaningModel.Clean(room, mounds)
    
    # moves = []

    # while not Mech.is_clean(room):

    #     room, move = model.find_move(room)

    #     moves.append(move)

    # print(moves)


if __name__ == '__main__':

    main()
