import numpy as np
import random as rand

import matplotlib.pyplot as plt

def generate_room(shape: tuple[int, int]):

    room = np.zeros(shape)

    k=int(rand.uniform(3, 12))

    m, n=room.shape # m - height, n-width

    for i in range(m):
        for j in range(n):
            if i%k==0 or j%k==0 or i==m-1 or j==n-1:
                room[i, j]=2 

    for l in range(0, n, k):
        for j in range(0, m, k):
            if rand.uniform(0, 1)>0.25:
                for i in range(k):
                    if i+j <m:
                        room[i+j, l]=0
    for l in range(0, n, k):
        for i in range(0, m, k):
            if rand.uniform(0, 1)>0.25:
                for j in range(k):
                    if(j+l<n):
                        room[i, j+l]=0
    for i in range(n):
        for j in range(m):
            if i==n-1 or j==m-1 or i==0 or j==0:
                room[j, i]=2 
    
    for i in range(n):
        for j in range(m):
            if room[j, i]!=2:
                room[j, i]=rand.uniform(0, 1)


    
    # TODO: warped grid with randomly deleted edges
    # cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    # cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    

    # plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    # plt.axis('off')
    # plt.show()
    return room