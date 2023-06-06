import numpy as np
import matplotlib.pyplot as plt

def generate_room(shape: tuple[int, int]):

    room = np.zeros(shape)

    k=np.random.randint(8, 12)

    m, n=room.shape # m - height, n-width
    # The room is generated from top left to bottom right, from top to bottom

    #generating walls on the sides
    for i in range(n):
        for j in range(m):
            if i==n-1 or j==m-1 or i==0 or j==0:
                room[j, i]=2
            else:room[j, i]=np.random.rand()
    
    # Randomly generating walls
    for i in range(0, n, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[j+p, i]=2
    for i in range(0, m, k):
        for p in range(0, n-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[i, j+p]=2
    
    #getting the mean of neighbouring squares (cross model) and assigning it to the current square

    for k in range(0, 4):
        for i in range(n):
            for j in range(m):
                if j-1>=0 and i-1>=0 and j+1<m and i+1<n and room[j, i]!=2:
                    if(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5<=1:
                        room[j, i]=(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5
                    else: room[j, i]=(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5 - 0.2
        for i in range(2, n-2):
            room[1, i]=(room[1, i-1]+room[1, i+1]+room[1, i])/3
            room[m-1, i]=(room[m-1, i-1]+room[m-1, i+1]+room[m-1, i])/3
            room[1, i]=(room[1, i-1]+room[1, i+1]+room[1, i])/3
            room[m-1, i]=(room[m-1, i-1]+room[m-1, i+1]+room[m-1, i])/3
        
        for j in range(2, m-2):
            room[j, 1]=(room[j-1, 1]+room[j+1, 1]+room[j, 1])/3
            room[j, n-1]=(room[j-1, n-1]+room[j+1, n-1]+room[j, n-1])/3
            room[j, 1]=(room[j-1, 1]+room[j+1, 1]+room[j, 1])/3
            room[j, n-1]=(room[j-1, n-1]+room[j+1, n-1]+room[j, n-1])/3
        


    
    # TODO: warped grid with randomly deleted edges
    cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    

    plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    return room