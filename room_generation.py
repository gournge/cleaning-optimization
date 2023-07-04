import numpy as np
import matplotlib.pyplot as plt
import opensimplex
from perlin_noise import PerlinNoise

def generate_room_method1(shape):

    room = np.zeros((shape, shape))

    k=np.random.randint(8, 12)

    m=shape # m - height, n-width
    # The room is generated from top left to bottom right, from top to bottom

    #generating walls on the sides
    for i in range(m):
        for j in range(m):
            # if i==m-1 or j==m-1 or i==0 or j==0:
            #     room[j, i]=2
            # else:
            room[j, i]=np.random.rand()
    
    
    # Randomly generating walls
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[j+p, i]=2
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[i, j+p]=2
    
    #getting the mean of neighbouring squares (cross model) and assigning it to the current square

    for k in range(0, 2):
        for i in range(m):
            for j in range(m):
                if j-1>=0 and i-1>=0 and j+1<m and i+1<m and room[j, i]!=2:
                    if(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5<=1:
                        room[j, i]=(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5
                    else: room[j, i]=(room[j, i]+room[j-1, i]+room[j+1, i]+room[j, i-1]+room[j, i+1])/5 - 0.2
        for i in range(2, m-2):
            room[0, i]=(room[1, i-1]+room[1, i+1]+room[1, i])/3
            room[m-1, i]=(room[m-1, i-1]+room[m-1, i+1]+room[m-1, i])/3
            room[0, i]=(room[1, i-1]+room[1, i+1]+room[1, i])/3
            room[m-1, i]=(room[m-1, i-1]+room[m-1, i+1]+room[m-1, i])/3
        
        for j in range(2, m-2):
            room[j, 0]=(room[j-1, 1]+room[j+1, 1]+room[j, 1])/3
            room[j, m-1]=(room[j-1, m-1]+room[j+1, m-1]+room[j, m-1])/3
            room[j, 0]=(room[j-1, 1]+room[j+1, 1]+room[j, 1])/3
            room[j, m-1]=(room[j-1, m-1]+room[j+1, m-1]+room[j, m-1])/3
        
    cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

    return room          
            

def generate_room_method2(shape):

    room = np.zeros((shape, shape))

    k=np.random.randint(8, 12)

    m=shape # m - height, n-width

    if(m==32):
        for i in range(0, 16):
            rand1=np.random.randint(0, m)
            rand2=np.random.randint(0, m)
            if(room[rand2, rand1]!=0.99999999):
                room[rand2, rand1]=0.99999999
                for j in range(1, k):
                    if(rand2-j>=0):
                        room[rand2-j, rand1]=room[rand2-j+1, rand1]/1.5
                for j in range(1, k):
                    if(rand2+j<m):
                        room[rand2+j, rand1]=room[rand2+j-1, rand1]/1.5
                for j in range(1, k):
                    if(rand1-j>=0):
                        room[rand2, rand1-j]=room[rand2, rand1-j+1]/1.5
                for j in range(1, k):
                    if(rand1+j<m):
                        room[rand2, rand1+j]=room[rand2, rand1+j-1]/1.5
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[j+p, i]=2
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[i, j+p]=2
    elif(m==64):
        for i in range(0, 32):
            rand1=np.random.randint(0, m)
            rand2=np.random.randint(0, m)
            if(room[rand2, rand1]!=0.99999999):
                room[rand2, rand1]=0.99999999
            for j in range(1, 2*k):
                if(rand2-j>=0):
                    room[rand2-j, rand1]=room[rand2-j+1, rand1]/1.5
            for j in range(1, k):
                if(rand2+j<m):
                    room[rand2+j, rand1]=room[rand2+j-1, rand1]/1.5
            for j in range(1, k):
                if(rand1-j>=0):
                    room[rand2, rand1-j]=room[rand2, rand1-j+1]/1.5
            for j in range(1, k):
                if(rand1+j<m):
                    room[rand2, rand1+j]=room[rand2, rand1+j-1]/1.5
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[j+p, i]=2
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[i, j+p]=2
    elif(m==128):
        for i in range(0, 64):
            rand1=np.random.randint(0, m)
            rand2=np.random.randint(0, m)
            if(room[rand2, rand1]!=0.99999999):
                room[rand2, rand1]=0.99999999
            for j in range(1, 3*k):
                if(rand2-j>=0):
                    room[rand2-j, rand1]=room[rand2-j+1, rand1]/1.5
            for j in range(1, k):
                if(rand2+j<m):
                    room[rand2+j, rand1]=room[rand2+j-1, rand1]/1.5
            for j in range(1, k):
                if(rand1-j>=0):
                    room[rand2, rand1-j]=room[rand2, rand1-j+1]/1.5
            for j in range(1, k):
                if(rand1+j<m):
                    room[rand2, rand1+j]=room[rand2, rand1+j-1]/1.5
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[j+p, i]=2
        for i in range(0, m, k):
            for p in range(0, m-k, k):
                if(np.random.randint(0, 10)>7.5):
                    for j in range(k):
                        room[i, j+p]=2
    
    cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

    return room   

def generate_room_method3(shape):
    m=shape
    k=np.random.randint(8, 12)
    room=np.zeros((m, m))
    opensimplex.seed(1234)
    for i in range(m):
        for j in range(m):
            random=np.random.randint(0, m*m)
            if(opensimplex.noise2(x=random, y=random)<0):
                room[j, i]=opensimplex.noise2(x=random, y=random)+1
            else: room[j, i]=opensimplex.noise2(x=random, y=random)
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[j+p, i]=2
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[i, j+p]=2

    cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    return room
def generate_room_method4(shape):
    m=shape
    k=np.random.randint(8, 12)
    noise1 = PerlinNoise(octaves=3)
    noise2 = PerlinNoise(octaves=6)
    noise3 = PerlinNoise(octaves=12)
    noise4 = PerlinNoise(octaves=24)
    room = []
    for i in range(m):
        row = []
        for j in range(m):
            noise_val = noise1([i/m, j/m])
            noise_val += 0.5 * noise2([i/m, j/m])
            noise_val += 0.25 * noise3([i/m, j/m])
            noise_val += 0.125 * noise4([i/m, j/m])

            row.append(noise_val)
        room.append(row)
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[j+p][i]=2
    for i in range(0, m, k):
        for p in range(0, m-k, k):
            if(np.random.randint(0, 10)>7.5):
                for j in range(k):
                    room[i][j+p]=2
    cmap = plt.cm.get_cmap('gray') # type: ignore # grayscale
    cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
    plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    return room
generate_room_method4(64)