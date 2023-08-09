import numpy as np
import matplotlib.pyplot as plt

# dirt generation
import opensimplex
import perlin_noise


class RoomGenerator:

    def __init__(self, room_size: int):
        """
        Args:
            room_size: How big is the whole grid in the simulation.
                       See Readme what sizes are available.

        Returns:
            `RoomGenerator` object with various generation methods

        """
        
        room_sizes_dict = { 35  : (6, 9), 
                            71  : (8, 9),
                            119 : (6, 10) }

        if room_size not in room_sizes_dict.keys():
            raise NotImplementedError("A room of such size cannot be generated")
        
        self.room_size = room_size
        self.subroom_sizes = room_sizes_dict[room_size]


    def any_method(self):
        """Generate a room with any (of preset) subroom size by a randomly chosen method

        """

        subroom_size = np.random.choice(self.subroom_sizes)

        i = np.random.randint(2)
        if i == 0:
            return self.average_pooling_method(subroom_size)
        elif i == 1:
            return self.simplex_method(subroom_size)

    def average_pooling_method(self, subroom_size: int):
        """Generates random dirt and averages it out. 

        Args:
            subroom_size

        Returns:
            2d `np.array`        
        
        """
        if subroom_size not in self.subroom_sizes:
            raise NotImplementedError("A room with such subroom dimensions cannot be generated")
        else:
            m=self.room_size
            room=np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    room[j, i]=np.random.rand()
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

            walls=self.__generate_walls()

            for i in range(m):
                for j in range(m):
                    if(walls[j, i]>room[j, i]):
                        room[j, i]=walls[j, i]
            
            return room

        temp = self.__generate_walls()

        # special case
        if (self.room_size, subroom_size) == (35, 7):
            pass

        pass


    def simplex_method(self, subroom_size: int):
        """Generates random dirt based on simplex method

        Args:
            subroom_size

        Returns:
            2d `np.array`        
        
        """

        if subroom_size not in self.subroom_sizes:
            raise NotImplementedError("A room with such subroom dimensions cannot be generated")
        else:
            m=self.room_size
            room=np.zeros((m, m))
            opensimplex.seed(1234)
            for i in range(m):
                for j in range(m):
                    random=np.random.randint(0, m*m)
                    if(opensimplex.noise2(x=random, y=random)<0):
                        room[j, i]=opensimplex.noise2(x=random, y=random)+1
                    else: room[j, i]=opensimplex.noise2(x=random, y=random)
            
            walls=self.__generate_walls()

            for i in range(m):
                for j in range(m):
                    if(walls[j, i]>room[j, i]):
                        room[j, i]=walls[j, i]
            
            return room


        temp = self.__generate_walls()

        # special case
        if (self.room_size, subroom_size) == (35, 7):
            pass

        pass

    def perlin_method(self, subroom_size: int):
        if subroom_size not in self.subroom_sizes:
            raise NotImplementedError("A room with such subroom dimensions cannot be generated")
        else:
            m=self.room_size
            noise1 = perlin_noise.PerlinNoise(octaves=4)
            noise2 = perlin_noise.PerlinNoise(octaves=8)
            noise3 = perlin_noise.PerlinNoise(octaves=16)
            noise4 = perlin_noise.PerlinNoise(octaves=32)
            room = np.zeros((m, m))
            for i in range(m):
                row = []
                for j in range(m):
                    noise_val = noise1([i/m, j/m])
                    noise_val += 0.5 * noise2([i/m, j/m])
                    noise_val += 0.25 * noise3([i/m, j/m])
                    noise_val += 0.125 * noise4([i/m, j/m])

                    row.append(abs(noise_val))
                room[i]=row

            walls=self.__generate_walls()

            for i in range(m):
                for j in range(m):
                    if(walls[j, i]>room[j, i]):
                        room[j, i]=walls[j, i]
            
            return room

    def __generate_walls(self):
        m=self.room_size
        subroom = self.subroom_sizes[np.random.randint(0, 2)]
        prob=8
        if m==35 and subroom==6:
            prob=6
        print(subroom)
        walls=np.zeros((m, m))

        for i in range (subroom, m, subroom):
            for j in range(subroom, m, subroom):
                random=np.random.randint(0, 10)
                if(np.random.randint(0, 10)>prob):
                    for z in range(subroom-1):
                        if(random<5 and i+subroom-1<m):
                            walls[j, i+z+1]=2
                        elif(random>5 and j+subroom-1<m):
                             walls[j+z+1, i]=2
                        # if(i+subroom-1 <m and j+subroom-1<m):
                        #     walls[j, i+z]=2
                        #     walls[j+z, i]=2
                        #     walls[j+subroom, i+z]=2
                        #     walls[j+z, i+subroom]=2
        print(walls)
        return walls


        """Private method

        Returns:
            2d `np.array` with value 2 where walls should be  
        
        """

        

        pass
        

roomGenerator = RoomGenerator(35)

room=roomGenerator.perlin_method(6)

cmap = plt.cm.get_cmap('gray_r') # type: ignore # grayscale
cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
plt.axis('off')
plt.show()

room=roomGenerator.simplex_method(6)

cmap = plt.cm.get_cmap('gray_r') # type: ignore # grayscale
cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
plt.axis('off')
plt.show()

room=roomGenerator.average_pooling_method(6)

cmap = plt.cm.get_cmap('gray_r') # type: ignore # grayscale
cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
plt.imshow(room, cmap=cmap, vmin=0, vmax=1)
plt.axis('off')
plt.show()
