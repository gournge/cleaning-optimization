import numpy as np

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

        i = np.random.randint(3)
        if i == 0:
            return self.average_pooling_method()
        elif i == 1:
            return self.simplex_method()
        elif i == 2:
            return self.perlin_method()

    def average_pooling_method(self):
        """Generates random dirt and averages it out. 

        Returns:
            2d `np.array`        
        
        """
        
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

    def simplex_method(self):
        """Generates random dirt based on simplex method

        Args:
            subroom_size

        Returns:
            2d `np.array`        
        
        """
        
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

    def perlin_method(self):
        
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

                row.append(np.clip(noise_val, 0, 1))
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
        walls=np.zeros((m, m))

        for i in range (0, m, subroom):
            for j in range(0, m, subroom):
                if(np.random.random() > 0.7):
                    for z in range(subroom):
                        if(i+subroom <m and j+subroom<m):
                            walls[j, i+z]=2
                            walls[j+z, i]=2
                            walls[j+subroom, i+z]=2
                            walls[j+z, i+subroom]=2

        return walls