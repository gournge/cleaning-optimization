import numpy as np
import configparser 
import matplotlib.pyplot as plt

import Utility

class RoomMechanics:

    def __init__(self, room: np.array):
        """
            toolset for operating in a room 
        """
    
        self.room = room

        # read the config file settings 
        config = configparser.ConfigParser()
        config.read('config-cleaning-mechanics.ini')

        # tiles with values under this threshold will be considered to be clean
        self.CLEAN_THRESHOLD = float( config['GENERAL']['clean_threshold'] ) 
        
        # measured in cells
        self.BROOM_WIDTH     = float( config['BROOM']['broom_width'] )
        # measured in collective value of dirt
        self.BROOM_CAPAACITY = float( config['BROOM']['broom_capacity'] )

    def show_room(self):
        """
            matplotlib graph pops up
        """

        cmap = plt.cm.get_cmap('gray') # grayscale
        cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
        

        plt.imshow(self.room, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

    def move_broom(self, pos1: np.array, pos2: np.array):
        """
            Transport cells forward along the line L created by pos1, pos2.\n
            Area of the broom is the rectangle such that the midpoints of its 2 sides with lengths 

            >>> self.BROOM_WIDTH

            form a line coinciding with L.\n
            See Readme for a more precise explanation. 
        """
        
        # if they are too close we don't consider any movement to take place
        if (abs(pos1[0] - pos2[0]) < 0.1) and (abs(pos1[1] - pos2[1]) < 0.1):
            return

        # plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], marker = 'o')


        # params of the line equation 
        a, b, c = Utility.line_equation(pos1, pos2)

        # angle of the line 
        tilt = None
        if   a == 0:
            tilt = .0
        elif b == 0:
            tilt = np.pi/2
        else: 
            tilt = np.arctan(-a/b)

        # construct the rectangle/area of influence of the broom 

        # vector from the middle of side of the rectangle to its left corner
        half = Utility.rotate( np.array([self.BROOM_WIDTH/2, 0]), 
                               tilt + np.pi/2 )

        corner_left_1  = pos1 + half 
        corner_right_1 = pos1 - half

        corner_left_2  = pos2 + half
        corner_right_2 = pos2 - half

        sides = [ (corner_left_1,  corner_right_1),
                  (corner_left_2,  corner_right_2), 
                  (corner_left_1,  corner_left_2 ), 
                  (corner_right_1, corner_right_2) ]


        # show the rectangle
        # for side in sides:
        #     plt.plot([ side[0][0], side[1][0] ], [side[0][1], side[1][1]])

        # for bottom, top, left, right sides of rect
        equations = [Utility.line_equation(p1, p2) for p1, p2 in sides]

        def is_inside(pos: np.array): 

            if (abs(pos[0] - pos1[0]) <= 0.5) and (abs(pos[1] - pos1[1])) <= 0.5:
                return True

            if (abs(pos[0] - pos2[0]) <= 0.5) and (abs(pos[1] - pos2[1])) <= 0.5:
                return True

            bottom = Utility.to_left(equations[0], pos)
            top    = Utility.to_left(equations[1], pos)
            
            left   = Utility.to_left(equations[2], pos)
            right  = Utility.to_left(equations[3], pos)

            x1, y1 = pos1
            x2, y2 = pos2

            if (x1 <= x2) and (y1 <= y2):
                return ((not bottom) and top ) and ( right and (not left) )

            if (x1 < x2) and (y1 > y2):
                return ((not bottom) and top ) and ( (not right) and left )
            
            if (x1 > x2) and (y1 < y2):
                return (bottom and (not top) ) and ( (not right) and left )
            
            if (x1 > x2) and (y1 > y2):
                return (bottom and (not top) ) and ( right and (not left) )

            


        verts =  [corner_left_1, corner_right_1, corner_left_2, corner_right_2]
        x_s = [vert[0] for vert in verts ]
        y_s = [vert[1] for vert in verts ]

        # print(verts)

        # ... the rect contains
        min_x = int ( np.floor(min(x_s)) )         
        max_x = int ( np.ceil(max(x_s)) ) 
        min_y = int ( np.floor(min(y_s)) )         
        max_y = int ( np.ceil(max(y_s)) )

        # print(min_x, min_y, max_x, max_y)

        points_inside = [] # the rectangle

        # iterate through candidates to see what points are contained in the rectangle
        for x in np.arange(min_x, max_x+1):
            for y in np.arange(min_y, max_y+1):
                point = np.array([x, y])
                if not self.is_valid(point): continue
                if is_inside(point):
                    points_inside.append(point)

        for point in points_inside:
            # print(point)
            x, y = point
            self.room[y, x] = 2

    def is_valid(self, point):
        x, y = point
        return (0 <= x) and (x < len(self.room)) and (0 <= y) and (y < len(self.room[0]))

    def is_clean(self):
        """
            returns false if there is a dirty cell, true otherwise
        """

        for r in self.room: 
            for el in r:
                # discard walls
                if el == 2: continue

                if el > self.CLEAN_THRESHOLD:
                    return False
                
        return True