import numpy as np
import configparser 
import matplotlib.pyplot as plt

import Utility

class RoomMechanics:

    def __init__(self, room: np.array):
        """
            toolset for operating in a room 
        """
    
        self.mRoom = room

        # read the config file settings 
        config = configparser.ConfigParser()
        config.read('config-cleaning-mechanics.ini')

        # tiles with values under this threshold will be considered to be clean
        self.mCLEAN_THRESHOLD = float( config['GENERAL']['clean_threshold'] ) 
        
        # measured in cells
        self.mBROOM_WIDTH     = float( config['BROOM']['broom_width'] )
        # measured in collective value of dirt
        self.mBROOM_WIDTH     = float( config['BROOM']['broom_capacity'] )

    def show_room(self):
        """
            matplotlib graph pops up
        """

        cmap = plt.cm.get_cmap('gray') # grayscale
        cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
        
        plt.imshow(self.mRoom, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

    def move_broom(self, pos1: np.array, pos2: np.array):
        """
            Transport cells forward along the line L created by pos1, pos2.\n
            Area of the broom is the rectangle such that the midpoints of its 2 sides with lengths 

            >>> self.mBROOM_WIDTH

            form a line coinciding with L.\n
            See Readme for a more precise explanation. 
        """
        
        # params of the line equation 
        a, b, c = Utility.line_equation(pos1, pos2)

        # angle of the line 
        tilt = None
        if   a == 0:
            tilt = .0
        elif b == 0:
            tilt = np.pi/2
        else: 
            tilt = np.tan( -a/b )

        # construct the rectangle/area of influence of the broom 

        # vector from the middle of side of the rectangle to its left corner
        half = Utility.rotate( np.array([self.mBROOM_WIDTH, 0]), 
                               tilt + np.pi/2 )

        corner_left_1  = pos1 + half 
        corner_right_1 = pos1 - half

        corner_left_2  = pos2 + half
        corner_right_2 = pos2 - half

        # TODO: find cells included in the rectangle

        pass        

    def is_clean(self, room):
        """
            returns false if there is a dirty cell, true otherwise
        """

        for r in room: 
            for el in r:
                # discard walls
                if el == 2: continue

                if el > self.mCLEAN_THRESHOLD:
                    return False
                
        return True