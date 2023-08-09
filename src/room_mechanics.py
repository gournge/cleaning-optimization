from pathlib import Path 
import configparser 

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from src import utility

class RoomMechanics:
    """
        Conventions:
        -------
        - points are 2d vectors - np.arrays 
        - Rects (rectangles) are lists of points
    """

    def __init__(self, room: np.array, mounds: np.array):
        """Environment simulating moving a broom 
        """
    
        self.room = room

        self.room_width = len(room)
        self.room_height = len(room[0])

        self.mounds = mounds

        # reset every move
        self.cleaned_dirt = 0

        # read the config file settings 

        config = configparser.ConfigParser()
        parent_dir = Path(__file__).parent.parent.absolute() 
        path_to_config = Path.joinpath(parent_dir, "config.cfg")
        config.read(path_to_config)

        self.BROOM_MOVEMENT_LENGTH_RANGE = ( float(config['BROOM']['min_movement_length']),
                                             float(config['BROOM']['max_movement_length']) )
        
        self.BROOM_WIDTH    = float( config['BROOM']['broom_width'] )
        self.BROOM_CAPACITY = float( config['BROOM']['broom_capacity'] )
        
        self.BROOM_IMPACT_IS_RELATIVE = config['BROOM']['broom_impact_is_relative'].lower() == "true"
        
        self.BROOM_IMPACT_RELATIVE_FRONT = float( config['BROOM']['broom_impact_relative_front'] )
        self.BROOM_IMPACT_RELATIVE_SIDE  = float( config['BROOM']['broom_impact_relative_side'] )
        self.BROOM_IMPACT_ABSOLUTE_FRONT = float( config['BROOM']['broom_impact_absolute_front'] )
        self.BROOM_IMPACT_ABSOLUTE_SIDE  = float( config['BROOM']['broom_impact_absolute_side'] )

        self.BROOM_REDISTRIBUTION_MAX_ITERATIONS = int( config['BROOM']['broom_redistribution_max_iterations'] )


    def show_room(self, list_of_rects = [], colored_line_eqs = [], colored_segments=[]):
        """Show current room layout with overlayed elements.

            Args:
            ----
            - `list_of_rects` (optional) - list of tuples of positions of form `(x, y)`
            - `colored_line_eqs` (optional) - list of tuples of three numbers
            - `colored_segments` (optional) - list of tuples of four numbers. tuples are of form `(x1, y1, x2, y2)`
            
        """

        # note the transposition
        new_room = 1 - self.room.T
        new_room[new_room <= -1] = 2

        cmap = plt.cm.get_cmap('gray') # grayscale
        cmap.set_over((0, 0.8, 1)) # specific value for walls (value 2)
        
        plt.imshow(new_room, cmap=cmap, vmin=0, vmax=1)
        
        for mound in self.mounds:
            rect = patches.Rectangle(np.array(mound) - 0.5, 1, 1, facecolor='purple')
            ax = plt.gca()
            ax.add_patch(rect)

        for rect in list_of_rects:
            rect = utility.create_list_of_sides(rect)
            for side in rect:
                plt.plot([ side[0][0], side[1][0] ], [side[0][1], side[1][1]])

        for color, line_eq in colored_line_eqs:
            a, b, c = line_eq

            if b == 0: 
                plt.plot([0, self.room_width], [-c/b, -c/b], color)
                continue

            # y = - a/b x - c/b
            plt.plot([0, self.room_width], [-c/b, - a/b * self.room_width - c/b ], color)

        if colored_segments:
            if len(colored_segments[0]) == 2:
                for color, coords in colored_segments:
                    x1, y1, x2, y2 = coords
                    plt.plot([x1, x2], [y1, y2], color)
            else:
                for coords in colored_segments:
                    x1, y1, x2, y2 = coords
                    plt.plot([x1, x2], [y1, y2])

        plt.axis('off')
        plt.show()

    def move_broom(self, pos1: np.array, pos2: np.array):
        """
            Transport cells forward along the line L created by pos1, pos2.\n
            
            Parameters
            ----------
            - 2d np.array initial position
            - 2d np.array ending position

            Returns:
            --------
            cleaned_dirt : float
                Amount of dirt that has fallen down inside the imaginary holes at the spots of mounds.
            spillover_dirt : float
                the amount of dirt that disappeared by accident (the alg couldn't have found space for it)
            collided : bool
                `True` if the movement collided with a wall (had to be clipped) or was too short; `False` otherwise.

            If `cleaned_dirt` is negative it means that an error occured inside the environment - it should be discarded.

            Area of the broom is the rectangle such that the midpoints of its 2 sides with lengths 

            >>> self.BROOM_WIDTH

            form a line coinciding with L.\n
            If the movement collides with some cell or exceeds the wall, we clip the ractangle to the smallest possible without collision.
            The dirt is distributed to the front (and to the sides in case of exceeded capacity during movement.)
            
            See Readme for a more precise explanation. 
        """
        
        self.cleaned_dirt = 0

        if_corrected_forwards, if_corrected_sides = False, False

        rect_front, rect_main, half, tilt = self.__create_pointing_forward(pos1, pos2)

        if_corrected_forwards, rect_main, rect_front, too_close = self.__correct_pointing_forward(rect_main, rect_front, tilt)

        if too_close: 
            return 0, 0, True 

        def valid_len(len_rect):
            return (len_rect >= self.BROOM_MOVEMENT_LENGTH_RANGE[0] ) and (len_rect <= self.BROOM_MOVEMENT_LENGTH_RANGE[1] )

        if not valid_len( np.linalg.norm(rect_main[0] - rect_main[3]) ): 
            # no spillover dirt
            return 0, 0, True

        point_of_exceeded_capacity, front_mass, residue_mass, points_main = self.__find_exceeded_capacity_point(rect_main)

        output_rects_points = [self.__inside_points(rect_front)]

        if point_of_exceeded_capacity is not None:

            rect_left, rect_right = self.__create_pointing_sides(rect_main, half, tilt, point_of_exceeded_capacity)

            if_corrected_sides, rect_left, rect_right = self.__correct_pointing_sides(rect_left, rect_right, tilt)

            output_rects_points.extend( [self.__inside_points(r) for r in [rect_left, rect_right] ] )
        
        for point in points_main:
            x, y = point
            if self.room[x, y] != 2:
                self.room[x, y] = 0

        points_redistribution = None
        if point_of_exceeded_capacity is not None:
            facing_downward_rect_main = (rect_main[0][1] > rect_main[2][1])
            beg_line = utility.line_equation(rect_main[0], rect_main[1])
            discard_line = utility.parallel_through(beg_line, point_of_exceeded_capacity)
            points_redistribution = utility.discard_to_left(points_main, discard_line, bottom = facing_downward_rect_main)
        else:
            points_redistribution = points_main

        if not points_redistribution:
            return -1, 0, True

        spillover_dirt = 0
        for i, output_points in enumerate(output_rects_points):
            if i == 0:
                spillover_dirt += self.__distribute_dirt(front_mass, output_points, points_redistribution)
            else: 
                spillover_dirt += self.__distribute_dirt(residue_mass/2, output_points, points_redistribution)

        return self.cleaned_dirt, spillover_dirt, (if_corrected_forwards or if_corrected_sides)

    def __inside_points(self, corners):
        """
            Parameters
            ----------
            - list of 2d vectors (corners: left bottom, right bottom, left top, right top)
            - (optional) whether the corners describe main rectangle
            - (optional) positions generating the main rectangle

            Returns: 
            - list of points inside the rectangle created by corners

            Conditions:
            - corners has to be a 2d np.array vector
            - if is_main_rect is set to True then guiding_points has to include these points 

        """

        x_s = [vert[0] for vert in corners ]
        y_s = [vert[1] for vert in corners ]

        # ... the rect contains
        min_x = int ( np.floor(min(x_s)) )         
        max_x = int ( np.ceil(max(x_s)) ) 
        min_y = int ( np.floor(min(y_s)) )         
        max_y = int ( np.ceil(max(y_s)) )

        points_inside = [] # the rectangle

        # to apply operations from shapely module
        rect = Polygon([corners[0], corners[1], corners[3], corners[2]])

        # iterate through candidates to see what points are contained in the rectangle
        # and check if the rect is valid
        for x in np.arange(min_x, max_x+1):
            for y in np.arange(min_y, max_y+1):
                point = np.array([x, y])

                if not self.is_valid(point): continue

                shapely_point = Point(point)
                if rect.contains(shapely_point):
                    points_inside.append(point)

        return points_inside

    def __create_pointing_forward(self, pos1, pos2):
        """
            Parameters
            ----------
            - pos1, pos2 are np.array 2d vectors

            Returns: 
            -------
            - list of vertices of front rect
            - list of vertices of main rect
            - vector from pos1 to left vertex 
            - the tilt of the main rectangle
        """

        def create_rect_main():
            rect_main = []

            # params of the line equation 
            a, b, c = utility.line_equation(pos1, pos2)

            # angle of the line 
            tilt = None
            if   a == 0:
                tilt = .0
            elif b == 0:
                tilt = np.pi/2
            else: 
                tilt = np.arctan(-a/b)

            # vector from the middle of side of the rectangle to its left corner
            half = utility.rotate( np.array([self.BROOM_WIDTH/2, 0]), 
                                tilt + np.pi/2 )
            
            rect_main.append(pos1 + half) 
            rect_main.append(pos1 - half)

            rect_main.append(pos2 + half)
            rect_main.append(pos2 - half)

            return tilt, half, rect_main
        
        tilt, half, rect_main = create_rect_main()

        def create_front_rect():
            extension_vector = None
            if self.BROOM_IMPACT_IS_RELATIVE:
                whole_movement_length = np.linalg.norm(pos1 - pos2)
                extension_vector_length = self.BROOM_IMPACT_RELATIVE_FRONT * whole_movement_length
                extension_vector = utility.rotate(np.array([extension_vector_length, 0]), tilt)
            else:
                extension_vector = utility.rotate(np.array([self.BROOM_IMPACT_ABSOLUTE_FRONT, 0]), tilt)

            new_left  = rect_main[2]  + extension_vector
            new_right = rect_main[3] + extension_vector 

            return [rect_main[2], rect_main[3], new_left, new_right]

        return create_front_rect(), rect_main, half, tilt

    def __correct_pointing_forward(self, rect_main, rect_front, tilt):
        """
            Parameters
            ----------
            - old main rect
            - old front rect

            Returns:
            - True/False : whether they are new
            - new main rect
            - new front rect
            - whether the collision point was too close
        """

        points_main  = self.__inside_points(rect_main) 
        points_front = self.__inside_points(rect_front) 

        collision_points = [] 

        for point in points_main:
            x, y = point
            if self.room[x, y] == 2:
                collision_points.append(point)

        if not collision_points:
            for point in points_front:
                x, y = point
                if self.room[x, y] == 2:
                    collision_points.append(point)

        if not collision_points: 
            return False, rect_main, rect_front, False
        
        b1, b2 = rect_main[0], rect_main[1]
        collision_point = min( collision_points, key = (lambda p: utility.dist_point_to_line(p, b1, b2)) )

        dist_collision_to_front = np.linalg.norm(rect_main[0] - rect_front[3]) - utility.dist_point_to_line(collision_point, rect_main[0], rect_main[1])

        if dist_collision_to_front < self.BROOM_MOVEMENT_LENGTH_RANGE[0]: 
            return False, None, None, True

        vec = utility.rotate([dist_collision_to_front, 0], tilt)

        new_main_top_left  = rect_front[0] - vec
        new_main_top_right = rect_front[1] - vec

        new_front_left  = rect_front[2] - vec
        new_front_right = rect_front[3] - vec

        return True, [rect_main[0], rect_main[1], new_main_top_left, new_main_top_right], [new_main_top_left, new_main_top_right, new_front_left, new_front_right], False
    
    def __find_exceeded_capacity_point(self, rect_main):
        """
            Parameters
            ---------- 
            - the main rectangle

            Returns:
            ---
            - point at which main rectangle's capacity is exceeded
            - 
        
        """
        points = self.__inside_points(rect_main)

        tot_mass, residue_mass = 0, 0
        exceeded_point = None
        for point in points:
            x, y = point
            tot_mass += self.room[x, y]
            if exceeded_point is not None: continue           
            if tot_mass > self.BROOM_CAPACITY:
                exceeded_point = point
                residue_mass = tot_mass

        # residue mass is moved to front since it meets broom's capacity 
        return exceeded_point, residue_mass, tot_mass - residue_mass, points

    def __create_pointing_sides(self, rect_main, half, tilt, exceeded_capacity_point):
        """
            Parameters
            ----------
            - main rectangle
            - vector from mid of side of main_rect to its closest vertex
            - angle between X axis and line through guiding midpoints of main_rect
            - 2d np.array vector where brooms capacity has been exceeded
            ---
            Returns: 
            - left and right rectangles on the side of the rectangle
        
        """

        side_extension = self.BROOM_IMPACT_RELATIVE_SIDE * (2 * half) if self.BROOM_IMPACT_IS_RELATIVE else utility.rotate([self.BROOM_IMPACT_ABSOLUTE_SIDE, 0], np.pi/2 + tilt)
        
        dist_critical_to_begin = utility.dist_point_to_line(exceeded_capacity_point, rect_main[0], rect_main[1])
        side_parallel_to_tilt = utility.rotate([dist_critical_to_begin, 0], tilt)

        # inside corners at bottom
        inside_left  = rect_main[0]  + side_parallel_to_tilt
        inside_right = rect_main[1] + side_parallel_to_tilt

        # outside corners at bottom
        corner_left_left_1   = inside_left  + side_extension
        corner_right_right_1 = inside_right - side_extension

        # outside corners at top
        corner_left_left_2   = rect_main[2] + side_extension
        corner_right_right_2 = rect_main[3] - side_extension 

        rleft  = [corner_left_left_1,   inside_left,  corner_left_left_2,   rect_main[2] ]
        rright = [inside_right, corner_right_right_1, rect_main[3], corner_right_right_2]

        return rleft, rright

    def __correct_pointing_sides(self, rect_left, rect_right, tilt):
        """
            Parameters
            ----------
            - rectangles on the left and right
            - tilt of the main rectangle
            
            Returns:
            ------
            - whether they have changed
            - updated rectangles on the left and right
        
        """
        
        points_left  = self.__inside_points(rect_left)
        points_right = self.__inside_points(rect_right)

        collision_left, collision_right = [], []

        for point in points_left:
            x, y = point
            if self.room[x, y] == 2:
                collision_left.append(point)

        for point in points_right:
            x, y = point
            if self.room[x, y] == 2:
                collision_right.append(point)

        # if nothing changed
        if (not collision_left) and (not collision_right):
            return False, rect_left, rect_right

        closest_left  = None if not collision_left else min(collision_left, key = lambda p : utility.dist_point_to_line(p, rect_left[1], rect_left[3]))
        closest_right = None if not collision_right else min(collision_right, key = lambda p : utility.dist_point_to_line(p, rect_right[0], rect_right[2]))
            
        d_l = utility.dist_point_to_line(closest_left, rect_left[1], rect_left[3]) if closest_left is not None else None
        d_r = utility.dist_point_to_line(closest_right, rect_right[0], rect_right[2]) if closest_right is not None else None


        side_width = np.linalg.norm(rect_right[0] - rect_right[1])

        vec_l = np.array([0, 0]) if d_l is None else utility.rotate([side_width - d_l, 0], tilt - np.pi/2)
        vec_r = np.array([0, 0]) if d_r is None else utility.rotate([side_width - d_r, 0], tilt + np.pi/2)

        new_left  = [vec_l + rect_left[0],          rect_left[1], vec_l + rect_left[2],          rect_left[3]]
        new_right = [       rect_right[0], vec_r + rect_right[1],        rect_right[2], vec_r + rect_right[3]]

        return True, new_left, new_right

    def __distribute_dirt(self, amount_of_dirt, points_output, points_redistribution):
        """

            Distributes randomly dirt in the output rectangle;
            if its capacity is exceed, dirt is distributed back to the main rectangle (above the line of exceeded capacity point)

            Parameters
            ----------
            - amount of dirt to put in the output rectangle
            - output points (e.g. sides, front)
            - redistribition points (if exceeded capacity in initial output points)

            Returns:
            -------
            - amount of dirt that has been lost (no space for it has been found)

        """

        # all dirt has been moved to a mound
        for mound in self.mounds:
            for point in points_output:
                dist_vec = np.array(mound) - point
                if abs(dist_vec[0]) < 0.5 and abs(dist_vec[1]) < 0.5:
                    self.cleaned_dirt += amount_of_dirt
                    return 0

        capacity_rect_output = 0
        for point in points_output:
            x, y = point
            capacity_rect_output += 1 - self.room[x, y]

        if amount_of_dirt > capacity_rect_output:
                    
            residue_mass_from_front = amount_of_dirt - capacity_rect_output

            no_iterations = 0
            m = self.BROOM_REDISTRIBUTION_MAX_ITERATIONS * len(points_redistribution)

            while residue_mass_from_front > 0:
                if no_iterations > m: break
                no_iterations += 1

                point = random.choice(points_redistribution)
                x, y = point
                val = self.room[x, y]
                if val >= 1: continue

                # amount of dirt we will append to a cell
                inc = np.clip(np.random.rand(), 0, (1 - val) * 0.1 )
                self.room[x, y] += inc
                residue_mass_from_front -= inc

            for x, y in points_output:
                if self.room[x, y] != 2:
                    self.room[x, y] = 1

            # if after redistriubtion any leftovers exist
            return residue_mass_from_front

        no_iterations = 0
        m = self.BROOM_REDISTRIBUTION_MAX_ITERATIONS * len(points_output)
        
        while amount_of_dirt > 0: 
            if no_iterations > m: break
            no_iterations += 1 

            point = random.choice(points_output)
            x, y = point
            val = self.room[x, y]
            if val >= 1: continue

            # amount of dirt we will append to a cell
            inc = 1 - val 
            self.room[x, y] = 1 
            amount_of_dirt -= inc

        # if after redistribution there are any leftovers
        return amount_of_dirt

    def is_valid(self, point):
        x, y = point
        return (0 <= x) and (x < self.room_width) and (0 <= y) and (y < self.room_height)