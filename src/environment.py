from src import room_mechanics
from src import room_generation
from src import plan_mounds_naive
from src import utility

import numpy as np

class CleaningEnv:
    """
        State of the environment is a `(1, N, N, 3)` tensor. 
        The depth is not 4 since previous broom movement is already remembered by the agent - there is no need to create a seperate layer in the tensor for it.
    
    """

    def __init__(self, room_size, punish_clipping, mounds = None, mounds_number = None):
        """Create an env of given size. Provide mounds plan yourself or let the program generate it automatically.
        
        
        Args:
        ----
            - `room_size`
            - `punish_clipping` how much smaller should the reward be if movement was clipped (e.g `0.5`)
            - `mounds` optional - list of positions
            - `mounds_number` 
            
        """

        if mounds is None and mounds_number is None: 
            raise KeyError('you have to provide number of mounds to be generated.') 

        self.room_size = room_size
        self.punish_clipping = punish_clipping

        self.room_generator = room_generation.RoomGenerator(room_size)
        
        # room = self.room_generator.any_method()
        room = room_generation.generate_room_method4(self.room_size)


        final_mounds = None
        if mounds is None:
            final_mounds = plan_mounds_naive.plan_mounds_naive(room, mounds_number)
        else:
            final_mounds = mounds

        # needs the room layout (with dirt) and the plan of mounds
        self.room_mechanics = room_mechanics.RoomMechanics(room=room, mounds=final_mounds)

        self.previous_action = None

    def render(self):
        actions = [] if self.previous_action is None else ['r', self.previous_action]
        self.room_mechanics.show_room(colored_segments=actions)

    def act(self, broom):
        """
            Broom is a tuple of four floats describing coords of endpoints of the broom.
        """

        # pos1, pos2 = broom[:1], broom[2:]

        pos1 = np.random.rand(2) * self.room_size
        pos2 = np.random.rand(2) * self.room_size 

        print(pos1, pos2)

        cleaned_dirt, _, clipped = self.room_mechanics.move_broom(pos1, pos2)

        reward = cleaned_dirt if not clipped else (1 + cleaned_dirt) * (1 - self.punish_clipping) 

        return reward, utility.preprocess(self.room_mechanics.room, self.room_mechanics.mounds)
    
    def reset(self, mounds = None, mounds_number = None):
        """New room layout. Returns the state of the environment. (No broom)
        
        """

        if mounds is None and mounds_number is None:
            mounds_number = len(self.room_mechanics.mounds)

        self.previous_action = None

        # room = self.room_generator.any_method()
        room = room_generation.generate_room_method4(self.room_size)

        final_mounds = None
        if mounds is None:
            final_mounds = plan_mounds_naive.plan_mounds_naive(room, mounds_number)
        else:
            final_mounds = mounds

        self.room_mechanics = room_mechanics.RoomMechanics(room=room, mounds=final_mounds)

        return utility.preprocess(self.room_mechanics.room, final_mounds)