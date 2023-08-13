from src import room_mechanics
from src import room_generation
from src import plan_mounds_random
from src import utility

import numpy as np

class CleaningEnv:
    """
        State of the environment is a `(1, N, N, 3)` tensor. 
        The depth is not 4 since previous broom movement is already remembered by the agent - there is no need to create a seperate layer in the tensor for it.
    
    """

    def __init__(self, room_size, punish_clipping, mounds_number = None, manual_mounds = None):
        """Create an env of given size. Provide mounds plan yourself or let the program generate it automatically.
        
        
        Args:
        ----
            - `room_size`
            - `punish_clipping` how much smaller should the reward be if movement was clipped (e.g `0.5`)
            - `mounds` optional - list of positions
            - `mounds_number` 
            
        """

        if manual_mounds is None and mounds_number is None: 
            raise KeyError('you have to provide number of mounds to be generated.') 

        self.room_size = room_size
        self.punish_clipping = punish_clipping

        self.room_generator = room_generation.RoomGenerator(room_size)
        
        room = self.room_generator.any_method()

        self.room_mechanics = None
        if not isinstance(manual_mounds, np.ndarray):
            mounds = plan_mounds_random.plan_mounds_random(room, mounds_number)
            self.room_mechanics = room_mechanics.RoomMechanics(room=room, mounds=mounds)
        else:
            self.room_mechanics = room_mechanics.RoomMechanics(room, manual_mounds)

        # needs the room layout (with dirt) and the plan of mounds

        self.previous_actions = []


    def apply_mounds_plan(self, mounds):
        self.room_mechanics = room_mechanics.RoomMechanics(room=self.room, mounds=mounds)

    def render(self, actions = None):

        if actions is None:
            actions = [] if self.previous_actions else list(('r', action) for action in self.previous_actions)

        self.room_mechanics.show_room(colored_segments=actions)

    def act(self, broom):
        """
            Args:
            ----
                `broom` tuple of four floats describing coords of endpoints of the broom.
            
            Returns:
            ------
                - `reward`
                - `Tensor` of size `(room_size, room_size, 3)`
        """

        def too_close(broom1, broom2):
            start_too_close = (abs(broom1[0] - broom2[0]) < 1) and (abs(broom1[1] - broom2[1]) < 1)
            end_too_close   = (abs(broom1[2] - broom2[2]) < 1) and (abs(broom1[3] - broom2[3]) < 1)

            return start_too_close and end_too_close

        x1, y1, x2, y2 = broom

        cleaned_dirt, _, clipped = self.room_mechanics.move_broom((x1, y1), (x2, y2))

        # error might have occured in the env
        reward = (cleaned_dirt - self.punish_clipping * clipped) if cleaned_dirt >= 0 else None

        if self.previous_actions and too_close(broom, self.previous_actions[-1]):
            reward -= self.punish_clipping

        self.previous_actions.append(broom)

        return reward, utility.preprocess(self.room_mechanics.room, self.room_mechanics.mounds)
    
    def reset(self, mounds = None, mounds_number = None):
        """New room layout. Returns the state of the environment. (No broom)
        
        """

        if mounds is None and mounds_number is None:
            mounds_number = len(self.room_mechanics.mounds)

        self.previous_actions = []

        room = self.room_generator.any_method()

        final_mounds = None
        if mounds is None:
            final_mounds = plan_mounds_random.plan_mounds_random(room, mounds_number)
        else:
            final_mounds = mounds

        self.room_mechanics = room_mechanics.RoomMechanics(room=room, mounds=final_mounds)

        return utility.preprocess(self.room_mechanics.room, final_mounds)