"""

This approach was discarded. 

Inspired by:
    https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch/tree/master
    Viet Nguyen <nhviet1009@gmail.com>
"""

import configparser
import argparse
import numpy as np

import torch

from src import clean_agent_dql
from src import plan_mounds_naive
from src import room_generation
from src import room_mechanics
from src import utility

def get_args():

    config = configparser.ConfigParser()
    config.read("config.cfg")

    possible_sizes = eval(config['GENERAL']['possible_sizes'])

    default_room_size = possible_sizes[0]
    default_mounds_number = int(config['GENERAL']['default_mounds_number'])

    parser = argparse.ArgumentParser("Train the cleaning agent")

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)

    parser.add_argument("--room_size",     type=int, default=default_room_size, choices=possible_sizes, help="The common width and height for all images")
    parser.add_argument("--mounds_number", type=int, default=default_mounds_number,                     help="Amount of mounds")
    
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_moves",    type=int, default=100, help="How many times does the algorithm generate a move") # TODO: check default 
    parser.add_argument("--batch_size",   type=int, default=32, help="The number of images per batch the agent replays.")
    parser.add_argument("--replay_memory_size", type=int, default=2000, help="Number of epoches between testing phases")

    parser.add_argument("--punish_clipping", type=float, default=0.5, help="How many percent smaller will the reward be if a movement is clipped?") # TODO: check default 

    parser.add_argument("--show_snapshots", type=str, default=True, help="Whether to show visualisations of movements in each episode every 10 moves")

    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_models_path", type=str, default="trained models/trained cleaning models")

    args = parser.parse_args()
    
    return args


def train(opts):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    agent = clean_agent_dql.DQLCleanAgent(opts.room_size, opts.replay_memory_size)
    print("Agent initilized.")

    room_generator = room_generation.RoomGenerator(opts.room_size)
    print("Room generator initilized.")

    # # room layouts
    # init_states = []
    # for _ in range(opts.num_episodes):

    #     which_method = np.random.randint(4)
    #     subroom_size = np.random.choice( room_generator.subroom_sizes )

    #     new_init_state = None
    #     if which_method == 0:
    #         new_init_state = room_generator.average_pooling_method(subroom_size)
    #     elif which_method == 1:
    #         new_init_state = room_generator.simplex_method(subroom_size)
    #     elif which_method == 2:
    #         new_init_state = room_generator.simplex_method(subroom_size) # TODO: change
    #     else: 
    #         new_init_state = room_generator.simplex_method(subroom_size) # TODO: change

    #     init_states.append(new_init_state)

    init_states = [np.array(room_generation.generate_room_method4(opts.room_size))]

    for num_episode, room in enumerate(init_states):
        
        mounds = plan_mounds_naive.plan_mounds_naive(room, opts.mounds_number)
        env = room_mechanics.RoomMechanics(room=room, mounds=mounds)

        # start/end positions 
        broom = np.random.randint(opts.room_size, size = 4)

        state = utility.preprocess(env.room, env.mounds, broom = broom)

        amount_of_dirt_before = utility.amount_of_dirt(room, mounds)
        
        # cleaned with each movement
        dirt_amounts = []

        for move_index in range(opts.num_moves):

            action = agent.act(state)

            reward = 0

            if action == 8: 
                # if broom was moved
                x1, y1, x2, y2 = broom

                cleaned_dirt, spillover_dirt, clipped = env.move_broom((x1, y1), (x2, y2))
                dirt_amounts.append(cleaned_dirt)

                reward = cleaned_dirt if not clipped else (1 + cleaned_dirt) * (1 - opts.punish_clipping) 

            else:
                # if broom endpoints have moved
                new_broom = utility.update_broom(opts.room_size, action, broom)
                broom = new_broom

            next_state = utility.preprocess(env.room, env.mounds, broom = broom)

            agent.memorize(state, action, reward, next_state)
            state = next_state

            if len(agent.memory) > opts.batch_size:
                agent.replay(opts.batch_size)

            print("\tmove: {}/{}".format(move_index+1, opts.num_moves))

            # every 10% of an episode
            if opts.show_snapshots and (move_index % (0.5 * opts.num_moves) == 0):
                
                if action == 8:
                    env.show_room(colored_segments=[('g', broom)])
                else:
                    # old broom vs new broom
                    env.show_room(colored_segments=[ ('r--', broom),
                                                     ('r-',  utility.update_broom(opts.room_size, action, broom)) ])

        # TODO: update this metric
        # tells what percent of dirt has been cleaned
        score = 100 * np.sum(dirt_amounts) / amount_of_dirt_before 

        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(num_episode, opts.num_episodes, score, agent.epsilon))
        agent.save("./trained models/trained cleaning models/roomsize{}.h5".format(opts.room_size))
              

if __name__ == '__main__':
    opts = get_args()
    train(opts)