"""
Inspired by:
    https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch/tree/master
    Viet Nguyen <nhviet1009@gmail.com>
"""

import configparser
import argparse
import numpy as np

import torch

from src import dql_clean_agent
from src import plan_mounds_naive
from src import room_generation
from src import room_mechanics
from src import utility

def get_args():

    config = configparser.ConfigParser()
    config.read("config.ini")
    possible_sizes = eval(config['GENERAL']['possible_sizes'])
    print(possible_sizes)

    mounds_number = int(config["GENERAL"]['mounds_number'])
    
    parser = argparse.ArgumentParser("Train the cleaning agent")

    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)

    parser.add_argument("--image_size",   type=int, default=32, choices=eval(possible_sizes), help="The common width and height for all images")
    parser.add_argument("--batch_size",   type=int, default=32, help="The number of images per batch")
    parser.add_argument("--num_episodes", type=int, default=100) # TODO: check default 
    parser.add_argument("--num_moves",    type=int, default=100, help="How many times does the algorithm generate a move") # TODO: check default 

    parser.add_argument("--punish_clipping", type=float, default=0.5, help="How many percent smaller will the reward be if a movement is clipped?") # TODO: check default 

    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained cleaning models")

    args = parser.parse_args()
    
    return args, mounds_number


def train(opts):

    parser, mounds_number = opts

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    agent = dql_clean_agent.DQLCleanAgent(parser.image_size)

    # room layouts
    init_states = list( room_generation.generate_room(parser.image_size) for _ in range(parser.num_episodes) )

    for num_episode, room in enumerate(init_states):
        
        mounds = plan_mounds_naive.plan_mounds_naive(room, mounds_number)
        env = room_mechanics.RoomMechanics(room=room, mounds=mounds)

        state = utility.preprocess(env.room, env.mounds)

        amount_of_dirt_before = utility.amount_of_dirt(room, mounds)
        
        # cleaned with each movement
        dirt_amounts = []

        for move_index in range(opts.num_moves):

            action = agent.act(state)

            x1, y1, x2, y2 = action

            cleaned_dirt, spillover_dirt, clipped = env.move_broom((x1, y1), (x2, y2))

            dirt_amounts.append(cleaned_dirt)

            r = cleaned_dirt if not clipped else cleaned_dirt * opts.punish_clipping 

            next_state, reward = utility.preprocess(env.room, env.mounds), r

            agent.memorize(state, action, reward, next_state)
            state = next_state

            if len(agent.memory) > parser.batch_size:
                agent.replay(parser.batch_size)


        # TODO: update this metric
        score = np.sum(dirt_amounts) / amount_of_dirt_before

        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(num_episode, parser.num_episodes, score, agent.epsilon))
              

if __name__ == '__main__':
    opts = get_args()
    train(opts)