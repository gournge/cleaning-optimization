import numpy as np
import argparse
import configparser

import os

from src.environment import CleaningEnv
from src.cleaning_agents.ddpg.clean_agent_ddpg import Agent


def get_args():

    config = configparser.ConfigParser()
    config.read("config.cfg")

    possible_sizes = eval(config['GENERAL']['possible_sizes'])

    default_room_size = possible_sizes[0]
    default_mounds_number = int(config['GENERAL']['default_mounds_number'])

    parser = argparse.ArgumentParser("Train the cleaning agent")

    parser.add_argument("--load_previous_models", type=str, default='True', choices=['True', 'False'], help="If there are any model weights inside predefined path, start training with them.")
    parser.add_argument("--load_models_path", type=str, default="trained models/trained cleaning models")
    parser.add_argument("--save_models_path", type=str, default="trained models/trained cleaning models")

    parser.add_argument("--actor_learning_rate", type=float, default=0.001)
    parser.add_argument("--critic_learning_rate", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--noise", type=float, default=0.1)

    parser.add_argument("--room_size",     type=int, default=default_room_size, choices=possible_sizes, help="The common width and height for all images")
    parser.add_argument("--mounds_number", type=int, default=default_mounds_number,                     help="Amount of mounds")
    
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_moves",    type=int, default=100, help="How many times does the algorithm generate a move")
    parser.add_argument("--batch_size",   type=int, default=32, help="The number of images per batch the agent replays.")
    parser.add_argument("--replay_memory_size", type=int, default=2000, help="Number of epoches between testing phases")

    parser.add_argument("--punish_clipping", type=float, default=5.0, help="How much do we subtract from a reward if a movement is clipped?")

    parser.add_argument("--between_snapshots", type=int, default=10, help="How many episodes need to pass in order for the snapshot to appear after a full episode.\nSet to -1 if you don\'t want to see snapshots.")

    args = parser.parse_args()
    
    return args

def train(opts):

    env = CleaningEnv(opts.room_size, opts.punish_clipping, mounds_number=opts.mounds_number)
    print("Environment initialized")

    agent = Agent( 
                   input_dims=(opts.room_size, opts.room_size, 3), n_actions=4,
                   min_action=0, max_action=opts.room_size,
                   max_size=opts.replay_memory_size,
                   tau   = opts.tau,
                   gamma = opts.gamma,
                   alpha = opts.actor_learning_rate,
                   beta  = opts.critic_learning_rate,
                   noise = opts.noise,
                   loaded = opts.load_previous_models
                 )
    print("Agent initialized")

    if opts.load_previous_models == 'True':
        load_models_absolute_path = os.path.abspath(opts.load_models_path)
        agent.load_models(load_models_absolute_path)
        print("Weights loaded successfuly.")

    score_history = []

    for e in range(opts.num_episodes):

        moves = []

        observation = env.reset()
        score = 0
        for n_move in range(opts.num_moves):
            action = agent.choose_action(observation)

            # print("At move", n_move, "agent has chosen", action)
            moves.append(action)

            reward, observation_ = env.act(action)
            
            if reward == 0: continue

            score += reward
            agent.remember(observation, action, reward, observation_)
            agent.learn()
            observation = observation_

            # print('\t', n_move, reward)

        score_history.append(score)
        avg_score = np.mean(score_history)

        if (opts.between_snapshots != -1) and (e % opts.between_snapshots == 0):
            env.render(actions = moves)

        if score > avg_score:
            saved_models_absolute_path = os.path.abspath(opts.save_models_path)
            agent.save_models(saved_models_absolute_path)

        print('episode ', e, 'score %.1f' % score, 'avg score %.1f' % avg_score)


if __name__ == '__main__':
    args = get_args()
    train(args)