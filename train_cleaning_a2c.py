import configparser
import argparse
import numpy as np

from src import environment
from src import clean_agent_a2c
from src import room_mechanics

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

    parser.add_argument("--show_snapshots", type=bool, default=True, help="Whether to show visualisations of movements in each episode every 10 moves")
    parser.add_argument("--between_snapshots", type=int, default=10, help="How many movements need to pass in order for the snapshot to appear during one episode. Valid only if --show_snapshots is set to True.")

    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_models_path", type=str, default="trained models/trained cleaning models")

    args = parser.parse_args()
    
    return args


def train(opts):

    env = environment.CleaningEnv(opts.room_size, punish_clipping=opts.punish_clipping, mounds_number=opts.mounds_number)
    print("Environment initialized.")    
        
    agent = clean_agent_a2c.A2CCleanAgent(opts.room_size, stand_dev='computed')
    print("Agent initialized.")    

    state_size = agent.state_size
    action_size = agent.action_size

    scores, episodes = [], []

    for e in range(opts.num_episodes):
        score = 0
        state = env.reset()
        # print(state)
        # state = np.reshape(state[0], [1, state_size])

        for m in range(opts.num_moves):

            if opts.show_snapshots and (m % opts.num_moves == 0):
                env.render()

            action = agent.get_action(state)

            reward, next_state = env.act(action)
            # next_state = np.reshape(next_state, [1, state_size])

            agent.train_model(state, action, reward, next_state)

            score += reward
            state = next_state

        if e % 50 == 0:
            agent.save()



if __name__ == '__main__':
    opts = get_args()
    train(opts)