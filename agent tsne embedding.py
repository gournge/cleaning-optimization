import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.cleaning_agents.ddpg.clean_agent_ddpg import Agent
from src.environment import CleaningEnv

def get_opts():

    parser = argparse.ArgumentParser("Train the cleaning agent")

    parser.add_argument("--num_data_points_expected", type=int, required=True)

    parser.add_argument("--model_path", type=str, required=True, help="Model for analysis.")
    
    parser.add_argument("--room_size", type=int, default=35)
    parser.add_argument("--punish_clipping", type=float, default=2)
    parser.add_argument("--mounds_number", type=int, default=8)
    
    parser.add_argument("--num_episodes", type=int, default=15)

    args = parser.parse_args()
    
    return args

def main(opts):

    env = CleaningEnv(opts.room_size, opts.punish_clipping, mounds_number=opts.mounds_number)
    agent = Agent(input_dims=(opts.room_size, opts.room_size, 3), n_actions=4, loaded=True,
                  min_action=0, max_action=opts.room_size, noise=0.00001)

    agent.load_models(opts.model_path)
    print("Weights loaded successfuly.")

    observations = []
    prev_layer_outputs = []
    rewards = []

    print('collecting data . . .')
    while len(observations) < opts.num_data_points_expected:

        obs = env.reset()
        for _ in range(opts.num_episodes):
            # print('\tnew episode!')

            action, prev_layer_output = agent.choose_action(obs, prev_layer_output=True)
            reward, new_obs = env.act(action)
            
            if np.random.random() < 0.5:
                observations.append(obs)
                prev_layer_outputs.append(prev_layer_output)
                rewards.append(reward)
                # print('\tgot one data point')

            obs = new_obs

        if len(observations) % 100 == 0:
            print("Collected", len(observations), "data points.")

    num_data_points = len(prev_layer_outputs)
    flat_states = np.array(prev_layer_outputs).reshape(num_data_points, -1)
    print(len(flat_states[0]))

    tsne = TSNE(n_components=2, perplexity=30)
    tsne_embeddings = tsne.fit_transform(flat_states)
    print('tsne embeddings:')
    print(tsne_embeddings)

    x_s = [point[0] for point in tsne_embeddings]
    y_s = [point[1] for point in tsne_embeddings]
    rewards = [ (r - min(rewards)) / (max(rewards) - min(rewards)) for r in rewards]
    # print(len(x_s), len(y_s), len(rewards))
    scatter = plt.scatter(x_s, y_s, c = rewards, cmap='plasma', s=100)
    plt.colorbar(scatter)
    plt.show()



if __name__ == '__main__':
    opts = get_opts()
    main(opts)