import numpy as np
import argparse
import configparser

import matplotlib.pyplot as plt

def get_args():

    config = configparser.ConfigParser()
    config.read("config.cfg")

    parser = argparse.ArgumentParser("Show history")

    parser.add_argument("--model_path", type=str, required=True, help="How many episodes need to pass in order for the snapshot to appear after a full episode.\nSet to -1 if you don\'t want to see snapshots.")

    parser.add_argument("--window_size", type=int, required=True, help="Set to -1 if you dont want to see it")
    parser.add_argument("--font_size", type=int, required=True)

    args = parser.parse_args()
    
    return args

def plot(opts):

    values = []

    with open(opts.model_path + '/history.txt', 'r') as file:
        for line in file:
            value = float(line.strip())
            values.append(value)

    values = values[:702]

    episodes = np.arange(len(values))

    plt.rcParams.update({'font.size': opts.font_size})
    fig, ax1 = plt.subplots()

    ax1.plot(episodes, values, color='b', label='Scores')
    ax1.set_xlabel('Episode number')
    ax1.set_ylabel('Score')

    lines, labels = ax1.get_legend_handles_labels()

    if opts.window_size != -1:
        

        moving_var = np.convolve(values, np.ones(opts.window_size)/opts.window_size, mode='same')
        moving_var = np.convolve((values - moving_var)**2, np.ones(opts.window_size)/opts.window_size, mode='same')
        # moving_var = np.sqrt(moving_var)

        crop = opts.window_size // 2

        cropped_episodes   =   episodes[crop:-crop]
        cropped_moving_var = moving_var[crop:-crop]

        ax2 = ax1.twinx()
        ax2.set_ylabel(f'Moving variance of window size {opts.window_size}')
        ax2.plot(cropped_episodes, cropped_moving_var, color='r', label = 'Moving variance')

        lines2, labels2 = ax2.get_legend_handles_labels()

        lines += lines2
        labels += labels2

    ax1.legend(lines, labels, loc='upper left')

    plt.title('Score history of a training session')
    plt.show()

if __name__ == '__main__':

    args = get_args()
    plot(args)