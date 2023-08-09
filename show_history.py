import numpy as np
import argparse
import configparser

import matplotlib.pyplot as plt

def get_args():

    config = configparser.ConfigParser()
    config.read("config.cfg")

    parser = argparse.ArgumentParser("Show history")

    parser.add_argument("--model_path", type=str, help="How many episodes need to pass in order for the snapshot to appear after a full episode.\nSet to -1 if you don\'t want to see snapshots.")

    args = parser.parse_args()
    
    return args

def plot(opts):

    values = []

    import os 
    os.listdir(opts.model_path)

    with open(opts.model_path + '/history.txt', 'r') as file:
        for line in file:
            value = float(line.strip())
            values.append(value)

    plt.plot(values, color='b')
    plt.hlines(np.average(values), 0, len(values), color='r')
    plt.show()

if __name__ == '__main__':

    args = get_args()
    plot(args)