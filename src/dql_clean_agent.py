"""
Based on:
    https://github.com/keon/deep-q-learning
"""

import numpy as np
import random 
from collections import deque

import configparser
from pathlib import Path # for path for configparser

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
import keras.layers as layers

class DQLCleanAgent:

    def __init__(self, grid_size: int):
        """
        Args:
        ----
        - `int` grid size        
        """

        config = configparser.ConfigParser()
        parent_dir = Path(__file__).parent.parent.absolute() 
        path_to_config = Path.joinpath(parent_dir, "config.cfg")
        config.read(path_to_config)

        memory = int(config['AGENT']['memory']) 
        possible_sizes = eval(config['GENERAL']['possible_sizes'])

        if grid_size not in possible_sizes: 
            raise NotImplementedError("No corresponding model architecture has been implemented")

        self.grid_size = grid_size

        # batch size
        self.state_shape = (grid_size, grid_size, 3)
        self.action_size = 4
        self.memory = deque(maxlen = memory)

        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        
        # TODO: convolutional nn

        model = Sequential()

        if self.grid_size == 35:
            
            # `kernel` is a single little window that traverses through the 2d image
            # since 2d image has 3 channels, for each channel we would have a different kernel
            # `filter` is the vector of these kernels collectively

            # filters increasing
            # kernel decreasing

            # kernel size chosen based on width of subrooms 

            # since room size at grid is either 6 or 7 it seems a good idea to make a kernel inclusive of two neighboring subrooms
            model.add(layers.Conv2D(filters=3,  kernel_size=15, strides=(2, 2), activation='relu', input_shape = self.state_shape))
            
            model.add(layers.Conv2D(filters=24, kernel_size=5,  activation='relu'))
            model.add(layers.Conv2D(filters=36, kernel_size=3,  activation='relu'))

            # ??? instead of flatten ?
            # see https://stats.stackexchange.com/a/194450 
            # model.add(layers.Conv2D(filters=1, kernel_size=1, activation='relu'))

            model.add(layers.Flatten())

            # input is of size (None, 1200)
            model.add(layers.Dense(units=200, activation='relu'))
            model.add(layers.Dense(units=4))
            model.add(layers.ReLU(max_value=1.0)) # TODO: consider changing to room dimensions

        elif self.grid_size == 71:
            raise NotImplementedError("No corresponding model architecture has been implemented")
        elif self.grid_size == 119:
            raise NotImplementedError("No corresponding model architecture has been implemented")

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("random")
            return np.random.rand(self.action_size)
        return self.model.predict(state, verbose = 0)

    def replay(self, batch_size):
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state in minibatch:
        #     target = reward

        #     target = (reward + self.gamma *
        #                 np.amax(self.model.predict(next_state, verbose=0)[0]))
        #     target_f = self.model.predict(state, verbose=0)
        #     target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        pass

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)