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
        # parent_dir = Path(__file__).parent.absolute() 
        # path_to_config = Path.joinpath(parent_dir, "config.cfg")
        # print(path_to_config)

        # config.read(path_to_config)
        config.read(r"C:\Users\filip\OneDrive\Dokumenty\Pulpit\Filip - Laptop\cleaning-optimization\config.cfg")
        # config.read(r"..\..\config.cfg")

        print(config.sections())

        print( len(config.items()) )

        memory = int(config['AGENT']['memory']) 

        if grid_size not in [32, 64, 128]: 
            raise NotImplementedError("No corresponding model architecture has been implemented")

        self.grid_size = grid_size

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

        if self.grid_size == 32:
            
            # `kernel` is a single little window that traverses through the 2d image
            # since 2d image has 3 channels, for each channel we would have a different kernel
            # `filter` is the vector of these kernels collectively

            # filters increasing
            # kernel decreasing

            model.add(layers.Conv2D(filters=3,  kernel_size=24, activation='relu', input_shape = self.state_size))
            model.add(layers.Conv2D(filters=24, kernel_size=12, activation='relu'))
            model.add(layers.Conv2D(filters=48, kernel_size=8,  activation='relu'))

            print(model.output_shape)

            # decreasing

            model.add(layers.Dense(units=128, activation='relu'))
            model.add(layers.Dense(units=4, activation='relu'))

        elif self.grid_size == 64:
            raise NotImplementedError("No corresponding model architecture has been implemented")
        elif self.grid_size == 128:
            raise NotImplementedError("No corresponding model architecture has been implemented")

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose = 0)
        return np.argmax(act_values[0])  # returns action

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