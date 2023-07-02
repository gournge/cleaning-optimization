"""
Based on:
    https://github.com/keon/deep-q-learning
"""

import numpy as np
import random 
from collections import deque

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:

    def __init__(self, state_size, action_size):

        raise NotImplementedError

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        
        # TODO: convolutional nn

        model = Sequential()

        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))

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