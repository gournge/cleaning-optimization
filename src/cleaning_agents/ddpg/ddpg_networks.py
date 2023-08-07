import os
import tensorflow as tf
import keras.layers as layers
from keras import Model


class CriticNetwork(Model):
    def __init__(self, state_shape, n_actions, name='critic'):
        super(CriticNetwork, self).__init__()

        self.model_name = name

        self.conv2d_1 = None
        self.conv2d_2 = None 
        self.conv2d_3 = None 
        self.flatten = None 
        self.dense_1 = None
        self.dense_2 = None

        if state_shape[0] == 35:

            self.conv2d_1 = layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation='relu', input_shape = state_shape)
            self.conv2d_2 = layers.Conv2D(filters=24, kernel_size=5,  activation='relu')
            self.conv2d_3 = layers.Conv2D(filters=36, kernel_size=3,  activation='relu')

            self.flatten = layers.Flatten()

            self.dense_1 = layers.Dense(units=200 + n_actions, activation='relu')
            self.dense_2 = layers.Dense(units=1, activation='relu')

        else:
            raise NotImplementedError("No corresponding model architecture has been implemented")

    def call(self, state, action):

        state_value = self.conv2d_1(state)
        state_value = self.conv2d_2(state_value)
        state_value = self.conv2d_3(state_value)

        state_value = self.flatten(state_value)


        action_value = self.dense_1(tf.concat([state_value, action], axis=1))
        return self.dense_2(action_value)

class ActorNetwork(Model):
    def __init__(self, state_shape, n_actions, name='actor'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        self.model_name = name

        self.conv2d_1 = None
        self.conv2d_2 = None 
        self.conv2d_3 = None 
        self.flatten = None 
        self.dense_1 = None
        self.dense_2 = None

        if state_shape[0] == 35:

            self.conv2d_1 = layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation='relu', input_shape = state_shape)
            self.conv2d_2 = layers.Conv2D(filters=24, kernel_size=5,  activation='relu')
            self.conv2d_3 = layers.Conv2D(filters=36, kernel_size=3,  activation='relu')

            self.flatten = layers.Flatten()

            self.dense_1 = layers.Dense(units=200)
            self.dense_2 = layers.Dense(units=n_actions, activation='relu')
        
        else:
            raise NotImplementedError("No corresponding model architecture has been implemented")

    def call(self, state):

        prob = self.conv2d_1(state)
        prob = self.conv2d_2(prob)
        prob = self.conv2d_3(prob)

        prob = self.flatten(prob)

        prob = self.dense_1(prob)
        prob = self.dense_2(prob)

        return prob
