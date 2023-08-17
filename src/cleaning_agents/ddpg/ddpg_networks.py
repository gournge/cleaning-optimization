import os
import tensorflow as tf
import keras.layers as layers
from keras import Model


class CriticNetwork(Model):
    def __init__(self, state_shape, n_actions, activation='tanh', name='critic'):
        super(CriticNetwork, self).__init__()

        self.model_name = name

        self.activation = activation

        self.conv2d_1 = None
        self.conv2d_2 = None 
        self.conv2d_3 = None 
        self.flatten = None 
        self.dense_1 = None
        self.dense_2 = None

        if state_shape[0] == 35:

            self.conv2d_1 = layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation=self.activation, input_shape = state_shape)
            self.conv2d_2 = layers.Conv2D(filters=24, kernel_size=5,  activation=self.activation)
            self.conv2d_3 = layers.Conv2D(filters=36, kernel_size=3,  activation=self.activation)

            self.flatten = layers.Flatten()

            self.dense_1 = layers.Dense(units=200 + n_actions)
            self.dense_2 = layers.Dense(units=1, activation=self.activation)

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
    def __init__(self, state_shape, n_actions, activation='tanh', name='actor'):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        self.model_name = name

        self.activation = activation

        self.conv2d_1 = None
        self.conv2d_2 = None 
        self.conv2d_3 = None 
        self.flatten = None 
        self.dense_1 = None
        self.dense_2 = None

        if state_shape[0] == 35:

            self.conv2d_1 = layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation=self.activation, input_shape = state_shape)
            self.conv2d_2 = layers.Conv2D(filters=24, kernel_size=5,  activation=self.activation)
            self.conv2d_3 = layers.Conv2D(filters=36, kernel_size=3,  activation=self.activation)

            self.flatten = layers.Flatten()

            self.dense_1 = layers.Dense(units=200)
            self.dense_2 = layers.Dense(units=n_actions, activation=self.activation)
        
        else:
            raise NotImplementedError("No corresponding model architecture has been implemented")

    def call(self, state, prev_layer_output = False):

        values = self.conv2d_1(state)
        values = self.conv2d_2(values)
        values = self.conv2d_3(values)

        values = self.flatten(values)

        prev_values = None
        values = self.dense_1(values)
        if prev_layer_output:
            prev_values = values
        values = self.dense_2(values)

        if prev_layer_output:
            return values, prev_values

        return values
