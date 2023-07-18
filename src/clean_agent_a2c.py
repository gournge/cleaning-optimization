"""

Based on:
    https://github.com/rlcode/reinforcement-learning 

"""

import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
import keras.layers as layers

class A2CCleanAgent:
    def __init__(self, 
                 room_size, 
                 stand_dev = None,
                 stand_dev_constant = 0.4, 
                 stand_dev_gradually_annealed = (1.0, 0.95, 0.001)
                 ):
        """Agent making decisions based on the Actor-Critic model.

        Args:
        ----
        - `room_size`
        - `stand_dev` must be either
            - `'constant'`
            - `'computed' `
            - `'gradually annealed'`

        - `stand_dev_gradually_annealed` should be a tuple of form `(beg, mult, end)`. 
            After every broom movement we multiply current standard deviation by `mult`, 
            beginning with `beg`, until it reaches `end`.
        
        """

        if stand_dev is None:
            raise KeyError('You must specify the method of calculating Agent\'s actions\' standard deviations.')

        if stand_dev not in ['constant', 'computed', 'gradually annealed']:
            raise KeyError('There is no such method of calculating Agent\'s actions\' standard deviations.')

        self.stand_dev = stand_dev
        self.stand_dev_constant = stand_dev_constant
        self.stand_dev_gradually_annealed = stand_dev_gradually_annealed
        self.stand_dev_current = self.stand_dev_gradually_annealed[0]

        self.room_size = room_size

        # get size of state and action
        self.state_size = (room_size, room_size, 3)
        self.action_size = 4
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):

        actor = Sequential(name='actor_model')

        if self.room_size == 35:
            
            # `kernel` is a single little window that traverses through the 2d image
            # since 2d image has 3 channels, for each channel we would have a different kernel
            # `filter` is the vector of these kernels collectively

            # filters increasing
            # kernel decreasing

            # kernel size chosen based on width of subrooms 

            # since room size at grid is either 6 or 7 it seems a good idea to make a kernel inclusive of two neighboring subrooms
            actor.add(layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation='relu', input_shape = self.state_size))
            
            actor.add(layers.Conv2D(filters=24, kernel_size=5,  activation='relu'))
            actor.add(layers.Conv2D(filters=36, kernel_size=3,  activation='relu'))

            # ??? instead of flatten ?
            # see https://stats.stackexchange.com/a/194450 
            # actor.add(layers.Conv2D(filters=1, kernel_size=1, activation='relu'))

            actor.add(layers.Flatten())

            # input is of size (None, 1200)
            actor.add(layers.Dense(units=200, activation='relu'))

        elif self.room_size == 71:
            raise NotImplementedError("No corresponding model architecture has been implemented")
        elif self.room_size == 119:
            raise NotImplementedError("No corresponding model architecture has been implemented")

        final_layer_units = 2 * self.action_size if self.stand_dev == 'computed' else self.action_size

        # needs to be softmax since the output represents probabilities 
        actor.add(layers.Dense(units=final_layer_units, activation='softmax'))

        actor.summary()

        actor.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):

        critic = Sequential(name='critic_model')

        if self.room_size == 35:
            critic.add(layers.Conv2D(filters=4,  kernel_size=15, strides=(2, 2), activation='relu', input_shape = self.state_size))
            
            critic.add(layers.Conv2D(filters=24, kernel_size=5,  activation='relu'))
            critic.add(layers.Conv2D(filters=36, kernel_size=3,  activation='relu'))

            critic.add(layers.Flatten())

            critic.add(layers.Dense(units=200, activation='relu'))

        elif self.room_size == 71:
            raise NotImplementedError("No corresponding model architecture has been implemented")
        elif self.room_size == 119:
            raise NotImplementedError("No corresponding model architecture has been implemented")

        final_layer_units = 2 * self.action_size if self.stand_dev == 'computed' else self.action_size
        critic.add(layers.Dense(units=final_layer_units, activation='softmax'))

        critic.summary()

        critic.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()

        mus = policy[:4]
        sigmas = []
        if self.stand_dev == 'computed':
            sigmas = policy[4:]
        elif self.stand_dev == 'constant':
            sigmas = [self.stand_dev_constant] * 4
        elif self.stand_dev == 'gradually annealed':
            beg, mult, end = self.stand_dev_gradually_annealed
            sigmas = np.clip( [self.stand_dev_current * mult] * 4, end, beg)

        print(policy, mus, sigmas)

        raw_sampled_actions = np.random.normal()

        # return raw_sampled 


    # update policy network every episode
    def train_model(self, state, action, reward, next_state):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        advantages[0][action] = reward + self.discount_factor * (next_value) - value
        target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def load(self):
        self.actor.load_weights("./trained models/trained cleaning models/a2c/actor{}.h5".format(self.room_size))
        self.critic.load_weights("./trained models/trained cleaning models/a2c/critic{}.h5".format(self.room_size))

    def save(self):
        self.actor.save_weights("./trained models/trained cleaning models/a2c/actor{}.h5".format(self.room_size))
        self.critic.save_weights("./trained models/trained cleaning models/a2c/critic{}.h5".format(self.room_size))
