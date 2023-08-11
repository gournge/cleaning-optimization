import tensorflow as tf
import keras
from keras.optimizers import Adam
from src.cleaning_agents.ddpg.ddpg_memory import ReplayBuffer
from src.cleaning_agents.ddpg.ddpg_networks import ActorNetwork, CriticNetwork

from numpy import random

import os

class Agent:
    def __init__(self, input_dims, n_actions, min_action = 0, max_action = 1, alpha=0.001, beta=0.002,
                 gamma=0.99, tau=0.005, max_size=1000000,
                batch_size=64, noise=0.1, prob_no_noise=0.2, loaded = True):
        
        self.noise = noise
        self.max_action = max_action
        self.min_action = min_action
        self.input_dims = input_dims

        self.gamma = gamma
        self.tau = tau
        self.prob_no_noise = prob_no_noise
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor         = ActorNetwork (input_dims, n_actions, name='actor')
        self.critic        = CriticNetwork(input_dims, n_actions, name='critic')
        self.target_actor  = ActorNetwork (input_dims, n_actions, name='target_actor')
        self.target_critic = CriticNetwork(input_dims, n_actions, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

        if loaded:
            for model in [self.actor, self.target_actor]:
                model.build((1,) + input_dims)
                model.summary()

            for model in [self.critic, self.target_critic]:
                model( tf.random.uniform((1,) + input_dims), tf.random.uniform((1, n_actions)) )
                model.summary()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)

    def choose_action(self, observation):
        
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        if random.random() > self.prob_no_noise:

            noise = tf.random.normal(shape=[self.n_actions],
                                    mean=0.0, stddev=self.noise)

            activation = self.actor.dense_2.activation

            if activation == 'tanh':
                actions += noise
                actions = (1 + actions) / 2

            elif activation == 'relu':
                actions += noise

            elif activation == 'sigmoid':
                actions += noise

        actions *= self.max_action
        actions = tf.clip_by_value(actions, self.min_action, self.max_action - 1)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

    def save_models(self, dir_name):
        print('... saving models ...')
        self.actor.save_weights(os.path.join(dir_name, self.actor.model_name + "_room" + str(self.input_dims[0]) +  '_ddpg.h5'))
        self.target_actor.save_weights(os.path.join(dir_name, self.target_actor.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.critic.save_weights(os.path.join(dir_name, self.critic.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.target_critic.save_weights(os.path.join(dir_name, self.target_critic.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))

        # self.memory.save(dir_name)

    def load_models(self, dir_name):
        print('... loading models ...')

        self.actor.load_weights(os.path.join(dir_name, self.actor.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.target_actor.load_weights(os.path.join(dir_name, self.target_actor.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.critic.load_weights(os.path.join(dir_name, self.critic.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.target_critic.load_weights(os.path.join(dir_name, self.target_critic.model_name + "_room" + str(self.input_dims[0]) + '_ddpg.h5'))
        self.update_network_parameters()

        # self.memory.load(dir_name)
