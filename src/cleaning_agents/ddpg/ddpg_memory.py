import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, new_state):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, states_
    
    def save(self, dir_name):
        np.savez(dir_name + '/memory.npz',
                 self.mem_size, 
                 self.mem_cntr, 
                 self.state_memory,
                 self.new_state_memory,
                 self.action_memory,
                 self.reward_memory)
        
    def load(self, dir_name):
        data = np.load(dir_name + '/memory.npz')

        self.mem_size         = data['mem_size']
        self.mem_cntr         = data['mem_cntr']
        self.state_memory     = data['state_memory']
        self.new_state_memory = data['new_state_memory']
        self.action_memory    = data['action_memory']
        self.reward_memory    = data['reward_memory']
    