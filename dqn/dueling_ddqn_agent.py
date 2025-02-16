import numpy as np
import torch as T
from dqn.dueling_net import DuelingDQNetwork
from dqn.replay_memory import MemoryBuffer
from  GPUtil import getAvailable
import os
import matplotlib.pyplot as plt
from utils.debug import debug_save_any_img

class DuelingDDQNAgent():
    def __init__(self, gamma, lr, n_actions, input_dims, mem_size, batch_size, epsilon_decay_cycle_length=1000, Tmax=1.0, Tmin=0.1, omega=1.0,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn', training_stats_path='tmp/dqn_stats'):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.training_stats_path = training_stats_path
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.total_steps = 0
        available_gpus = getAvailable(order='memory', limit=1)  # Get the best GPU by memory
        if available_gpus:
            self.device = T.device(f'cuda:{available_gpus[0]}')
        else:
            self.device = T.device('cpu')  # Default to CPU if no GPUs are available


        self.memory = MemoryBuffer(mem_size, input_dims)

        self.q_eval = DuelingDQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_eval', chkpt_dir=self.chkpt_dir, device = self.device)

        self.q_next = DuelingDQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_next', chkpt_dir=self.chkpt_dir, device = self.device)
    
        self.learning_curve = []
        
        self.cycle_length = epsilon_decay_cycle_length # Length of the epsilon decay cycle
        self.T = Tmax  # Initial temperature for the softmax distribution
        self.Tmin = Tmin # Minimum temperature for the softmax distribution
        self.Tmax = Tmax  # Maximum temperature for the softmax distribution
        self.omega = omega  # Parameter for the Mellowmax function

    def update_temperature(self):
        """Cyclically updates the temperature using a cosine function."""
        self.T = self.Tmin + (self.Tmax - self.Tmin) * (1 + np.cos(2 * np.pi * self.learn_step_counter / self.cycle_length)) / 2


    def mellowmax(self, q_values):
        """
        Applies the Mellowmax function to Q-values.
        :param q_values: List of action-value estimates (Q-values)
        :return: Probabilities for action selection
        """
        max_Q = np.max(q_values)  # Stabilize exponentials
        mellow_value = np.log(np.mean(np.exp(self.omega * (q_values - max_Q)))) / self.omega + max_Q
        return mellow_value


    def choose_action(self, observation, evaluate=False):
        """
        Select an action based on Mellowmax Exploration with Cyclic Temperature Decay.
        
        :param observation: The current state observation.
        :param evaluate: If True, selects the best action (exploitation).
        :return: The chosen action.
        """
        observation_array = np.array(observation)
        if observation_array.ndim == 3:
            observation_array = np.array([observation_array])
        state = T.tensor(observation_array, dtype=T.float).to(self.q_eval.device)
        
        self.q_eval.eval()
        _, advantage = self.q_eval.forward(state)
        
        if evaluate:
            # Exploitation: Choose the best action
            action = T.argmax(advantage, dim=-1).item()
        else:
            # Convert Q-values into Mellowmax probability distribution
            q_values = advantage.detach().cpu().numpy().squeeze()  # Convert (1, num_actions) → (num_actions)
            mellow_val = self.mellowmax(q_values)
            probabilities = np.exp(self.omega * (q_values - mellow_val))  # Soft assignment

            # Normalize probabilities
            probabilities /= np.sum(probabilities)

            # Sample an action
            action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def add_to_learning_curve(self, loss):

        self.learning_curve.append({
            'loss': loss,
            'epsilon': self.T,
            'learn_step_counter': self.learn_step_counter,
            'total_steps': self.total_steps
        })

        # Save it to a file
        self.save_learning_curve()
            
    def save_learning_curve(self):
        if (self.learn_step_counter % 1000 == 0):
            np.save(os.path.join(self.training_stats_path, self.env_name + '_learning_curve.npy'), self.learning_curve, allow_pickle=True)

    def learn(self):
        self.total_steps += 1 
        # Check if there are enough experiences in memory to sample a batch for training
        if self.memory.mem_cntr < self.batch_size:
            return  # Exit if not enough samples
        self.q_eval.train()
        self.q_next.train()
        # Reset the gradients of the optimizer to zero
        self.q_eval.optimizer.zero_grad()

        # Update target network parameters periodically
        # self.learn_step_counter % self.replace_target_cnt == 0 where replace_target_cnt == replace hyperparameter
        self.replace_target_network()

        # Sample a batch of transitions (state, action, reward, next state, done flag) from memory
        states, actions, rewards, states_, dones = self.sample_memory()

        # Generate a range of indices for batch processing
        indices = np.arange(self.batch_size)

        # Compute the value (V_s) and advantage (A_s) streams from the main Q-network for the current states 
        V_s, A_s = self.q_eval.forward(states)
        
        # Compute the value (V_s_) and advantage (A_s_) streams from the target Q-network for the next states
        V_s_, A_s_ = self.q_next.forward(states_)

        # Compute the value (V_s_eval) and advantage (A_s_eval) streams from the main Q-network for the current states (used for action selection)
        V_s_eval, A_s_eval = self.q_eval.forward(states)

        # Calculate the predicted Q-values for the actions taken (current Q-values) using dueling architecture
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        # Calculate the target Q-values (next Q-values) for the next states using the target network
        # Apply the dueling architecture, adjusting for mean advantage, and avoid max across actions for stability
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        # Calculate the Q-values of the current states for selecting max actions using the main Q-network
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        # Determine the actions with the highest Q-values from the main Q-network for the Double DQN update
        max_actions = T.argmax(q_eval, dim=1)

        # Zero out Q-values for terminal states to ensure no future reward is accumulated after episode end
        q_next[dones] = 0.0

        # Calculate the target Q-values for each action (using Double DQN formula)
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # Calculate the loss between the target and predicted Q-values
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        # Backpropagate the loss to update the network weights
        loss.backward()
        self.q_eval.optimizer.step()

        # Increment the learning step counter
        self.learn_step_counter += 1
        self.update_temperature()  # Update temperature before selecting action

        # add the avg loss to the learning curve
        self.add_to_learning_curve(T.mean(loss).item())
