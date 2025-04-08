import os
import numpy as np
import torch as T
import torch.nn.functional as F
from ddpg.networks import ActorNetwork, CriticNetwork
from ddpg.noise import OUActionNoise
from ddpg.buffer import ReplayBuffer
from  GPUtil import getAvailable

class DDPGAgent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, algo=None, env_name=None, chkpt_dir='tmp/dqn', training_stats_path='tmp/dqn_stats'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.training_stats_path = training_stats_path
        self.algo = algo
        self.env_name = env_name
        available_gpus = getAvailable(order='memory', limit=1)  # Get the best GPU by memory
        if available_gpus:
            self.device = T.device(f'cuda:{available_gpus[0]}')
        else:
            self.device = T.device('cpu')  # Default to CPU if no GPUs are available


        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  chkpt_dir=chkpt_dir,
                                n_actions=n_actions, name='actor', device=self.device)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                chkpt_dir=chkpt_dir,
                                n_actions=n_actions, name='critic', device=self.device)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         chkpt_dir=chkpt_dir,
                                n_actions=n_actions, name='target_actor', device=self.device)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                           chkpt_dir=chkpt_dir,
                                n_actions=n_actions, name='target_critic', device=self.device)
        self.learn_step_counter = 0
        self.learning_curve = []
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = (mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)) if not evaluate else mu
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        
        self.add_to_learning_curve(T.mean(critic_loss).item(), T.mean(actor_loss).item())

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)





    def add_to_learning_curve(self, critic_loss, actor_loss):
        """ Store loss, epsilon, and steps with downsampling and efficient saving """
        self.learn_step_counter += 1

        # Append new loss to temporary buffer
        self.learning_curve.append({
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'epsilon': 0,
            'learn_step_counter': self.learn_step_counter,
        })

        # Downsampling: Every 100 steps, store the averaged entry
        if len(self.learning_curve) >= 100:
            avg_critic_loss = np.mean([entry['critic_loss'] for entry in self.learning_curve])
            avg_actor_loss = np.mean([entry['actor_loss'] for entry in self.learning_curve])
            avg_epsilon = np.mean([entry['epsilon'] for entry in self.learning_curve])
            avg_step = self.learning_curve[-1]['learn_step_counter']

            avg_entry = {
                'critic_loss': avg_critic_loss,
                'actor_loss': avg_actor_loss,
                'epsilon': avg_epsilon,
                'learn_step_counter': avg_step,
            }

            # Save the averaged entry
            self.save_learning_curve(avg_entry)

            # Clear buffer
            self.learning_curve = []

    def save_learning_curve(self, entry):
        """ Append data to file instead of storing everything in memory """
        file_path = os.path.join(self.training_stats_path, self.env_name + '_learning_curve.npy')

        # If file exists, load existing data, append new entry, and save
        if os.path.exists(file_path):
            existing_data = list(np.load(file_path, allow_pickle=True))
        else:
            existing_data = []

        existing_data.append(entry)

        # Save the updated data
        np.save(file_path, existing_data, allow_pickle=True)

