from curses import raw
from datetime import datetime
import os
from turtle import forward
from typing import Any, List, Optional, Tuple
from matplotlib.style import available
from omegaconf import DictConfig
from py import log
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from torch.optim import Adam
from zmq import device
from .utils import soft_update, hard_update
# from train.BaseTrainer import BaseTrainer
from .Networks import ActorNetwork, CriticNetwork, EmbeddedNetwork, MessageEncoder, MessageDecoder
import numpy as np
import re

from .replay_memory import ReplayMemory
from  GPUtil import getAvailable

from dataclasses import dataclass
from typing import Any, Tuple, Optional


class Agent:
    def __init__(self, config: DictConfig):
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.direction_dim = config.direction_dim
        self.feature_dim = config.feature_dim
        self.message_dim = config.message_dim
        self.n_agents = config.n_agents

        self.tau = config.tau
        self.gamma = config.gamma
        self.lr = config.lr
        self.init_alpha = config.alpha
        self.memory_max_size = config.memory_max_size
        self.batch_size = config.batch_size

        self.reconstruction_coef = config.reconstruction_coef
        self.reg_coef = config.reg_coef

        self.target_update_interval = config.target_update_interval
        self.actor_update_frequency = config.actor_update_frequency
        self.automatic_entropy_tuning = config.automatic_entropy_tuning
        self.entropy_decay_rate = config.entropy_decay_rate
        self.agent_noise_states = {}  # Store noise states for each agent
        self.chunk_interval = 10  # Interval for noise chunk updates
    

        self.chkpt_dir = os.path.join( config.chkpt_dir, datetime.now().strftime("%Y%m%d"))

        self.updates = 0


        try:
            available_gpus = getAvailable(order='memory', limit=1)
            if available_gpus:
                self.device = torch.device(f"cuda:{available_gpus[0]}")
            else:
                self.device = torch.device("cpu")
        except Exception as e:
            print(f"Error checking available GPUs: {e}")
            print("Falling back to CPU.")
            self.device = torch.device("cpu")
        # Total raw input: [feature || direction || action]
        encoder_input_dim = self.feature_dim + self.direction_dim + self.action_dim

        # Embedding network
        self.embedded = EmbeddedNetwork(input_dim=self.input_dim, feature_dim=self.feature_dim).to(self.device)
        self.embedded_target = EmbeddedNetwork(input_dim=self.input_dim, feature_dim=self.feature_dim).to(self.device)
        hard_update(self.embedded_target, self.embedded)  # Initialize target network with same weights
        
        # Critic network
        self.critic_target = CriticNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=encoder_input_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim,
            device=self.device
        ).to(self.device)

        self.critic = CriticNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=encoder_input_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim,
            device=self.device
        ).to(self.device)
        hard_update(self.critic_target, self.critic)  # Initialize target network with same weights


        # === Actor Network ===
        self.policy = ActorNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=self.message_dim,
            action_dim=self.action_dim,
            hidden_dim=config.actor_hidden_dim,
            device=self.device
        ).to(self.device)
        

        self.message_encoder = MessageEncoder(
            input_dim=encoder_input_dim,
            message_dim=self.message_dim,
            hidden_dim=config.communication_hidden_dim
        ).to(self.device)
        
        self.message_decoder = MessageDecoder(
            message_dim=self.message_dim,
            output_dim=encoder_input_dim,
            hidden_dim=config.communication_hidden_dim
        ).to(self.device)


        # === Optimizers ===
        self.critic_optim = Adam(
            list(self.critic.parameters()) +
            list(self.embedded.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )
        self.policy_optim = Adam(
            list(self.policy.parameters()) + 
            list(self.message_encoder.parameters()), 
            lr=self.lr,  
            weight_decay=1e-4
            )

        # === Entropy tuning ===
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -0.4*self.action_dim
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.lr,)
            self.min_entropy = -0.1*self.action_dim  # Minimum entropy for the target entropy
            
            

            
        # === Replay Buffer ===
        self.memory = ReplayMemory(self.memory_max_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def store_transition(
        self,
        current_state: np.ndarray,
        current_messages: List[np.ndarray],
        direction: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        next_messages: List[np.ndarray],
        done: bool
    ) -> None:
        self.memory.push(
            current_state=current_state,
            current_messages=current_messages,
            direction=direction,
            action=action,
            reward=reward,
            next_state=next_state,
            next_messages=next_messages,
            done=done
        )
        
    def train_critic(
        self,
        current_state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        messages_batch: List[List[torch.Tensor]],
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        next_messages_batch: List[List[torch.Tensor]],
        done_batch: torch.Tensor,
        logger: Optional[Any] = None
    ):

        self.critic_target.eval()
        self.embedded_target.eval()
        self.message_encoder.eval()
        self.policy.eval()
        self.critic.train()
        self.embedded.train()
        
        with torch.no_grad():
            # 1. Embed next state
            embedded_next_state = self.embedded_target(next_state_batch)
            # 2. Encode messages
            raw_next_message, encoded_next_messages = self.get_messages(next_messages_batch)
            
            # Sample next action from the policy (should this be deterministic or stochastic?)
            dist, mu = self.policy.forward(embedded_next_state, direction_batch, encoded_next_messages)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(embedded_next_state, direction_batch, raw_next_message, next_action)
            target_V = torch.min(target_Q1,
                                target_Q2) - self.alpha.detach() * log_prob
            target_q = reward_batch + ((1 - done_batch) * self.gamma * target_V)
            target_q = target_q.detach()
            
        # 1. Embed current state
        embedded_state = self.embedded(current_state_batch)
        # 3. Compute Q-values for current state and action
    
        raw_current_messages, _ = self.get_messages(messages_batch, encode=False)

        
        q1, q2 = self.critic(embedded_state, direction_batch, raw_current_messages, action_batch)
        # Critic loss calculation  
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Optimize Critic Network
        self.critic_optim.zero_grad()  # Clear previous gradients
        critic_loss.backward()  # Compute gradients
        self.critic_optim.step() # Update the parameters based on gradients

        if logger is None:
            return
        
        logger.log_scalar("loss/critic", critic_loss.item(), self.updates)
        logger.log_scalar("lr/critic", self.critic_optim.param_groups[0]["lr"], self.updates)


    def get_messages(self, messages_batch: List[List[torch.Tensor]], encode = True) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        # === STEP 1: Flatten and batch embed ===
        flat_data = [
            (i, j, state, direction, action)
            for i, msg_list in enumerate(messages_batch)
            for j, (state, direction, action) in enumerate(msg_list)
        ]

        # Batch all states
        states = torch.FloatTensor([state for (_, _, state, _, _) in flat_data]).to(self.device)
        embedded_states = self.embedded(states)  # shape: (N, D1)

        # Combine with direction and action
        directions = torch.FloatTensor([direction for (_, _, _, direction, _) in flat_data]).to(self.device)  # shape: (N, D2)
        actions = torch.FloatTensor([action for (_, _, _, _, action) in flat_data]).to(self.device)          # shape: (N, D3)

        # Concatenate all components
        raw_message_tensor = torch.cat([embedded_states, directions, actions], dim=-1)  # shape: (N, D1+D2+D3)

        # Store index to reconstruct
        indices = [(i, j) for (i, j, _, _, _) in flat_data]
        num_messages = len(messages_batch)
        raw_current_message = [[] for _ in range(num_messages)]
        
        if not encode:
            # If no encoding is needed, return the raw messages directly
            for (i, j), raw_vec in zip(indices, raw_message_tensor):
                raw_current_message[i].append(raw_vec)
                
            return raw_current_message, None

        # === STEP 2: Batch pass through message encoder ===
        encoded_tensor = self.message_encoder(raw_message_tensor)  # shape: (N, D4)

        # === STEP 3: Reconstruct both raw_current_message and encoded_current_messages ===
        encoded_current_messages = [[] for _ in range(num_messages)]

        for (i, j), raw_vec, encoded_vec in zip(indices, raw_message_tensor, encoded_tensor):
            raw_current_message[i].append(raw_vec)
            encoded_current_messages[i].append(encoded_vec)
            
        return raw_current_message, encoded_current_messages

    def train_actor(
        self,
        state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        messages_batch: List[List[torch.Tensor]],
        logger: Optional[Any] = None
    ):    
        
        self.message_encoder.train()
        self.policy.train()
        self.critic.eval()
        self.embedded.eval()
        with torch.no_grad():
            # === Embed current state ===
            embedded_state = self.embedded(state_batch).detach()  # [B, feature_dim]
            
        # === Sample action and compute policy loss ===
        raw_current_message, encoded_current_messages = self.get_messages(messages_batch)
        dist, mu = self.policy.forward(embedded_state, direction_batch, encoded_current_messages)
        
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Compute the policy loss using the critic's Q-values
          
    
            
        actor_Q1, actor_Q2  = self.critic(embedded_state, direction_batch, raw_current_message, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # === Total loss ===
        total_loss = actor_loss
        # total_loss += self.reconstruction_coef * recon_loss
        regularization_loss = torch.mean(torch.clamp(torch.abs(mu) - 2.0, min=0.0) ** 2)
        total_loss += self.reg_coef * regularization_loss  # Add regularization loss

        # === Optimize policy ===
        self.policy_optim.zero_grad()
        total_loss.backward()  # Compute gradients
        self.policy_optim.step()  # Update the parameters based on gradients
        
        if self.automatic_entropy_tuning:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.log_alpha_optimizer.step()
            if logger is not None:
                logger.log_scalar("alpha", self.alpha.item(), self.updates)
                logger.log_scalar("loss/alpha", alpha_loss.item(), self.updates)
                logger.log_scalar("target_entropy", self.target_entropy, self.updates)
            # self.update_target_entropy()
        
        if logger is None:
            return

        logger.log_scalar("policy/log_prob_mean", log_prob.mean().item(), self.updates)
        logger.log_scalar("policy/log_prob_min", log_prob.min().item(), self.updates)
        logger.log_scalar("policy/log_prob_max", log_prob.max().item(), self.updates)

        logger.log_scalar("policy/action_mean", action.mean(), self.updates)
        logger.log_scalar("policy/action_std", action.std(), self.updates)
        logger.log_scalar("policy/action_min", action.min(), self.updates)
        logger.log_scalar("policy/action_max", action.max(), self.updates)
        logger.log_scalar("loss/actor", actor_loss.item(), self.updates)
        logger.log_scalar("policy/regularization", regularization_loss.item(), self.updates)
        logger.log_scalar("lr/actor", self.policy_optim.param_groups[0]["lr"], self.updates)
        # logger.log_scalar("loss/reconstruction", recon_loss.item(), self.updates)
        
        

    def learn(self, logger):
        if len(self.memory) < self.batch_size:
            return None

        # === Sample a batch of transitions ===
        (
            current_state_batch,
            current_messages_batch,
            direction_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_messages_batch,
            done_batch
        ) = self.memory.sample(self.batch_size)
        
        # === Convert to torch tensors ===
        current_state_batch = torch.FloatTensor(current_state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        direction_batch = torch.FloatTensor(direction_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)


        # === Train Critic ===
        self.train_critic(
            current_state_batch, 
            direction_batch, 
            current_messages_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            next_messages_batch, 
            done_batch,
            logger=logger
            )

        # === Train Actor ===
        if self.updates % self.actor_update_frequency == 0:
            self.train_actor(current_state_batch, direction_batch, current_messages_batch, logger)

        # === Update Target Network ===
        if self.updates % self.target_update_interval == 0:
            self.update_networks()
            
        self.updates += 1

    def update_target_entropy(self):
        """Decay the target entropy over time."""
        if not self.automatic_entropy_tuning:
            return
        
        # At each training step
        self.target_entropy = min(self.min_entropy, self.target_entropy + self.entropy_decay_rate)


    def update_networks(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.embedded_target, self.embedded, self.tau)
            
    def get_agent_noise(self, agent: str, action_dim: int) -> torch.Tensor:
        """
        Get or update the noise vector for a given agent.

        Args:
            agent (str): The agent's identifier.
            action_dim (int): Dimension of the action space.

        Returns:
            torch.Tensor: Noise vector for sampling.
        """
        if agent not in self.agent_noise_states:
            self.agent_noise_states[agent] = {
                "z": torch.zeros(action_dim),
                "counter": 0
            }

        noise_state = self.agent_noise_states[agent]
        if noise_state["counter"] % self.chunk_interval == 0:
            noise_state["z"] = torch.randn(action_dim)
        noise_state["counter"] += 1

        return noise_state["z"]
    
    def choose_action(self, state: np.ndarray, direction: np.ndarray, messages: np.ndarray, agent: str = None, evaluate: bool = False):
        """
        Select an action given own state, direction, and received messages.

        Args:
            state (np.ndarray): Agent's current state (e.g., image or vector).
            direction (np.ndarray): Agent's direction input.
            messages (List[np.ndarray]): Encoded messages from other agents.
            evaluate (bool): Whether to sample deterministically.

        Returns:
            Tuple[np.ndarray, torch.Tensor, torch.Tensor]: action, message, raw_message_input
        """
        self.policy.eval()
        self.embedded.eval()

        # === Format input ===
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        direction_tensor = torch.FloatTensor(direction).unsqueeze(0).to(self.device)
        messages_tensor = torch.FloatTensor(np.array(messages)).unsqueeze(0).to(self.device)

        # === Choose action ===
        with torch.no_grad():
            # === Embed own state ===
            embedded_state = self.embedded(state_tensor)  # shape: [1, D]
            dist, mu = self.policy.forward(embedded_state, direction_tensor, messages_tensor)

            # === Choose action ===
            if evaluate:
                action = dist.mean
            else:
                action_dim = dist.loc.shape[-1]
                z = self.get_agent_noise(agent, action_dim).to(dist.loc.device)
                action = dist.sample(z=z)

            action = action.clamp(-1 , 1)
            # === Prepare message input ===
            raw_message_input = torch.cat([embedded_state, direction_tensor, action], dim=-1)  # shape: [1, D + D_dir]
            message = self.message_encoder(raw_message_input)
            
            # if(agent == "Agent-0"):
            #     # Debugging visualization
            #     self.debug_step(state, direction, messages, action, message, embedded_state, agent, save_path="debug_step_1")

            action_number = action.detach().cpu().numpy()[0]  # Convert to numpy for easier handling
            # make sure message and raw_message_input are of 1-D
            message = message.detach().cpu().numpy()[0]
            raw_message_input = raw_message_input.detach().cpu().numpy()[0]
        
        self.policy.train()
        self.embedded.train()
        
        # self.embedded_target.visualize_head_output(state_tensor)  # Visualize the head output for debugging
        return action_number, message, (state_tensor.detach().cpu().numpy()[0], direction_tensor.detach().cpu().numpy()[0], action_number)

    # Save model parameters
    def save(self, filename: str = "agent_checkpoint.pth"):
         # === Paths & Device ===
        os.makedirs(self.chkpt_dir, exist_ok=True)
        save_path = os.path.join(self.chkpt_dir, filename)
        print(f"Saving models to {save_path}")

        checkpoint = {
            # === Network States ===
            'embedded_state_dict': self.embedded.state_dict(),
            'embedded_target_state_dict': self.embedded_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'message_encoder_state_dict': self.message_encoder.state_dict(),

            # === Optimizers ===
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),

            # === Optional entropy tuning ===
            'automatic_entropy_tuning': self.automatic_entropy_tuning,
            'target_entropy': getattr(self, 'target_entropy', None),
            'log_alpha': getattr(self, 'log_alpha', None).detach().cpu() if hasattr(self, 'log_alpha') else None,
            'alpha_optimizer_state_dict': getattr(self, 'log_alpha_optimizer', None).state_dict() if hasattr(self, 'log_alpha_optimizer') else None,

            # === Hyperparameters for tracking ===
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha': self.alpha,
                'batch_size': self.batch_size,
                'feature_dim': self.feature_dim,
                'message_dim': self.message_dim,
                'action_dim': self.action_dim,
                'n_agents': self.n_agents,
                'reconstruction_coef': self.reconstruction_coef,
            }
        }

        torch.save(checkpoint, save_path)

    # Load model parameters
    def load(self, path: str, evaluate: bool = False):
        print(f"Loading models from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # === Load model weights ===
        self.embedded.load_state_dict(checkpoint['embedded_state_dict'])
        self.embedded_target.load_state_dict(checkpoint['embedded_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.message_encoder.load_state_dict(checkpoint['message_encoder_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Filter and load policy optimizer state safely
        loaded_policy_optim_state = checkpoint['policy_optimizer_state_dict']

        # Create current optimizer param ID map
        current_param_ids = {id(p): p for group in self.policy_optim.param_groups for p in group['params']}

        # Filter the state to match only existing parameter IDs
        filtered_state = {
            k: v for k, v in loaded_policy_optim_state['state'].items()
            if k in current_param_ids
        }

        # Replace the 'state' field with filtered version
        loaded_policy_optim_state['state'] = filtered_state

        # Replace param_groups to avoid mismatch
        loaded_policy_optim_state['param_groups'] = self.policy_optim.state_dict()['param_groups']

        # Load filtered state dict
        self.policy_optim.load_state_dict(loaded_policy_optim_state)

        for group in self.policy_optim.param_groups:
            group["lr"] = self.lr

        # # === Load alpha/entropy if available ===
        # if checkpoint.get('automatic_entropy_tuning', False):
        #     self.automatic_entropy_tuning = True
        #     # self.target_entropy = checkpoint.get('target_entropy', self.target_entropy)
        #     self.log_alpha = checkpoint.get('log_alpha', self.log_alpha)
        #     self.log_alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        #     self.log_alpha = torch.tensor(self.log_alpha).to(self.device)
        #     self.log_alpha.requires_grad = True

        # === Apply device & mode ===
        for net in [self.embedded, self.embedded_target, self.policy, self.critic, self.critic_target, self.message_encoder]:
            net.to(self.device)
            net.train()

        print("Model loaded successfully.")
