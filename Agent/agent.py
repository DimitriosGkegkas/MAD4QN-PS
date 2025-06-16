from datetime import datetime
import os
from typing import Any, List, Optional, Tuple
from matplotlib.style import available
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from torch.optim import Adam
from Agent.utils import soft_update, hard_update
from .Networks import ActorNetwork, CriticNetwork, EmbeddedNetwork, MessageEncoder, MessageDecoder
import numpy as np
import re

from Agent.replay_memory import ReplayMemory
from  GPUtil import getAvailable

from dataclasses import dataclass
from typing import Any, Tuple, Optional

@dataclass
class AgentConfig:
    
    # hidden layers for networks
    communication_hidden_dim: List[int]
    critic_hidden_dim: List[int]
    actor_hidden_dim: List[int]
    
    input_dim: Tuple[int, int, int] = (3, 32, 32)
    action_dim: int = 1
    direction_dim: int = 1
    feature_dim: int = 100
    message_dim: int = 8
    n_agents: int = 4

    tau: float = 0.005
    gamma: float = 0.99
    lr: float = 1e-4
    alpha: float = 0.2
    memory_max_size: int = 1_000_000
    batch_size: int = 64

    reconstruction_coef: float = 0.01
    img_reconstruction_coef: float = 0.01
    reg_coef: float = 0.1
    smoothness_coef: float = 0.01

    target_update_interval: int = 1
    automatic_entropy_tuning: bool = True

    chkpt_dir: str = 'tmp/dqn'
    


class Agent:
    def __init__(self, config: AgentConfig):
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.direction_dim = config.direction_dim
        self.feature_dim = config.feature_dim
        self.message_dim = config.message_dim
        self.n_agents = config.n_agents

        self.tau = config.tau
        self.gamma = config.gamma
        self.lr = config.lr
        self.alpha = config.alpha
        self.memory_max_size = config.memory_max_size
        self.batch_size = config.batch_size

        self.reconstruction_coef = config.reconstruction_coef
        self.img_reconstruction_coef = config.img_reconstruction_coef
        self.reg_coef = config.reg_coef
        self.smoothness_coef = config.smoothness_coef

        self.target_update_interval = config.target_update_interval
        self.automatic_entropy_tuning = config.automatic_entropy_tuning

        self.chkpt_dir = os.path.join( config.chkpt_dir, datetime.now().strftime("%Y%m%d"))

        self.updates = 0
        self.total_message_dim = (self.n_agents - 1) * self.message_dim


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
            

        # Embedding network
        self.embedded = EmbeddedNetwork(input_dim=self.input_dim, feature_dim=self.feature_dim).to(self.device)
        self.embedded_target = EmbeddedNetwork(input_dim=self.input_dim, feature_dim=self.feature_dim).to(self.device)
        hard_update(self.embedded_target, self.embedded)  # Initialize target network with same weights
        
        # Critic network
        self.critic_target = CriticNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=self.total_message_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim
        ).to(self.device)

        self.critic = CriticNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=self.total_message_dim,
            action_dim=self.action_dim,
            hidden_dim=config.critic_hidden_dim
        ).to(self.device)
        hard_update(self.critic_target, self.critic)  # Initialize target network with same weights


        # === Actor Network ===
        self.policy = ActorNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=self.total_message_dim,
            action_dim=self.action_dim,
            hidden_dim=config.actor_hidden_dim
        ).to(self.device)
        
        # Total raw input: [feature || direction || action]
        encoder_input_dim = self.feature_dim + self.direction_dim + self.action_dim

        self.message_encoder = MessageEncoder(
            input_dim=encoder_input_dim,
            message_dim=self.message_dim,
            hidden_dim=config.communication_hidden_dim
        ).to(self.device)

        self.message_decoder = MessageDecoder(
            message_dim=self.message_dim,
            output_dim=encoder_input_dim,
            hidden_dim=config.communication_hidden_dim[::-1]  # Reverse the hidden dimensions for decoder
        ).to(self.device)


        # === Optimizers ===
        self.critic_optim = Adam(
            list(self.critic.parameters()) + list(self.embedded.parameters()) +
            list(self.message_encoder.parameters()) + list(self.message_decoder.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr,  weight_decay=1e-4)

        # === Entropy tuning ===
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((self.action_dim,)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        # === Replay Buffer ===
        self.memory = ReplayMemory(self.memory_max_size)


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

    def encode_messages(self, message_batch: List[List[torch.Tensor]]) -> torch.Tensor:  
              
        return torch.stack([
            torch.cat([self.message_encoder(torch.Tensor(m).to(self.device)) if m is not None else torch.zeros(self.message_dim).to(self.device) for m in raw_msgs], dim=-1)
            for raw_msgs in message_batch
        ]).to(self.device)  # [B, total_msg_dim]

    def get_reconstruction_loss(
        self,
        action_batch: torch.Tensor,
        state: torch.Tensor,
        direction_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            embedded_state = self.embedded(state)
            raw_input = torch.cat([action_batch, embedded_state, direction_batch], dim=-1)
        encoded = self.message_encoder(raw_input)
        decoded = self.message_decoder(encoded)
        recon_loss = F.mse_loss(decoded, raw_input)
        return recon_loss

    def get_smoothness_loss(
        self,
        encoded_current: torch.Tensor,
        encoded_next: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(encoded_current, encoded_next)
    
    
    def get_critic_loss(
        self,
        current_state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        message_batch: List[List[torch.Tensor]],
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        next_message_batch: List[List[torch.Tensor]],
        done_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute next action and Q-values for target update (no gradients)
        with torch.no_grad():
            # 1. Embed next state
            embedded_next_state = self.embedded_target(next_state_batch)
            # 2. Encode messages
            encoded_next_messages = self.encode_messages(next_message_batch)
            # Sample next action from the policy (should this be deterministic or stochastic?)
            next_action, _, _, _= self.policy.sample(embedded_next_state, direction_batch, encoded_next_messages)
            # Compute target Q-values
            q1_next, q2_next = self.critic_target(embedded_next_state, direction_batch, encoded_next_messages, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            # Compute target Q-value using Bellman equation
            target_q = reward_batch + self.gamma * (1 - done_batch) * min_q_next
            
            
        # 1. Embed current state
        embedded_state = self.embedded(current_state_batch)
        # 2. Encode messages
        encoded_current_messages = self.encode_messages(message_batch)
        # 3. Compute Q-values for current state and action
        q1, q2 = self.critic(embedded_state, direction_batch, encoded_current_messages, action_batch)
        # Critic loss calculation  
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        return critic_loss
        

    def train_critic(
        self,
        current_state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        message_batch: List[List[torch.Tensor]],
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        next_message_batch: List[List[torch.Tensor]],
        done_batch: torch.Tensor
    ):

        # Critic loss
        critic_loss = self.get_critic_loss(
            current_state_batch,
            direction_batch,
            message_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_message_batch,
            done_batch
        )

        # Reconstruction loss
        recon_loss = self.get_reconstruction_loss(action_batch, current_state_batch, direction_batch)
        
        # image reconstruction loss
        img_recon_loss = self.embedded.reconstruction_loss(current_state_batch)

        # Total loss
        total_loss = (
            critic_loss +
            self.reconstruction_coef * recon_loss +
            self.img_reconstruction_coef * img_recon_loss
        )

        # Optimize Critic Network
        self.critic_optim.zero_grad()  # Clear previous gradients
        total_loss.backward()  # Compute gradients

        # Clip gradients before performing the optimization step
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.embedded.parameters()) +
            list(self.message_encoder.parameters()) + list(self.message_decoder.parameters()), max_norm=1.0
        )

        # Perform the optimization step
        self.critic_optim.step()

        # Return loss values for monitoring/tracking
        return critic_loss.item(), recon_loss.item(), img_recon_loss.item()
    
    
    def get_policy_loss(
        self,
        state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        message_batch: List[List[torch.Tensor]]
    ):
        
        with torch.no_grad():
            # === Embed current state ===
            embedded_state = self.embedded(state_batch)  # [B, D]
            
            # === Aggregate messages ===
            encoded_current_messages = self.encode_messages(message_batch)

        # === Sample action and compute policy loss ===
        action, log_pi, _, raw_mean = self.policy.sample(embedded_state, direction_batch, encoded_current_messages)
        
        # Compute the policy loss using the critic's Q-values
        q1_pi, q2_pi = self.critic(embedded_state, direction_batch, encoded_current_messages, action)
        min_q_pi = torch.min(q1_pi, q2_pi)  # Minimum Q-value across the Q1 and Q2 streams

        # Policy loss: maximize Q-values (minimizing negative Q-values)
        policy_loss = (-min_q_pi).mean()
        
        # === Regularization loss (penalizing raw mean values) ===
        regularization_loss = torch.mean(torch.clamp(torch.abs(raw_mean) - 1.0, min=0.0) ** 2)
        
        return policy_loss, regularization_loss, log_pi
        
        
        
    def train_actor(
        self,
        state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        message_batch: List[List[torch.Tensor]]
    ):
        policy_loss, regularization_loss, log_pi = self.get_policy_loss(
            state_batch, direction_batch, message_batch
        )

        # === Total loss ===
        total_loss = policy_loss + self.reg_coef * regularization_loss

        # === Optimize policy ===
        self.policy_optim.zero_grad()
        total_loss.backward()  # Compute gradients
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.policy_optim.step()  # Update the parameters based on gradients

        return policy_loss.item(), regularization_loss.item(), log_pi

        
        
    def update_entropy(self, log_pi):
        if not self.automatic_entropy_tuning:
            return torch.tensor(0.).to(self.device), torch.tensor(self.alpha)

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        return alpha_loss.item(), self.alpha.clone().item()


    def learn(self):
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
        critic_loss, recon_loss, img_recon_loss = self.train_critic(current_state_batch, direction_batch, current_messages_batch, action_batch, reward_batch, next_state_batch, next_messages_batch, done_batch)

        # === Train Actor ===
        policy_loss, regularization_loss, log_pi = self.train_actor(current_state_batch, direction_batch, current_messages_batch)

        # # === Entropy Tuning ===
        # alpha_loss, alpha_tlogs = self.update_entropy(log_pi)
        alpha_loss = 0

        # === Update Target Network ===
        if self.updates % self.target_update_interval == 0:
            self.update_networks()
        
        self.updates += 1
        return critic_loss, recon_loss, img_recon_loss, policy_loss, alpha_loss, regularization_loss


    def update_networks(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.embedded_target, self.embedded, self.tau)
            
    
    
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
        messages_tensor = torch.FloatTensor(messages).unsqueeze(0).to(self.device)


        

        # === Choose action ===
        with torch.no_grad():
            # === Embed own state ===
            embedded_state = self.embedded(state_tensor)  # shape: [1, D]
            if evaluate:
                _, _, action, _ = self.policy.sample(embedded_state, direction_tensor, messages_tensor)
            else:
                action, _, _, _ = self.policy.sample(embedded_state, direction_tensor, messages_tensor)
                
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
        return action_number, message, raw_message_input

    def debug_step(self, state, direction, messages, action, message, embedded_state, agent, save_path="debug_step"):
        """
        Enhanced debug visualization for model step analysis.
        """
        self.critic.eval()
        self.embedded.head.eval()

        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        direction_tensor = torch.FloatTensor(direction).unsqueeze(0).to(self.device)
        messages_tensor = torch.FloatTensor(messages).unsqueeze(0).to(self.device)
        # Forward pass
        with torch.no_grad():
            q1, q2 = self.critic(embedded_state, direction_tensor, messages_tensor, action)
            q1 = q1.squeeze().cpu().numpy()
            q2 = q2.squeeze().cpu().numpy()
            reconstructed_image = self.embedded.head(state_tensor).squeeze().cpu().numpy()

        embedded_np = embedded_state.squeeze().cpu().numpy()
        flat_msg = messages.flatten()
        
        # === Evaluate critic over a sweep of test actions ===
        test_actions = torch.FloatTensor(np.linspace(-4, 4, 10)).unsqueeze(1).to(self.device)  # Shape: (5, 1)
        repeated_embedded = embedded_state.expand(test_actions.size(0), -1)
        repeated_dir = direction_tensor.expand(test_actions.size(0), -1)
        repeated_msgs = messages_tensor.expand(test_actions.size(0), -1)

        with torch.no_grad():
            q1s, q2s = self.critic(repeated_embedded, repeated_dir, repeated_msgs, test_actions)
            q_vals = ((q1s + q2s) / 2.0).squeeze().cpu().numpy()
            action_vals = test_actions.squeeze().cpu().numpy()


        # === Start plotting ===
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 0.5, 1])

        # ---- 4 State Channels (Top Left) ----
        for i in range(4):
            rgb_img = np.transpose(state[i*3:(i+1)*3], (1, 2, 0))  # Convert (3, H, W) → (H, W, 3)
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.imshow(rgb_img)
            ax.set_title(f"State Image {i}", fontsize=10)
            ax.axis('off')
        # ---- Reconstructed Image (Below State) ----
        ax = fig.add_subplot(gs[1, 2])

        # Assume reconstructed_image is (3, H, W) for RGB
        if reconstructed_image.ndim == 3 and reconstructed_image.shape[0] == 3:
            rgb_img = np.transpose(reconstructed_image, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            ax.imshow(np.clip(rgb_img, 0, 1))  # Optional: ensure values are in displayable range
        else:
            ax.imshow(reconstructed_image)  # Fallback in case shape is already (H, W, 3)

        ax.set_title("Reconstructed Image", fontsize=12)
        ax.axis('off')
        # ---- Action & Direction (Center Text Block) ----
        ax = fig.add_subplot(gs[2, :])
        action_text = f"Action Taken: {action}"
        direction_text = f"Direction: {int(direction[0])}"
        critic_text = f"Critic Values → Q1: {q1:.2f}, Q2: {q2:.2f}"
        ax.text(0.05, 0.6, action_text, fontsize=18, fontweight='bold')
        ax.text(0.05, 0.3, direction_text + " | " + critic_text, fontsize=14)
        ax.axis('off')

        # ---- Embedded State Features ----
        ax = fig.add_subplot(gs[3, 0])
        ax.bar(range(len(embedded_np)), embedded_np)
        ax.set_title("Embedded State Features", fontsize=12)

        # ---- Messages ----
        ax = fig.add_subplot(gs[3, 1])
        ax.bar(range(len(flat_msg)), flat_msg)
        ax.set_title("Messages Vector", fontsize=12)

        # ---- Raw Message Output ----
        ax = fig.add_subplot(gs[3, 2])
        message_np = message.squeeze().detach().cpu().numpy()
        ax.bar(range(len(message_np)), message_np)
        ax.set_title("Output Message Vector", fontsize=12)
        
        # ---- Q-Value Landscape for Sampled Actions ----
        ax = fig.add_subplot(gs[2, 2])
        ax.bar([f"{a:.2f}" for a in action_vals], q_vals)
        ax.set_title("Q-Value vs. Sampled Actions", fontsize=12)
        ax.set_xlabel("Action")
        ax.set_ylabel("Avg Q-Value")

        # Scale y-axis to the min/max of q_vals
        ax.set_ylim(q_vals.min(), q_vals.max())

        # Final touches
        fig.subplots_adjust(hspace=0.7, wspace=0.4)
        plt.suptitle("Debug Step Visualization", fontsize=18, fontweight='bold')

        save_dir = os.path.join(save_path, agent)
        os.makedirs(save_dir, exist_ok=True)

        # List files and extract numeric filenames like "0.png", "1.png", etc.
        existing_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        existing_indices = [
            int(re.match(r"(\d+)\.png", f).group(1))
            for f in existing_files
            if re.match(r"(\d+)\.png", f)
        ]
        next_index = max(existing_indices, default=-1) + 1  # Start from 0 if none exist

        # Build full file path
        filename = f"{next_index}.png"
        full_save_path = os.path.join(save_dir, filename)

        # Save the figure
        plt.savefig(full_save_path)
        plt.close(fig)

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
            'message_decoder_state_dict': self.message_decoder.state_dict(),

            # === Optimizers ===
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),

            # === Optional entropy tuning ===
            'automatic_entropy_tuning': self.automatic_entropy_tuning,
            'alpha': self.alpha,
            'target_entropy': getattr(self, 'target_entropy', None),
            'log_alpha': getattr(self, 'log_alpha', None).detach().cpu() if hasattr(self, 'log_alpha') else None,
            'alpha_optimizer_state_dict': getattr(self, 'alpha_optim', None).state_dict() if hasattr(self, 'alpha_optim') else None,

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
                'smoothness_coef': self.smoothness_coef
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
        self.message_decoder.load_state_dict(checkpoint['message_decoder_state_dict'])

        # === Load optimizers ===
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        # === Load alpha/entropy if available ===
        # if checkpoint.get('automatic_entropy_tuning', False):
        #     self.automatic_entropy_tuning = True
        #     self.alpha = checkpoint.get('alpha', self.alpha)
        #     self.target_entropy = checkpoint.get('target_entropy', self.target_entropy)
        #     self.log_alpha = checkpoint.get('log_alpha', self.log_alpha)
        #     self.log_alpha = self.log_alpha.to(self.device)
        #     self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        # === Apply device & mode ===
        for net in [self.embedded, self.embedded_target, self.policy, self.critic, self.critic_target, self.message_encoder, self.message_decoder]:
            net.to(self.device)
            net.train()

        print("Model loaded successfully.")
