import os
from typing import Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.optim import Adam
from Agent.utils import soft_update, hard_update
from .Networks import ActorNetwork, CriticNetwork, EmbeddedNetwork, MessageEncoder, MessageDecoder
import numpy as np

from Agent.replay_memory import ReplayMemory
# from  GPUtil import getAvailable

class Agent(object):
    def __init__(
        self,
        input_dim: Tuple[int, int, int] = (3, 32, 32),  # e.g., (C, H, W) for image input
        action_dim: int = 1,
        direction_dim: int = 2,  # Direction input dimension
        feature_dim: int = 100,
        message_dim: int = 8,
        n_agents: int = 4,
        
        tau: float = 0.005,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha: float = 0.2,
        max_size: int = 1_000_000,
        batch_size: int = 64,
        reconstruction_coef: float = 0.01,
        smoothness_coef: float = 0.01,
        target_update_interval: int = 1,
        automatic_entropy_tuning: bool = True,
        env_name: Optional[str] = None,
        chkpt_dir: str = 'tmp/dqn',
        training_stats_path: str = 'tmp/dqn_stats'
    ):
        # === Store Hyperparameters ===
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.reconstruction_coef = reconstruction_coef
        self.smoothness_coef = smoothness_coef
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.env_name = env_name
        self.updates = 0

        # === Store core dimensions ===
        self.input_dim = input_dim  # e.g., (C, H, W)
        self.action_dim = action_dim
        self.direction_dim = direction_dim
        self.feature_dim = feature_dim
        self.message_dim = message_dim
        self.n_agents = n_agents
        self.total_message_dim = (n_agents - 1) * message_dim


        # === Paths & Device ===
        self.chkpt_dir = os.path.join(chkpt_dir, env_name or "default")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding network
        self.embedded = EmbeddedNetwork(input_dim=input_dim, feature_dim=feature_dim).to(self.device)
        self.embedded_target = EmbeddedNetwork(input_dim=input_dim, feature_dim=feature_dim).to(self.device)
        hard_update(self.embedded_target, self.embedded)  # Initialize target network with same weights
        
        # Critic network
        self.critic_target = CriticNetwork(
            feature_dim=feature_dim,
            direction_dim=direction_dim,
            message_dim=self.total_message_dim,
            action_dim=action_dim,
            hidden_dim=[128, 128]
        ).to(self.device)

        self.critic = CriticNetwork(
            feature_dim=feature_dim,
            direction_dim=direction_dim,
            message_dim=self.total_message_dim,
            action_dim=action_dim,
            hidden_dim=[128, 128]
        ).to(self.device)
        hard_update(self.critic_target, self.critic)  # Initialize target network with same weights


        # === Actor Network ===
        self.policy = ActorNetwork(
            feature_dim=self.feature_dim,
            direction_dim=self.direction_dim,
            message_dim=self.total_message_dim,
            action_dim=self.action_dim,
            hidden_dim=[128, 128]
        ).to(self.device)
        
        # Total raw input: [feature || direction || action]
        encoder_input_dim = feature_dim + direction_dim + action_dim

        self.message_encoder = MessageEncoder(
            input_dim=encoder_input_dim,
            message_dim=message_dim,
            hidden_dim=[64, 64]
        ).to(self.device)

        self.message_decoder = MessageDecoder(
            message_dim=message_dim,
            output_dim=encoder_input_dim,
            hidden_dim=[64, 64]
        ).to(self.device)


        # === Optimizers ===
        self.critic_optim = Adam(
            list(self.critic.parameters()) + list(self.embedded.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        # === Entropy tuning ===
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((self.action_dim,)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

        # === Replay Buffer ===
        self.memory = ReplayMemory(max_size)


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
            torch.cat([self.message_encoder(m) for m in raw_msgs], dim=-1)
            for raw_msgs in message_batch
        ]).to(self.device)  # [B, total_msg_dim]

    def get_critic_loss(
        self,
        embedded_state: torch.Tensor,
        direction_batch: torch.Tensor,
        encoded_messages: torch.Tensor,
        action_batch: torch.Tensor,
        target_q: torch.Tensor
    ) -> torch.Tensor:
        q1, q2 = self.critic(embedded_state, direction_batch, encoded_messages, action_batch)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        return q1_loss + q2_loss, q1_loss, q2_loss

    def get_reconstruction_loss(
        self,
        action_batch: torch.Tensor,
        embedded_state: torch.Tensor,
        direction_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_input = torch.cat([action_batch, embedded_state, direction_batch], dim=-1)
        encoded = self.message_encoder(raw_input)
        decoded = self.message_decoder(encoded)
        recon_loss = F.mse_loss(decoded, raw_input)
        return recon_loss, encoded  # Return encoded for smoothness loss

    def get_smoothness_loss(
        self,
        encoded_current: torch.Tensor,
        encoded_next: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(encoded_current, encoded_next)

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
        embedded_state = self.embedded(current_state_batch)
        embedded_next_state = self.embedded_target(next_state_batch)

        encoded_current_messages = self.encode_messages(message_batch)
        encoded_next_messages = self.encode_messages(next_message_batch)

        with torch.no_grad():
            next_action, _, _ = self.policy.sample(embedded_next_state, direction_batch, encoded_next_messages)
            q1_next, q2_next = self.critic_target(embedded_next_state, direction_batch, encoded_next_messages, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = reward_batch + self.gamma * (1 - done_batch) * min_q_next

        # === Critic loss ===
        critic_loss, q1_loss, q2_loss = self.get_critic_loss(
            embedded_state, direction_batch, encoded_current_messages, action_batch, target_q
        )

        # === Reconstruction loss ===
        recon_loss_current, encoded_current = self.get_reconstruction_loss(action_batch, embedded_state, direction_batch)
        recon_loss_next, encoded_next = self.get_reconstruction_loss(next_action, embedded_next_state, direction_batch)
        recon_loss = recon_loss_current + recon_loss_next

        # === Smoothness loss ===
        smoothness_loss = self.get_smoothness_loss(encoded_current, encoded_next)

        # === Total loss ===
        total_loss = (
            critic_loss +
            self.reconstruction_coef * recon_loss +
            self.smoothness_coef * smoothness_loss
        )

        # === Optimize ===
        self.critic_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.embedded.parameters()), max_norm=10.0
        )
        self.critic_optim.step()

        return critic_loss.item(), recon_loss.item(), smoothness_loss.item()

    
    def train_actor(
        self,
        state_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        message_batch: List[List[torch.Tensor]]
    ):
        # === Embed current state ===
        embedded_state = self.embedded(state_batch)  # [B, D]

        # === Aggregate messages ===
        encoded_current_messages = self.encode_messages(message_batch)
    

        # === Sample action and compute policy loss ===
        action, log_pi, _ = self.policy.sample(embedded_state, direction_batch, encoded_current_messages)
        q1_pi, q2_pi = self.critic(embedded_state, direction_batch, encoded_current_messages, action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (-min_q_pi).mean()

        # === Optimize policy ===
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss.item(), log_pi

    
    
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
            return 0, 0, 0, 0, 0

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
        current_messages_batch = [torch.tensor(np.array(msg), dtype=torch.float32).to(self.device) for msg in current_messages_batch]
        next_messages_batch = [torch.tensor(np.array(msg), dtype=torch.float32).to(self.device) for msg in next_messages_batch]


        # === Train Critic ===
        critic_loss, recon_loss, smooth_loss = self.train_critic(current_state_batch, direction_batch, current_messages_batch, action_batch, reward_batch, next_state_batch, next_messages_batch, done_batch)

        # === Train Actor ===
        policy_loss, log_pi = self.train_actor(current_state_batch, direction_batch, current_messages_batch)

        # === Entropy Tuning ===
        alpha_loss, alpha_tlogs = self.update_entropy(log_pi)

        # === Update Target Network ===
        if self.updates % self.target_update_interval == 0:
            self.update_networks()
        
        self.updates += 1
        return critic_loss, recon_loss, smooth_loss, policy_loss, alpha_loss


    def update_networks(self):
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.embedded_target, self.embedded, self.tau)
            
    def get_embedded_input(self, input_tuple):
        
        img, direction = input_tuple
        img = torch.FloatTensor(img).to(self.device)
        direction = torch.FloatTensor(direction).to(self.device)

        img = torch.FloatTensor(img).to(self.device)
        direction = torch.FloatTensor(direction).to(self.device)

        if img.dim() == 3:  # single image: [C, H, W]
            img = img.unsqueeze(0)
        if direction.dim() == 1:  # single direction vector: [D]
            direction = direction.unsqueeze(0)

        return (img, direction)

    
    
    def choose_action(self, state: np.ndarray, direction: np.ndarray, messages: List[np.ndarray], evaluate: bool = False):
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

        # === Format input ===
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        direction_tensor = torch.FloatTensor(direction).unsqueeze(0).to(self.device)

        # === Embed own state ===
        embedded_state = self.embedded(state_tensor)  # shape: [1, D]

        # === Aggregate incoming messages ===
        # messages: List[Tensor], each shape: [msg_dim]
        # if any of the messages is None, we replace it with a zero tensor with the same shape
        messages = [
            torch.zeros(self.message_dim).to(self.device) if msg is None else torch.FloatTensor(msg).to(self.device)
            for msg in messages
        ]
        aggregated_message = torch.cat(messages, dim=-1).unsqueeze(0).to(self.device)  # [1, N * msg_dim]


        # === Choose action ===
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(embedded_state, direction_tensor, aggregated_message)
            else:
                action, _, _ = self.policy.sample(embedded_state, direction_tensor, aggregated_message)
                
        # === Prepare message input ===
        raw_message_input = torch.cat([embedded_state, direction_tensor, action], dim=-1)  # shape: [1, D + D_dir]
        message = self.message_encoder(raw_message_input)

        self.policy.train()
        return action.detach().cpu().numpy()[0], message, raw_message_input

    
    
    # Save model parameters
    def save(self, filename: str = "agent_checkpoint.pth"):
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
        if checkpoint.get('automatic_entropy_tuning', False):
            self.automatic_entropy_tuning = True
            self.alpha = checkpoint.get('alpha', self.alpha)
            self.target_entropy = checkpoint.get('target_entropy', self.target_entropy)
            self.log_alpha = checkpoint.get('log_alpha', self.log_alpha)
            self.log_alpha = self.log_alpha.to(self.device)
            self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        # === Apply device & mode ===
        for net in [self.embedded, self.embedded_target, self.policy, self.critic, self.critic_target, self.message_encoder, self.message_decoder]:
            net.to(self.device)
            net.eval() if evaluate else net.train()

        print("Model loaded successfully.")
