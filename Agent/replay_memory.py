import random
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Any

AgentState = np.ndarray
AgentAction = Any
AgentReward = float
AgentMessageInput = np.ndarray
AgentDirection = Any
AgentDone = bool

Transition = Tuple[
    AgentState,
    List[AgentMessageInput],  # current messages
    AgentDirection,
    AgentAction,
    AgentReward,
    AgentState,
    List[AgentMessageInput],  # next messages
    AgentDone
]

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer: List[Optional[Transition]] = []
        self.position: int = 0

    def push(
        self,
        current_state: AgentState,
        current_messages: List[AgentMessageInput],
        direction: AgentDirection,
        action: AgentAction,
        reward: AgentReward,
        next_state: AgentState,
        next_messages: List[AgentMessageInput],
        done: AgentDone
    ) -> None:
        transition: Transition = (
            current_state,
            current_messages,
            direction,
            action,
            reward,
            next_state,
            next_messages,
            done
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray,                  # current_state batch
        List[List[np.ndarray]],      # current_messages batch
        np.ndarray,                  # direction batch
        np.ndarray,                  # action batch
        np.ndarray,                  # reward batch
        np.ndarray,                  # next_state batch
        List[List[np.ndarray]],      # next_messages batch
        np.ndarray                   # done batch
    ]:
        batch = random.sample(self.buffer, batch_size)

        current_state = np.stack([t[0] for t in batch])
        current_messages = [t[1] for t in batch]
        directions = np.stack([t[2] for t in batch])
        actions = np.stack([t[3] for t in batch])
        rewards = np.stack([t[4] for t in batch])
        next_state = np.stack([t[5] for t in batch])
        next_messages = [t[6] for t in batch]
        dones = np.stack([t[7] for t in batch])

        return (
            current_state,
            current_messages,
            directions,
            actions,
            rewards,
            next_state,
            next_messages,
            dones
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def save_buffer(self, env_name: str, suffix: str = "", save_path: Optional[str] = None) -> None:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = f"checkpoints/per_agent_buffer_{env_name}_{suffix}.pkl"
        print(f'Saving buffer to {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path: str) -> None:
        print(f'Loading buffer from {save_path}')
        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
