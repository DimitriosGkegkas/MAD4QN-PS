import gymnasium as gym
import numpy as np
from typing import List, Dict, Any, Optional
from utils import position_to_road, roads_to_direction, road_to_communication


class CommunicationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        agent_names: List[str] = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3'],
        message_dim: int = 8,  # Default message dimension
        message_raw_dim: int = 8  # Default raw message dimension
    ) -> None:
        """
        Appends communication messages to each agent's observation.
        Observation becomes a tuple: (original_observation, aggregated_message).

        Args:
            env (gym.Env): The wrapped environment.
            agent_names (List[str]): Agents to consider for communication.
            message_dim (int): The dimension of each incoming message vector.
        """
        super().__init__(env)
        self.agent_names = agent_names
        self.message_dim = message_dim
        self.message_raw_dim = message_raw_dim

    def reset(self, **kwargs):
        observation, reward, terminated, truncated, info = self.env.reset(**kwargs)
        return self.observation(observation, messages=None), reward, terminated, truncated, info

    def step(self, action, messages: Optional[Dict[str, np.ndarray]] = None, raw_messages: Optional[Dict[str, np.ndarray]] = None):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation, messages, raw_messages), reward, terminated, truncated, info

    def observation(self, obs: Dict[str, Any], messages: Optional[Dict[str, np.ndarray]], raw_messages: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Adds an aggregated communication message to each agent's observation tuple.
        Input shape: (obs_data, direction_vector)
        Output shape: (obs_data, direction_vector, aggregated_message)
        """
        if messages is None:
            messages = {agent: np.zeros(self.message_dim) for agent in self.agent_names}
            
        if raw_messages is None:
            raw_messages = {agent: np.zeros(self.message_raw_dim) for agent in self.agent_names}

        return {
            agent: (*obs[agent], self.create_communication_message(messages), self.create_communication_message(raw_messages))
            if agent in self.agent_names and isinstance(obs[agent], tuple)
            else obs[agent]
            for agent in obs
        }


    def create_communication_message(self, messages: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregates messages from all communicable agents.
        Returns:
            np.ndarray: Concatenated message vector.
        """
        agent_msg = [
            msg for msg in messages.values() if msg is not None
        ]
        return agent_msg
     
