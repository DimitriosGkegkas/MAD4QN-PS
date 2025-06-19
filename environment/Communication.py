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
        self.communication_map = {agent: [] for agent in agent_names}
        self.message_dim = message_dim
        self.message_raw_dim = message_raw_dim

    def reset(self, **kwargs):
        observation, reward, terminated, truncated, info = self.env.reset(**kwargs)
        self.communication_map = get_communication_map(info)
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
            agent: (*obs[agent], self.create_communication_message(agent, messages), self.create_communication_raw_message(agent, raw_messages))
            if agent in self.agent_names and isinstance(obs[agent], tuple)
            else obs[agent]
            for agent in obs
        }


    def create_communication_message(self, agent: str, messages: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregates messages from all communicable agents.
        Returns:
            np.ndarray: Concatenated message vector.
        """
        agent_msg = [
            messages.get(communicating_agent, np.zeros(self.message_dim))
            for communicating_agent in self.communication_map.get(agent, [])
        ]
        return np.concatenate(agent_msg) if agent_msg else np.zeros(self.message_dim)
    
    def create_communication_raw_message(self, agent: str, messages) -> List[Optional[np.ndarray]]:
        agent_msg = [
                    messages[communicating_agent] if communicating_agent in messages else None
                    for communicating_agent in self.communication_map.get(agent, [])
                    ]
        return agent_msg    


def get_communication_map(info: dict) -> dict:
    """
    Builds a map of which agents can communicate with which based on road layout.
    """
    agent_start_roads = {}

    for agent, agent_info in info.items():
        # Try to extract mission start position
        try:
            if 'env_obs' in agent_info:
                start_pos = agent_info['env_obs'][5].mission.start.position
            elif 'mission' in agent_info:
                start_pos = agent_info['mission'].start.position
            else:
                continue
            start_road = position_to_road([start_pos.x, start_pos.y])
            agent_start_roads[agent] = start_road
        except Exception as e:
            print(f"[WARN] Could not get start road for {agent}: {e}")

    communication_map = {}
    for agent, start_road in agent_start_roads.items():
        connected_roads = road_to_communication.get(start_road, [])
        communication_map[agent] = [
            other_agent for other_agent, other_start in agent_start_roads.items()
            if other_agent != agent and other_start in connected_roads
        ]

    return communication_map
