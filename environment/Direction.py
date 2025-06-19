import gymnasium as gym
import numpy as np
from typing import List, Dict, Any
from utils import position_to_road, roads_to_direction


class Direction(gym.ObservationWrapper):
    def __init__(
        self, 
        env: gym.Env, 
        agent_names: List[str] = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
    ) -> None:
        """
        Appends a direction vector to each agent's observation.
        """
        super().__init__(env)
        self.agent_names = agent_names
        self.directions = {agent: None for agent in agent_names}
        
    def reset(self, **kwargs):
        observation, reward, terminated, truncated, info = self.env.reset(**kwargs)
        self.directions = {
            agent: get_direction_vector_from_info(info[agent])
            for agent in info
            if agent in self.agent_names
        }
        return self.observation(observation), reward, terminated, truncated, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms the observation by prepending a direction vector to each agent's data.
        Result is a tuple: (direction_vector, original_observation)
        """
        return {
            agent: (obs[agent], self.directions[agent])
            if agent in self.agent_names else obs[agent]
            for agent in obs
        }



def get_direction_vector(mission):
    start = position_to_road([mission.start.position.x, mission.start.position.y])
    goal = position_to_road([mission.goal.position.x, mission.goal.position.y])
    roads = start + goal
    return roads_to_direction[roads]


def get_direction_vector_from_info(info):
    if 'env_obs' in info:
        ego = info['env_obs'].ego_vehicle_state
        return get_direction_vector(ego.mission)
    elif 'mission' in info:
        return get_direction_vector(info['mission'])
    else:
        raise Exception("No mission or env_obs found in info")