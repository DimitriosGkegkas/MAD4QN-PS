import re
import gymnasium as gym
import numpy as np
from smarts.core.sensor import AccelerometerSensor
import math
from smarts.core.utils.core_math import signed_dist_to_line, radians_to_vec

la = 0.1
lj = 0.05
lt = 2
lx = 1
k = 1

class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_names=None):
        """
        Initializes the Reward wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            agent_names (List[str], optional): List of agent names. Defaults to 
                                               ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3'].
        """
        super().__init__(env)
        self.agent_names = agent_names or ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
        self.env = env

    def reset(self, **kwargs):
        """Resets the environment."""
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Steps through the environment.

        Args:
            action: The actions to be performed.

        Returns:
            Tuple: Observation, wrapped reward, termination flags, truncation flags, and info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_reward = self._compute_reward(obs, reward, info)
        return obs, wrapped_reward, terminated, truncated, info
    


    def dist_to(self, a, b) -> float:
        """Calculates straight line distance to the given 2D point"""
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def get_lateral_error(self, obs: dict) -> dict:
        position = obs["ego_vehicle_state"]["position"]
        heading = obs["ego_vehicle_state"]["heading"]
        look_ahead_wp_num = 4
        look_ahead_wp = obs["waypoint_paths"]["position"][0][look_ahead_wp_num]
        look_ahead_wp_head = obs["waypoint_paths"]["heading"][0][look_ahead_wp_num]
        look_ahead_dist = self.dist_to(look_ahead_wp[:2], position[:2])
        vehicle_look_ahead_pt = [
            position[0] - look_ahead_dist * math.sin(heading),
            position[1] + look_ahead_dist * math.cos(heading),
        ]

        lat_error = signed_dist_to_line(
            vehicle_look_ahead_pt, look_ahead_wp[:2], radians_to_vec(look_ahead_wp_head)
        )
        return abs(lat_error)  # Return the absolute value of the lateral error

    def _compute_reward(self, obs: dict, env_reward: dict, info: dict) -> np.ndarray:
        """
        Computes the reward for each agent.

        Args:
            obs (dict): The observation dictionary.
            env_reward (dict): The environment-provided rewards.

        Returns:
            np.ndarray: The computed rewards for all agents.
        """
        reward = {}
        for i, agent_name in enumerate(self.agent_names):
            if agent_name in obs.keys():
                timeSeperation = info[agent_name]["time_separation"] if "time_separation" in info[agent_name] else np.inf
                seperation = np.exp(-0.5 * timeSeperation)
                acceleration = np.linalg.norm(
                    obs[agent_name]["ego_vehicle_state"]["linear_acceleration"]
                )
                jerk = np.linalg.norm(
                    obs[agent_name]["ego_vehicle_state"]["linear_jerk"]
                )
                reward[agent_name] = 0
                if obs[agent_name]["events"]["not_moving"] or env_reward[agent_name] < 0.01:
                    reward[agent_name] -= k
                elif obs[agent_name]["events"]["collisions"] \
                    or obs[agent_name]["events"]["off_route"] \
                    or obs[agent_name]["events"]["off_road"] \
                    or obs[agent_name]["events"]["on_shoulder"] \
                    or obs[agent_name]["events"]["wrong_way"]:
                    reward[agent_name] -= 10 * k
                elif obs[agent_name]["events"]["reached_goal"]:
                    reward[agent_name] += 10 * k
                else:
                    reward[agent_name] += lx*env_reward[agent_name]
                    lat_error = self.get_lateral_error(obs[agent_name])
                    if lat_error > 0.1:
                        reward[agent_name] -= lat_error
                    
                    # reward[agent_name] -= la * acceleration + lj * jerk + lt * seperation
                    # reward[w] -= lt*seperation

        return reward
