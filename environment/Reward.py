import gymnasium as gym
import numpy as np
from utils import get_lateral_error



class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_names=None, gains= {"la": 0.1, "lj": 0.05, "lt": 2, "lx": 1, "k": 1, "lat": 1}):
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
        self.la = gains.get("la", 0.1)  # Linear acceleration gain
        self.lj = gains.get("lj", 0.05)  # Jerk
        self.lt = gains.get("lt", 2)     # Time separation gain
        self.lx = gains.get("lx", 1)     # Lateral error gain
        self.k = gains.get("k", 1)       # Penalty for not moving
        self.lat = gains.get("lat", 1)  # Lateral error gain
        

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
                    reward[agent_name] -= self.k
                elif obs[agent_name]["events"]["reached_goal"]:
                    reward[agent_name] += 10 * self.k
                elif obs[agent_name]["events"]["collisions"] \
                    or obs[agent_name]["events"]["off_route"] \
                    or obs[agent_name]["events"]["off_road"] \
                    or obs[agent_name]["events"]["on_shoulder"] \
                    or obs[agent_name]["events"]["wrong_way"]:
                    reward[agent_name] -= 10 * self.k
                else:
                    reward[agent_name] += self.lx*env_reward[agent_name]
                    reward[agent_name] -= self.lat*get_lateral_error(obs[agent_name])
                     
                    # reward[agent_name] -= la * acceleration + lj * jerk + lt * seperation
                    # reward[w] -= lt*seperation

        return reward
