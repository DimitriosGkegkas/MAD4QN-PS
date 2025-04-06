import gymnasium as gym
import numpy as np
from smarts.core.sensor import AccelerometerSensor

la = 0.1
lj = 0.05
lt = 2
lx = 2
k = 2
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
        obs, info = self.env.reset(**kwargs)
        return obs, info

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
        num_vehs = len(obs.keys())
        reward = [0 for _ in range(num_vehs)]
        w = 0
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
                
                if obs[agent_name]["events"]["not_moving"]:
                    reward[w] -= k
                elif obs[agent_name]["events"]["collisions"]:
                    reward[w] -= 10 * k
                elif obs[agent_name]["events"]["reached_goal"]:
                    reward[w] += 10 * k
                else:
                    reward[w] += lx*env_reward[agent_name]
                    reward[w] -= la * acceleration + lj * jerk + lt * seperation
                w += 1

        return np.float64(reward)
