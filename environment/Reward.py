import gymnasium as gym
import numpy as np
from smarts.core.sensor import AccelerometerSensor


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

    def reset(self, **kwargs):
        """Resets the environment."""
        obs, info = self.env.reset(**kwargs)
        info = self._add_social_traffic_info(info)
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
        wrapped_reward = self._compute_reward(obs, reward)
        info = self._add_social_traffic_info(info)
        return obs, wrapped_reward, terminated, truncated, info

    def _add_social_traffic_info(self, info: dict) -> dict:
        """
        Adds social traffic information to the info dictionary.

        Args:
            info (dict): The original info dictionary.

        Returns:
            dict: The updated info dictionary with social traffic data.
        """
        info['social_traffic'] = []
        for vehicle in self.env.env.smarts.vehicle_index.vehicles:
            # Attach accelerometer sensor if not already attached
            if not vehicle.subscribed_to_accelerometer_sensor:
                vehicle.attach_sensor(AccelerometerSensor(), "accelerometer_sensor")
            
            # Skip vehicles subscribed to RGB sensor
            if vehicle.subscribed_to_rgb_sensor:
                continue

            # Calculate accelerations and jerks
            linear_acc, angular_acc, linear_jerk, angular_jerk = vehicle.accelerometer_sensor(
                vehicle.state.linear_velocity,
                vehicle.state.angular_velocity,
                self.env.env.smarts.last_dt,
            )

            # Add vehicle traffic info
            info['social_traffic'].append({
                'id': vehicle.id,
                'speed': vehicle.speed,
                'linear_acceleration': linear_acc,
                'dt': self.env.env.smarts.last_dt,
                'travel_distance': 0,  # Placeholder for missing info
            })

        return info

    def _compute_reward(self, obs: dict, env_reward: dict) -> np.ndarray:
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

                if obs[agent_name]["events"]["not_moving"]:
                    reward[w] -= 1
                elif obs[agent_name]["events"]["collisions"]:
                    reward[w] -= 10
                elif obs[agent_name]["events"]["reached_goal"]:
                    reward[w] += 10
                else:
                    reward[w] += env_reward[agent_name]

                w += 1

        return np.float64(reward)
