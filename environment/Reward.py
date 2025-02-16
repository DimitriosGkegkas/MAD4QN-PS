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
        self.env = env

    def reset(self, **kwargs):
        """Resets the environment."""
        obs, info = self.env.reset(**kwargs)
        info = self._add_social_traffic_info(info)
        return obs, info

    def step(self, action, conflicts):
        """
        Steps through the environment.

        Args:
            action: The actions to be performed.

        Returns:
            Tuple: Observation, wrapped reward, termination flags, truncation flags, and info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._add_social_traffic_info(info)
        info = self._add_passed_intersection_info(info)
        wrapped_reward = self._compute_reward(obs, reward, info, conflicts)
        return obs, wrapped_reward, terminated, truncated, info
    
    def _add_passed_intersection_info(self, info: dict) -> dict:
        for agent_name in self.agent_names:
            if agent_name in info:
                info[agent_name]['passed_intersection'] = self._check_if_agent_passed_intersection(agent_name, info)
        return info
    
    def _check_if_agent_passed_intersection(self, agent_name: str, info: dict) -> bool:
        """
        Checks if the specified agent passed an intersection.

        Args:
            agent_name (str): The name of the agent.
            info (dict): The original info dictionary.

        Returns:
            bool: True if the agent passed the intersection, False otherwise.
        """
        road_id = info[agent_name]["env_obs"].ego_vehicle_state.road_id
        decode = road_id.split('-')
        if len(decode)!= 3:
            return False  # Invalid road ID format "junction" is 2
        return decode[1][0].upper() == decode[2][-1].upper()

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
    
    def _get_time_to_intersection(self, info):
        """
        Computes the time to the nearest intersection for each agent.

        Args:
            info (dict): The info dictionary containing social traffic data.
    
        Returns:
            np.ndarray: The time to the nearest intersection for each agent.
        """
        time_to_intersection = {}
        for agent_name in self.agent_names:
            if agent_name in info:
                position = info[agent_name]["env_obs"].ego_vehicle_state.position
                # ditance to  38.54,39.02
                distance = np.sqrt((position[0] - 38.54) ** 2 + (position[1] - 39.02) ** 2)
                velocity = info[agent_name]["env_obs"].ego_vehicle_state.speed
                time_to_intersection[agent_name] = (np.float(distance / velocity if velocity > 0.01 else np.inf), distance)
        return time_to_intersection
    
    def _get_reward_based_on_time_to_intersection(self, info, conflicts):
        """
        Computes the reward for each agent based on their time to the nearest intersection,
        considering potential conflicts with other agents.

        Args:
            info (dict): Dictionary containing social traffic data for each agent.

        Returns:
            dict: A dictionary mapping each agent's name to their respective reward.
        """
        time_to_intersection = self._get_time_to_intersection(info)
        rewards = {}

        for agent in self.agent_names:
            if agent in info:
                # Compute the maximum time to intersection among conflicting agents
                conflicting_times = [
                    time_to_intersection[other_agent][0]
                    for other_agent in self.agent_names
                    if (
                        other_agent != agent
                        and other_agent in info
                        and other_agent in conflicts[agent]
                        and not info[other_agent]['passed_intersection']
                        and time_to_intersection[agent] > time_to_intersection[other_agent]
                    )
                ]
                if conflicting_times:
                    max_conflicting_time = np.max(conflicting_times)
                    distance = np.abs(time_to_intersection[agent][0] - max_conflicting_time)
                    
                    # Compute reward using an exponential function
                    rewards[agent] = -np.exp(-1.5 * distance) * np.exp(-0.2 * time_to_intersection[agent][1])
                else:
                    rewards[agent] = 0
        
        return rewards
    

    def _compute_reward(self, obs: dict, env_reward: dict, info: dict, conflicts) -> np.ndarray:
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
        reward_based_on_time_to_intersection = self._get_reward_based_on_time_to_intersection(info, conflicts)
        for i, agent_name in enumerate(self.agent_names):
            if agent_name in obs.keys():
                
                print(reward_based_on_time_to_intersection[agent_name])

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
