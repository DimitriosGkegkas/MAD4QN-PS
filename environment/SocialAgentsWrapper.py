import gymnasium as gym
from smarts.core.sensor import AccelerometerSensor

class SocialAgentsWrapper(gym.Wrapper):
    def __init__(self, env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super(SocialAgentsWrapper, self).__init__(env)
        self.agent_names = agent_names
        self.env = env
        
        
    def reset(self,
         **kwargs
    ):
        observation, info = self.env.reset(**kwargs)
        info = self._add_social_traffic_info(info)
        return observation, info 
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._add_social_traffic_info(info)
        return obs, reward, terminated, truncated, info 
    
    def _add_social_traffic_info(self, info: dict) -> dict:
        info['social_traffic'] = self._gather_social_vehicle_data()
        return info

    def _gather_social_vehicle_data(self):
        social_info = []
        for vehicle in self.env.env.smarts.vehicle_index.vehicles:
            if not vehicle.subscribed_to_accelerometer_sensor:
                vehicle.attach_sensor(AccelerometerSensor(), "accelerometer_sensor")
            if vehicle.subscribed_to_rgb_sensor:
                continue
            linear_acc, angular_acc, linear_jerk, angular_jerk = vehicle.accelerometer_sensor(
                vehicle.state.linear_velocity,
                vehicle.state.angular_velocity,
                self.env.env.smarts.last_dt,
            )
            social_info.append({
                'id': vehicle.id,
                'speed': vehicle.speed,
                'linear_velocity': vehicle.state.linear_velocity,
                'linear_jerk': linear_jerk,
                'linear_acceleration': linear_acc,
                'position': vehicle.position,
                'dt': self.env.env.smarts.last_dt,
                'travel_distance': 0,
            })
        return social_info




