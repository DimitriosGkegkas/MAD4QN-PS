import collections
from math import e
from turtle import distance
import gymnasium as gym
import numpy as np
from sympy import E
from utils import position2road, roads2t_i, has_conflict, has_conflict_v2v
from smarts.core.sensor import AccelerometerSensor

from utils.debug import debug_save_any_img

class AgentInformationHelper():
    def __init__(self, agent_name, position, mission = None, roads = None):
        self.agent_name = agent_name
        
        self.roads = None
        self.turning_intention = None
        self.direction = None
        if mission:
            self.set_turning_intention(mission)
        if roads:
            self.roads = roads
            self.turning_intention = roads2t_i[self.roads]
        self.passed_intersection = False
        self.position = position
        self.distance_to_intersection = 100
        self.min_time_separation = np.inf
    
        
    def set_turning_intention(self, mission):
        start = position2road([mission.start.position.x, mission.start.position.y])
        goal = position2road([mission.goal.position.x, mission.goal.position.y])
        self.roads = start + goal
        self.turning_intention = roads2t_i[self.roads]
        return self.turning_intention
    
    def check_time_separation(self, time_separation):
        # if time_separation < self.min_time_separation:
        self.min_time_separation = time_separation
        return self.min_time_separation
    
    def get_distance_from(self, position):
        return np.sqrt((position[0] - self.position[0]) ** 2 + (position[1] - self.position[1]) ** 2)
    
    def get_distance_to_intersection(self):
        return self.get_distance_from([38.54,39.02])

        
    def get_direction_speed(self, vector, linear_velocity):
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0:
            vector_direction = vector / vector_norm
        else:
            vector_direction = np.zeros_like(vector)
        speed = np.dot(linear_velocity, vector_direction)
        return speed
    
    def step(self, position, linear_velocity):
        # self.direction = ego_vehicle_state.position - self.position
        self.direction = [position[0] - self.position[0], position[1] - self.position[1]]
        self.position = position
        self.velocity = linear_velocity
        _distance_to_intersection = self.get_distance_to_intersection()
        if (_distance_to_intersection > self.distance_to_intersection ) and _distance_to_intersection > 6 and _distance_to_intersection < 50:
            self.passed_intersection = True
        self.distance_to_intersection = _distance_to_intersection
        
class AgentsInformationController():
    def __init__(self, agent_names):
        self.agent_names = agent_names
        
    def reset(self, info):
        self.info = info
        self.agents = {}
        
        self.set_agents()
        self.set_social_agents()
        self.set_conflicts()
        
    def set_agents(self):
        for agent_name in self.agent_names:
            if agent_name in self.info:
                ego = self.info[agent_name]["env_obs"]
                self.agents[agent_name] = AgentInformationHelper(agent_name, ego.ego_vehicle_state.position, ego[5].mission)
        return self.agents
    
    def extract_direction_code(self, id_string):
        # Split the string by "-edge-"
        parts = id_string.split("-edge-")
        
        # Ensure the expected structure exists
        if len(parts) < 3:
            raise Exception(f"Unexpected ID format: {id_string}")
        
        # The starting part (after the first occurrence)
        start_section = parts[1]
        # The ending part (after the second occurrence)
        end_section = parts[2]
        
        # Split these sections by "-" and take the first token as the direction.
        start_direction = start_section.split("-")[0]
        end_direction = end_section.split("-")[0]
        
        # Format the result as the first letter of each, uppercase.
        return start_direction[0].upper() + end_direction[0].upper()



    def set_social_agents(self):
        if self.info['social_traffic'] is not None:
            for agent in self.info['social_traffic']:
                self.agents[agent["id"]] = AgentInformationHelper(agent["id"], agent["position"], roads = self.extract_direction_code(agent["id"]))
        return self.agents
    
    def step(self, info):
        self.update_agents(info)
        self.update_social_agents(info)
        self.get_confidence()
        
    def find_intersection(self, A, d1, B, d2):
        def cross(v, w):
            return v[0] * w[1] - v[1] * w[0]
        
        # Compute the cross product of the direction vectors.
        r_cross_s = cross(d1, d2)
        
        # If the cross product is zero, the lines are parallel or collinear.
        if abs(r_cross_s) < 1e-3:
            return None  # No unique intersection exists.
        
        # Compute the vector from A to B.
        B_minus_A = [B[0] - A[0], B[1] - A[1]]
        
        # Compute parameter t for the line starting at A in direction d1.
        t = cross(B_minus_A, d2) / r_cross_s
        # Compute parameter u for the line starting at B in direction d2.
        u = cross(B_minus_A, d1) / r_cross_s
        
        # Check if the intersection point is in the forward direction for both rays.
        if t < -1 or u < -1:
            return None  # Intersection not in the direction of the vectors.
        
        # Calculate the intersection point.
        intersection = [A[0] + t * d1[0], A[1] + t * d1[1]]
        return intersection

    
    def get_time_seperation(self, ego, other):
        e = 0.0001
        if (ego.direction == None) or (other.direction == None):
            return np.inf, np.inf
        intersection_point = self.find_intersection(ego.position, ego.direction, other.position, other.direction)
        
        if intersection_point is None:
            return np.inf, np.inf
        dOther = np.linalg.norm(np.array(intersection_point) - np.array([other.position[0], other.position[1]])) - 3.5
        
        if dOther < 0:
            dEgo = max(np.linalg.norm(np.array(intersection_point) - np.array([ego.position[0], ego.position[1]])) - 3.5,0)
            return dEgo, dEgo / (np.linalg.norm(ego.velocity[0]) + e)
    
        else:
            return np.inf, np.inf
    
        
    
    def get_confidence(self):
        for ego in self.agents:
            if not self.agents[ego].passed_intersection:
                for conflict in self.conflicts[ego]:
                    if not self.agents[conflict].passed_intersection:
                        distance, time_separation = self.get_time_seperation(self.agents[ego], self.agents[conflict])
                        self.agents[ego].check_time_separation(time_separation)
        return
                

    
    def update_agents(self, info):
        for agent_name in self.agent_names:
            if agent_name in info:
                ego = info[agent_name]["env_obs"].ego_vehicle_state
                self.agents[agent_name].step(ego.position, ego.linear_velocity)
        return 
    
    def update_social_agents(self, info):
        if(info['social_traffic'] is not None):
            for agent in info['social_traffic']:
                if agent['id'] in self.agents:
                    self.agents[agent['id']].step(agent["position"], agent["linear_velocity"])
                else:
                    self.agents[agent['id']] = AgentInformationHelper(agent['id'], agent['position'], roads=self.extract_direction_code(agent['id']))
                    self.set_conflicts()
        return 
    
    def get_turning_intention(self):
        turning_intentions = {}
        for agent_name in self.agent_names:
            if agent_name in self.info:
                turning_intentions[agent_name] = self.agents[agent_name].turning_intention
        return turning_intentions
    
    def set_conflicts(self):
        self.conflicts = {}
        for ego in self.agents:
            self.conflicts[ego] = []
            for other_agent in self.agents:
                if (other_agent != ego):
                    if (has_conflict_v2v(self.agents[ego].roads, self.agents[other_agent].roads)):
                        self.conflicts[ego] = self.conflicts.get(ego, []) + [other_agent]
        return self.conflicts
    
    def get_conflicts(self):
        return self.conflicts
    
    def get_min_time_separation(self):
        min_time_separation = {}
        for agent in self.agents:
            min_time_separation[agent] = self.agents[agent].min_time_separation
        return min_time_separation
        
        


class InfoWrapper(gym.Wrapper):
    def __init__(self, env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super(InfoWrapper, self).__init__(env)
        self.agent_names = agent_names
        self.env = env
        self.agents_information_controller = AgentsInformationController(self.agent_names)
    def reset(self,
         **kwargs
    ):
        observation, info = self.env.reset(**kwargs)
        info = self._add_social_traffic_info(info)
        self.agents_information_controller.reset(info)
        return observation, info
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

            info['social_traffic'].append({
                'id': vehicle.id,
                'speed': vehicle.speed,
                'linear_velocity': vehicle.state.linear_velocity,
                'linear_jerk': linear_jerk,
                'linear_acceleration': linear_acc,
                'position': vehicle.position,
                'dt': self.env.env.smarts.last_dt,
                'travel_distance': 0,  # Placeholder for missing info
            })

        return info
    
    def add_time_separation(self, info):
        self.agents_information_controller.step(info)
        time_separation = self.agents_information_controller.get_min_time_separation()
        if 'social_traffic' in info:
            for agent in info['social_traffic']:
                if agent['id'] in time_separation:
                    agent['time_separation'] = time_separation[agent['id']]
        for agent_name in self.agent_names:
            if agent_name in info:
                info[agent_name]['time_separation'] = time_separation[agent_name]
        return info
        
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._add_social_traffic_info(info)
        info = self.add_time_separation(info)
        return obs, reward, terminated, truncated, info 
    

    




