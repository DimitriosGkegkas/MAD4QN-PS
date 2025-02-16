import collections
import gymnasium as gym
import numpy as np
from utils import position2road, roads2t_i, has_conflict

from utils.debug import debug_save_any_img


class InfoWrapper(gym.Wrapper):
    def __init__(self, env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super(InfoWrapper, self).__init__(env)
        self.agent_names = agent_names
        self.conflicts = {}    
        self.env = env
    def reset(self,
        *,
        seed = None
    ):
        observation, info = self.env.reset(seed=seed)
        info = self.add_turning_intention(info)
        info = self.add_conflict(info)
        return observation, info
    
    def add_turning_intention(self, info):
        for k in self.agent_names:
            if k in info:
                start = position2road([info[k]['env_obs'][5].mission.start.position.x, info[k]['env_obs'][5].mission.start.position.y])
                goal = position2road([info[k]['env_obs'][5].mission.goal.position.x, info[k]['env_obs'][5].mission.goal.position.y])
                info[k]["roads"] = start + goal
                info[k]["turning_intention"] = roads2t_i[info[k]["roads"]]
        return info
    
    def step(self, action):
        return self.env.step(action, self.conflicts)
    
    def add_conflict(self, info):
        self.conflicts = {}
        for ego in self.agent_names:
            if ego in info:
                other_agents = [other_agent for other_agent in self.agent_names if (other_agent in info) and (other_agent != ego)]
                other_agents_turning_intentions = [info[other_agent]['roads'] for other_agent in other_agents]
                
                conflicts =  has_conflict(info[ego]['roads'], other_agents_turning_intentions)
                info[ego]['conflicts'] = dict(zip(other_agents, conflicts))
                self.conflicts[ego] = dict(zip(other_agents, conflicts))
        return info
    

    




