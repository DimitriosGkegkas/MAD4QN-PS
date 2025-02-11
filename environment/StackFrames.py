import collections
import gymnasium as gym
import numpy as np

from utils.debug import debug_save_any_img


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = [collections.deque(maxlen=repeat), collections.deque(maxlen=repeat),
                      collections.deque(maxlen=repeat), collections.deque(maxlen=repeat)]
        self.agent_names = agent_names
    
    def reset(self,
        *,
        seed = None,
        options = None,
              ):
        self.stack[0].clear()
        self.stack[1].clear()
        self.stack[2].clear()
        self.stack[3].clear()

        observation, info = self.env.reset(seed=seed, options=options)
        for i, j in enumerate(self.agent_names):
            for _ in range(self.stack[i].maxlen):
                self.stack[i].append(observation[j])
        obs_dict = {}

        for i, j in enumerate(self.agent_names):
            obs_dict[j] = np.array(self.stack[i]).reshape(self.observation_space.low.shape)
        
        return self.observation(observation), info 

    def observation(self, observation):
        obs_dict = {}

        for i, j in enumerate(self.agent_names):
            if j in observation.keys():
                self.stack[i].append(observation[j])
                obs_dict[j] = np.array(self.stack[i]).reshape(self.observation_space.low.shape)

        return obs_dict



