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
        self.stack = {}  # agent_id -> deque
        self.repeat = repeat
        self.agent_names = agent_names

    def step(self, *args, **kwargs) -> tuple:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        return self.observation(observation), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.stack = {}
        
        observation, termination, truncation, reward, infos = self.env.reset(**kwargs)
        return self.observation(observation), termination, truncation, reward, infos


    def observation(self, observation):
        obs_dict = {}

        for agent_id, obs in observation.items():
            if agent_id not in self.stack:
                # New agent → initialize deque with repeated current obs
                self.stack[agent_id] = collections.deque([obs] * self.repeat, maxlen=self.repeat)
            else:
                self.stack[agent_id].append(obs)

            obs_dict[agent_id] = np.array(self.stack[agent_id]).reshape(self.observation_space.low.shape)

        return obs_dict


