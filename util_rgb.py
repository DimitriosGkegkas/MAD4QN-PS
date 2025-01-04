import collections
from typing import Any
import gymnasium as gym
import numpy as np
import cv2
from smarts.env.configs.hiway_env_configs import ScenarioOrder
from smarts.core.sensor import AccelerometerSensor
from smarts.core.scenario import Scenario
from pathlib import Path
import warnings
from smarts.core.utils.core_math import (
    combination_pairs_with_unique_indices,
)
from itertools import product
from typing import (
    Sequence,
)

roads2t_i = {'EW': 'straight', 'ES': 'left', 'EN': 'right',
             'WE': 'straight', 'WN': 'left', 'WS': 'right',
             'SN': 'straight', 'SW': 'left', 'SE': 'right',
             'NS': 'straight', 'NE': 'left', 'NW': 'right'}


def position2road(position):
    if position[0] > 110:
        return 'E'
    elif position[0] < 80:
        return 'W'
    elif position[1] < 90:
        return 'S'
    else:
        return 'N'


class Reward(gym.Wrapper):
    def __init__(self, env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super().__init__(env)
        self.agent_names = agent_names

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        return obs, wrapped_reward, terminated, truncated, self._add_social_traffic_info(info)
    
    def _add_social_traffic_info(self, info):
        info['social_traffic'] = []
        for vehicle in self.env.env.smarts.vehicle_index.vehicles:
            if not vehicle.subscribed_to_accelerometer_sensor:
                vehicle.attach_sensor(AccelerometerSensor(), "accelerometer_sensor")
            if vehicle.subscribed_to_rgb_sensor:
                # TODO Find a better way to do this
                continue
            linear_acc, angular_acc, linear_jerk, angular_jerk = vehicle.accelerometer_sensor(
                vehicle.state.linear_velocity, 
                vehicle.state.angular_velocity, 
                self.env.env.smarts.last_dt
                )
            info['social_traffic'].append({
                'id': vehicle.id,
                'speed': vehicle.speed,
                'linear_acceleration': linear_acc,
                'dt': self.env.env.smarts.last_dt,
                'travel_distance': 0, # I do not have the info
            })
        return info

    def _reward(self, obs, env_reward):
        num_vehs = len(obs.keys())
        reward = [0 for _ in range(num_vehs)]
        w = 0
        for i, j in enumerate(self.agent_names):
            if j in obs.keys():

                if obs[j]["events"]["not_moving"]:
                    reward[w] -= 1
                elif obs[j]["events"]["collisions"]:
                    reward[w] -= 10
                elif obs[j]["events"]["reached_goal"]:
                    reward[w] += 10
                else:
                    reward[w] += env_reward[j]

                w += 1

        return np.float64(reward)


class Observation(gym.ObservationWrapper):
    def __init__(self, shape, env: gym.Env, agent_names=['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )
        self.agent_names = agent_names

    def observation(self, obs):
        new_obs = {}
        for i in self.agent_names:
            if i in obs.keys():
                new_obs[i] = obs[i]["top_down_rgb"]
                new_obs[i] = cv2.resize(new_obs[i], self.shape[1:], interpolation=cv2.INTER_AREA)
                new_obs[i] = np.array(new_obs[i], dtype=np.uint8).reshape(self.shape)
                new_obs[i] = new_obs[i] / 255.0
        return new_obs
    
class Scenarios(gym.Wrapper):
    def __init__(self, env, agent_names, scenario_path):
        super().__init__(env)
        self.evaluation_scenario = -1
        self._scenarios_iterator = self.new_scenario_variations(
            [str(Path(scenario).resolve()) for scenario in scenario_path],
            agent_names
        )
        
        self.scenarios_probs = [1/len(self._scenarios_iterator) for _ in range(len(self._scenarios_iterator))]

    def new_scenario_variations(
        self,
        scenarios_or_scenarios_dirs: Sequence[str],
        agents_to_be_briefed: Sequence[str],
        ) -> Sequence["Scenario"]:

        scenario_roots = Scenario.get_scenario_list(scenarios_or_scenarios_dirs)
        final_list = []

        for scenario_root in scenario_roots:
            surface_patches = Scenario.discover_friction_map(scenario_root)

            agent_missions = Scenario.discover_agent_missions(
                scenario_root, agents_to_be_briefed
            )

            social_agent_infos = Scenario._discover_social_agents_info(scenario_root)
            social_agents = [
                {
                    agent_id: (agent.to_agent_spec(), (agent, mission))
                    for agent_id, (
                        agent,
                        mission,
                    ) in per_episode_social_agent_infos.items()
                }
                for per_episode_social_agent_infos in social_agent_infos
            ]

            # `or [None]` so that product(...) will not return an empty result
            # but insted a [(..., `None`), ...].
            agent_missions = agent_missions or [None]
            if len(agents_to_be_briefed) > len(agent_missions):
                warnings.warn(
                    f"Scenario `{scenario_root}` has {len(agent_missions)} missions and"
                    f" but there are {len(agents_to_be_briefed)} agents to assign"
                    " missions to. The missions will be padded with random missions."
                )
            mission_agent_groups = combination_pairs_with_unique_indices(
                agents_to_be_briefed, agent_missions
            )
            social_agents = social_agents or [None]
            traffic_histories = Scenario.discover_traffic_histories(scenario_root) or [
                None
            ]
            traffic = Scenario.discover_traffic(scenario_root) or [[]]

            roll_traffic = 0
            roll_social_agents = 0
            roll_traffic_histories = 0

            for (
                concrete_traffic,
                concrete_agent_missions,
                concrete_social_agents,
                concrete_traffic_history,
            ) in product(
                np.roll(traffic, roll_traffic, 0),
                mission_agent_groups,
                np.roll(social_agents, roll_social_agents, 0),
                np.roll(traffic_histories, roll_traffic_histories, 0),
            ):
                concrete_social_agent_missions = {
                    agent_id: mission
                    for agent_id, (_, (_, mission)) in (
                        concrete_social_agents or {}
                    ).items()
                }

                # Filter out mission
                concrete_social_agents = {
                    agent_id: (_agent_spec, social_agent)
                    for agent_id, (_agent_spec, (social_agent, _)) in (
                        concrete_social_agents or {}
                    ).items()
                }

            final_list.append(Scenario(
                scenario_root,
                traffic_specs=concrete_traffic,
                missions={
                    **{a_id: mission for a_id, mission in concrete_agent_missions},
                    **concrete_social_agent_missions,
                },
                social_agents=concrete_social_agents,
                surface_patches=surface_patches,
                traffic_history=concrete_traffic_history,
            ))
        return final_list

    def modify_probs(self, returns_list):
        min_return = np.abs(min(returns_list))
        returns_list = np.array(returns_list) + 3*min_return
        returns_list = np.reciprocal(returns_list)
        self.scenarios_probs = returns_list / np.sum(returns_list)

    def set_scenario(self, scenario_index):
        self.evaluation_scenario = scenario_index

    def reset(self, seed=None):
        if self.evaluation_scenario >= 0:
            
            scenario = self._scenarios_iterator[self.evaluation_scenario]
        else:
            scenario = np.random.choice(self._scenarios_iterator, p=self.scenarios_probs)
        return super().reset(seed = seed, options={"scenarios": [scenario]})

        

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


def make_env(env_name, agent_interfaces, scenario_path, headless, seed, visdom = False) -> gym.Env:
    # Create environment
    env = gym.make(
        env_name,
        scenarios=scenario_path,
        agent_interfaces=agent_interfaces,
        headless=headless,  # If False, enables Envision display.
        visdom=visdom,  # If True, enables Visdom display.
        seed=seed,
        scenarios_order = ScenarioOrder.sequential
    )
    agent_names = agent_interfaces.keys()

    env = Reward(env=env, agent_names=agent_names)
    env = Observation(shape=(48, 48, 3), env=env, agent_names=agent_names)
    env = StackFrames(env, repeat=3, agent_names=agent_names)
    env = Scenarios(env, agent_names=agent_names, scenario_path=scenario_path)

    return env
