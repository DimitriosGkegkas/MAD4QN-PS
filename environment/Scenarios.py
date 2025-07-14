import gymnasium as gym
from smarts.core.scenario import Scenario as SMARTSScenario  # avoid name conflict
from environment.help_scenario import get_scenario_missions, scenarios_per_number_of_agents, scenario_weights_per_number_of_agents
import random
import numpy as np
from typing import Any
import gymnasium as gym
import numpy as np
from pathlib import Path
import warnings
from smarts.core.utils.core_math import (
    combination_pairs_with_unique_indices,
)
from itertools import product
from typing import (
    Sequence,
)
import os


class Scenarios(gym.Wrapper):
    def __init__(self, env, agent_names, scenario_path, traffic_base_path = None, dynamic_scenarios=False):
        super().__init__(env)
        self.evaluation_scenario = -1
        self.agent_names = agent_names
        self.dynamic_scenarios = dynamic_scenarios
        self.scenario_path = str(scenario_path[0])
        self.probs = np.ones(len(scenarios_per_number_of_agents[4]))
        
        if traffic_base_path is not None:
            self.traffic_path = [
                os.path.join(traffic_base_path, d, "basic.rou.xml")
                for d in os.listdir(traffic_base_path)
                if os.path.isdir(os.path.join(traffic_base_path, d)) and d.isdigit()
            ]
        

     
    def modify_probs(self, probs):
        self.probs = probs

    def set_scenario(self, scenario_index):
        self.evaluation_scenario = scenario_index
        
    def _sample_scenario_index(self):
        # choose number of agents based randomly
        num_agents = random.choices([1, 2, 3, 4], weights=[1, 2, 3, 16])[0]
        
        # Choose a scenario_id using the weights
        if num_agents == 4 :
            scenario_id = random.choices(scenarios_per_number_of_agents[4], weights=self.probs)[0]
        else:
            scenario_id = random.choice(scenarios_per_number_of_agents[num_agents])
            
        print(f"Scenario ID: {scenario_id} with {num_agents} agents")
        return scenario_id
    
    
    def get_dynamic_scenario(self):
        if self.evaluation_scenario >= 0:
            missions = {
                self.agent_names[i]: mission for i, mission in enumerate(
                    get_scenario_missions(self.evaluation_scenario)
                )
            }
            self.evaluation_scenario = -1
        else:
            missions = {
                self.agent_names[i]: mission for i, mission in enumerate(
                    get_scenario_missions(self._sample_scenario_index(), randomize_start_time=True)
                )
            }

        # Create a new Scenario instance with selected missions
        scenario = SMARTSScenario(
            self.scenario_path,
            missions=missions
        )
        return scenario

    def get_static_scenario(self):
        if self.evaluation_scenario >= 0:
            traffic_specs = [self.traffic_path[self.evaluation_scenario]]
            self.evaluation_scenario = -1
        else:
            traffic_specs = [np.random.choice(self.traffic_path)]
            
        scenario = SMARTSScenario(
            self.scenario_path,
            traffic_specs=traffic_specs,
        )
            
        return scenario
    


    def reset(self, seed=None):
        if self.dynamic_scenarios:
            scenario = self.get_dynamic_scenario()
        else:
            scenario = self.get_static_scenario()
        return self.env.reset(
            seed=seed,
            options={"scenario": scenario},
        )


    def step(self, *args, **kwargs) -> tuple:
        return self.env.step(*args, **kwargs)