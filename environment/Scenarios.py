import gymnasium as gym
from smarts.core.scenario import Scenario as SMARTSScenario  # avoid name conflict
from environment.help_scenario import get_scenario_missions, scenarios_per_number_of_agents
import random
import numpy as np

class Scenarios(gym.Wrapper):
    def __init__(self, env, agent_names, scenario_path):
        super().__init__(env)
        self.evaluation_scenario = -1
        self.agent_names = agent_names
        self.scenario_path = str(scenario_path[0])
        self.n_episodes = 0

     
    def modify_probs(self, n_episodes):
        self.n_episodes = n_episodes

    def set_scenario(self, scenario_index):
        self.evaluation_scenario = scenario_index
        
    def _sample_scenario_index(self):
        agent_count = 1
        
        if self.n_episodes > 10:
            agent_count = 1
        elif self.n_episodes > 30:
            agent_count = 2
        elif self.n_episodes > 70:
            agent_count = 3
        elif self.n_episodes > 150:
            agent_count = 4
        scenario_id = random.choice(scenarios_per_number_of_agents[agent_count])
        print(f"Scenario {scenario_id} selected for episode {self.n_episodes}")
        return scenario_id

    def reset(self, seed=None):
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
                    get_scenario_missions(self._sample_scenario_index())
                )
            }

        # Create a new Scenario instance with selected missions
        scenario = SMARTSScenario(
            self.scenario_path,
            missions=missions
        )

        return self.env.reset(
            seed=seed,
            options={"scenario": scenario},
        )


    def step(self, *args, **kwargs) -> tuple:
        return self.env.step(*args, **kwargs)