from typing import Any
import gymnasium as gym
import numpy as np
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
            self.evaluation_scenario = -1
        else:
            scenario = np.random.choice(self._scenarios_iterator, p=self.scenarios_probs)
        return self.env.reset(seed = seed, options={"scenario": scenario})

       