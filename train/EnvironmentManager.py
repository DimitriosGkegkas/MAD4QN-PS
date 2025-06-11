from typing import Optional, List, Dict, Any
import torch
import pathlib
import numpy as np
from environment import make_env, make_env_parallel



class EnvironmentManager:
    def __init__(
        self,
        agent_count: int,
        agent_spec: Any,  # Define a custom type for agent_spec if possible
        scenario_subdir: str = "scenarios/sumo/multi_scenario",
        parallel: bool = True,
        num_env: int = 1,
        seed: int = 42
    ):
        self.agent_count = agent_count
        self.num_env = num_env
        self.scenario_subdir = scenario_subdir

        self.agent_names: List[str] = [f"Agent-{i}" for i in range(self.agent_count)]
        self.scenarios: List[str] = []

        torch.manual_seed(seed)
        np.random.seed(seed)

        agent_interfaces: Dict[str, Any] = {
            agent_id: agent_spec.interface for agent_id in self.agent_names
        }

        scenarios_path = pathlib.Path(__file__).absolute().parent.parent / self.scenario_subdir
        self.scenarios = [
            str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()
        ]
        self.scenarios.sort()

        if parallel:
            self.env = make_env_parallel(
                "smarts.env:hiway-v1", agent_interfaces, self.scenarios, True, seed, num_env=self.num_env
            )
        else:
            self.env = make_env(
                "smarts.env:hiway-v1", agent_interfaces, self.scenarios, False, seed, True
            )


    def reset(self, scenario_ids: Optional[List[int]] | Optional[int] = None) -> Any:
        if scenario_ids is not None:
            self.env.set_scenario(scenario_ids)
        return self.env.reset()


    def step(self, actions_batch: List[Dict[str, Any]] | Dict[str, Any]) -> Any:
        return self.env.step(actions_batch)

    def get_scenarios(self) -> List[str]:
        return self.scenarios

    def get_agent_names(self) -> List[str]:
        return self.agent_names
