from email import message
from typing import Optional, List, Dict, Any, Union
import torch
import pathlib
import numpy as np
from environment import make_env, make_env_parallel



class EnvironmentManager:
    def __init__(
        self,
        agent_count: int,
        agent_spec: Any,  # Define a custom type for agent_spec if possible
        scenario_subdir: str = "environment/scenarios/multi_scenario",
        parallel: bool = True,
        evaluate: bool = False,
        envision: bool = False,
        num_env: int = 1,
        seed: int = 42,
        stack_frames: int = 4,
        observation_shape: tuple = (32, 32, 3),
        message_dim: int = 8,
        message_raw_dim: int = 8
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
            scenarios_path
        ]
        self.scenarios.sort()

        if parallel:
            self.env = make_env_parallel(
                "smarts.env:hiway-v1",
                agent_interfaces,
                self.scenarios, not envision,
                seed, num_env=self.num_env,
                stack_frames=stack_frames, 
                observation_shape=observation_shape,
                message_dim=message_dim,
                message_raw_dim=message_raw_dim,
            )
        else:
            self.env = make_env(
                "smarts.env:hiway-v1",
                agent_interfaces,
                self.scenarios,
                headless=not envision,
                seed=seed,
                stack_frames=stack_frames,
                observation_shape=observation_shape,
                message_dim=message_dim,
                message_raw_dim=message_raw_dim
            )


    def reset(self, scenario_ids: Optional[Union[List[int], int]] = None) -> Any:
        if scenario_ids is not None:
            self.env.set_scenario(scenario_ids)
        return self.env.reset()


    def step(self, *args, **kwargs) -> Any:
        return self.env.step(*args, **kwargs)

    def get_scenarios(self) -> List[str]:
        return self.scenarios

    def get_agent_names(self) -> List[str]:
        return self.agent_names
