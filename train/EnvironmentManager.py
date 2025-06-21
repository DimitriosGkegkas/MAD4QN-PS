from email import message
from typing import Optional, List, Dict, Any, Union
from omegaconf import DictConfig
import torch
import pathlib
import numpy as np
from environment import make_env, make_env_parallel
from smarts.core.agent_interface import AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType


class EnvironmentManager:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.agent_count = cfg.agent_count
        self.num_env = cfg.num_env
        self.scenario_subdir = cfg.scenario_subdir

        self.agent_names: List[str] = [f"Agent-{i}" for i in range(self.agent_count)]
        self.scenarios: List[str] = []
        
        agent_spec = AgentSpec(
            interface=AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.RawThrottle,
                max_episode_steps=None, 
                top_down_rgb=True
            ),
        )

        agent_interfaces: Dict[str, Any] = {
            agent_id: agent_spec.interface for agent_id in self.agent_names
        }

        scenarios_path = pathlib.Path(__file__).absolute().parent.parent / self.scenario_subdir
        traffic_path = pathlib.Path(__file__).absolute().parent.parent / cfg.traffic_base_path
        
        self.scenarios = [
            str(scenarios_path)
        ]

        if cfg.parallel:
            self.env = make_env_parallel(
                "smarts.env:hiway-v1",
                agent_interfaces,
                self.scenarios, 
                not cfg.envision,
                num_env=self.num_env,
                seed=cfg.seed,
                stack_frames=cfg.stack_frames, 
                observation_shape=cfg.observation_shape,
                message_dim=cfg.message_dim,
                message_raw_dim=cfg.feature_dim + cfg.action_dim + cfg.direction_dim,
                dynamic_scenarios=cfg.dynamic_scenarios,
                traffic_base_path=traffic_path
            )
        else:
            self.env = make_env(
                "smarts.env:hiway-v1",
                agent_interfaces,
                self.scenarios,
                not cfg.envision,
                seed=cfg.seed,
                stack_frames=cfg.stack_frames, 
                observation_shape=cfg.observation_shape,
                message_dim=cfg.message_dim,
                message_raw_dim=cfg.feature_dim + cfg.action_dim + cfg.direction_dim,
                dynamic_scenarios=cfg.dynamic_scenarios,
                traffic_base_path=traffic_path
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
