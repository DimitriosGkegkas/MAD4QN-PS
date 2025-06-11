from typing import Any
from datetime import datetime
import numpy as np
from train.AgentManager import AgentManager, AgentConfig
from train.EnvironmentManager import EnvironmentManager
from train.Evaluator import Evaluator
from train.EpisodeManager import EpisodeManager
from train.BaseTrainer import BaseTrainer
from train.Trainer import Trainer
from train.StatisticsCollector import StatisticsCollector
from statistics.experiment_data_collector import ExperimentDataCollector
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    total_steps: int = int(1e6)
    agent_count: int = 4
    algorithm_identifier: str = 'DuelingDDQNAgents'
    scenario_subdir: str = "scenarios/sumo/multi_scenario"
    evaluation_step: int = 10
    evaluate: bool = False
    num_env: int = 1
    agent_spec: Any = None  # Define a custom type for agent_spec if possible
    seed: int = 42
    tensorboard: bool = True

class MultiAgentTrainerParallel:
    def __init__(
        self,
        config: TrainerConfig
    ):
        self.config = config

        self.env_manager = EnvironmentManager(
            agent_count=config.agent_count,
            agent_spec=config.agent_spec,
            scenario_subdir=config.scenario_subdir,
            parallel=config.evaluate,
            num_env=config.num_env,
            seed=config.seed
        )
        agent_names = self.env_manager.get_agent_names()
        self.episode_manager = EpisodeManager(agent_names)
        self.logger = BaseTrainer( algorithm_identifier = config.algorithm_identifier, enable_tensorboard=config.tensorboard, evaluate=config.evaluate,)
        
        agent_config = AgentConfig(
            input_dim=self.env_manager.env.observation_space.shape,
            n_actions=self.env_manager.env.action_space.shape[0],
            gamma=0.99,
            lr=1e-4,
            tau=1e-3,
            batch_size=64,
            mem_size_factor=1.5
        )
        self.agent_manager = AgentManager(
            agent_names=agent_names,
            algorithm_identifier=config.algorithm_identifier,
            evaluate=config.evaluate,
            agent_config=agent_config
        )
        
        
        self.evaluator = Evaluator(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            evaluation_step=config.evaluation_step,
            total_scenarios=len(self.env_manager.get_scenarios())
            # checkpoint_enabled=config.evaluate  # Enable checkpoints only if not evaluating
        )
        
        self.trainer = Trainer(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            config=config
        )
        # self.stats_collector = StatisticsCollector(agent_names)        

    def preload(self, path: str) -> None:
        self.agent_manager.load(path, evaluate=self.evaluate)
        
    def envision(self, scenario_id: int ) -> None:
        self.evaluator.envision(scenario_id)
        
    def train(self) -> None:
        self.trainer.train()
        
    
    # def collect_statistics(self, parallel: bool = True) -> None:
    #     collector = ExperimentDataCollector(self.algorithm_identifier)
    #     self.evaluator.collect_statistics(collector, parallel=parallel)
        
