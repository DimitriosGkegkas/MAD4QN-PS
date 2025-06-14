from math import gamma
from typing import Any
from datetime import datetime
import numpy as np
from Agent.agent import AgentConfig
from train.BaseTrainer import BaseTrainer
from train.AgentManager import AgentManager
from train.EnvironmentManager import EnvironmentManager
from train.Evaluator import Evaluator
from train.EpisodeManager import EpisodeManager
from train.Trainer import Trainer
from train.StatisticsCollector import StatisticsCollector
from statistics.experiment_data_collector import ExperimentDataCollector
from dataclasses import dataclass
from smarts.core.agent_interface import AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

@dataclass
class TrainerConfig:
    algorithm_identifier: str
    
    # Training parameters
    total_steps: int = int(1e6)
    mem_size_factor: float = 1.5
    num_env: int = 1 
    stack_frames: int = 4
    evaluation_step: int = 10
    max_training_steps: int = 1000  # Maximum steps per episode, can be adjusted based on the environment
    max_evaluation_steps: int = 1000  # Maximum steps per evaluation episode, can be adjusted based on the environment
    
    
    # Agent Learning
    seed: int = 42
    agent_spec: Any = AgentSpec(
            interface=AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.RawThrottle,
                max_episode_steps=None, 
                top_down_rgb=True
            ),
        )
    agent_count: int = 4
    
    observation_shape: tuple = (256, 256, 3)  # Shape of the observation space, can be adjusted based on the environment
    
    # Scenarios
    scenario_subdir: str = "scenarios/sumo/multi_scenario"
    
    # Logging and Evaluation
    evaluate: bool = False
    tensorboard: bool = True
    envision: bool = False
    parallel: bool = True  # If True, use parallel environments for training
    

class MultiAgentTrainerParallel:
    def __init__(
        self,
        config: TrainerConfig,
        agent_config: AgentConfig
    ):
        self.evaluate = config.evaluate

        self.env_manager = EnvironmentManager(
            agent_count=config.agent_count,
            agent_spec=config.agent_spec,
            scenario_subdir=config.scenario_subdir,
            parallel=config.parallel,
            num_env=config.num_env,
            seed=config.seed,
            stack_frames=config.stack_frames,
            envision=config.envision,
            evaluate=config.evaluate,  # If True, use parallel environments for evaluation
            observation_shape=config.observation_shape,
        )
        agent_names = self.env_manager.get_agent_names()
        self.episode_manager = EpisodeManager(agent_names, parallel=config.parallel)
        self.logger = BaseTrainer( algorithm_identifier = config.algorithm_identifier, enable_tensorboard=config.tensorboard, evaluate=config.evaluate,)
        
        agent_config.input_dim = (config.observation_shape[2] * config.stack_frames, config.observation_shape[0], config.observation_shape[1])
        agent_config.chkpt_dir = f"models/{config.algorithm_identifier}"
        self.agent_manager = AgentManager(
            agent_names=agent_names,
            agent_config=agent_config,
            evaluate=config.evaluate,
            parallel=config.parallel,
        )
        
        
        self.evaluator = Evaluator(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            evaluation_step=config.evaluation_step,
            max_evaluation_steps=config.max_evaluation_steps,
            total_scenarios=len(self.env_manager.get_scenarios())
        )
        
        self.trainer = Trainer(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            evaluator=self.evaluator,
            env_manager=self.env_manager,
            total_steps = config.total_steps,
            agent_count = config.agent_count,
            algorithm_identifier = config.algorithm_identifier,
            evaluation_step = config.evaluation_step,
            max_training_steps = config.max_training_steps
        )      

    def preload(self, path: str) -> None:
        self.agent_manager.load(path, evaluate=self.evaluate)
        
    def envision(self, scenario_id: int ) -> None:
        self.evaluator.envision(scenario_id)
        
    def train(self) -> None:
        self.trainer.train()
        
    
    # def collect_statistics(self, parallel: bool = True) -> None:
    #     collector = ExperimentDataCollector(self.algorithm_identifier)
    #     self.evaluator.collect_statistics(collector, parallel=parallel)
        
