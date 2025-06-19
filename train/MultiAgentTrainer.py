from math import gamma
from typing import Any, List
from datetime import datetime
import numpy as np
from py import log
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
from dataclasses import dataclass, field

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
    
    eval_scenarios: List[int] =  field(default_factory=lambda: [
                                # Length 1 (4)
                                0, 2, 5, 8,

                                # Length 2 (10)
                                13, 17, 21, 27, 31, 34, 39, 45, 48, 53,

                                # Length 3 (16)
                                65, 68, 72, 75, 78, 81, 85, 89,
                                92, 95, 98, 101, 105, 108, 111, 114,

                                # Length 4 (26)
                                120, 123, 126, 129, 132, 135, 138, 141, 144,
                                147, 150, 153, 156, 159, 162, 165, 168, 171,
                                174, 177, 180, 183, 186, 189, 192, 195
                            ])
                                
    
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
        
        self.logger = BaseTrainer( 
                    algorithm_identifier = config.algorithm_identifier, 
                    enable_tensorboard=config.tensorboard, 
                    evaluate=config.evaluate,
                    )
        

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
            message_dim=agent_config.message_dim,
            message_raw_dim=agent_config.feature_dim + agent_config.action_dim + agent_config.direction_dim,
        )
        agent_names = self.env_manager.get_agent_names()
        self.episode_manager = EpisodeManager(agent_names, parallel=config.parallel)

        agent_config.input_dim = (config.observation_shape[2] * config.stack_frames, config.observation_shape[0], config.observation_shape[1])
        agent_config.chkpt_dir = f"models/{config.algorithm_identifier}"
        self.agent_manager = AgentManager(
            agent_names=agent_names,
            agent_config=agent_config,
            evaluate=config.evaluate,
            parallel=config.parallel,
            logger=self.logger,
        )
        
        
        self.evaluator = Evaluator(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            evaluation_step=config.evaluation_step,
            max_evaluation_steps=config.max_evaluation_steps,
            eval_scenarios=config.eval_scenarios
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
        
