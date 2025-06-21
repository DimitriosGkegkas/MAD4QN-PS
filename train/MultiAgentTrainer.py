from omegaconf import DictConfig
from train.BaseTrainer import BaseTrainer
from train.AgentManager import AgentManager
from train.EnvironmentManager import EnvironmentManager
from train.Evaluator import Evaluator
from train.EpisodeManager import EpisodeManager
from train.StatisticsCollector import StatisticsCollector
from train.Trainer import Trainer

class MultiAgentTrainer:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.eval = cfg.shared.evaluate
        
        self.logger = BaseTrainer(cfg.trainer)
        self.env_manager = EnvironmentManager(cfg.environment)
        
        agent_names = self.env_manager.get_agent_names()
        self.episode_manager = EpisodeManager(agent_names, parallel=cfg.shared.parallel)

        cfg.agent.input_dim = (cfg.environment.observation_shape[2] * cfg.environment.stack_frames, cfg.environment.observation_shape[0], cfg.environment.observation_shape[1])
        self.agent_manager = AgentManager(
            agent_names=agent_names,
            cfg = cfg.agent,
            logger=self.logger,
        )
        
        
        self.evaluator = Evaluator(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            evaluation_step=cfg.evaluator.evaluation_step,
            max_evaluation_steps=cfg.evaluator.max_evaluation_steps,
            eval_scenarios=cfg.evaluator.eval_scenarios
        )
        
        self.statistics = StatisticsCollector(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            env_manager=self.env_manager,
            max_evaluation_steps=cfg.evaluator.max_evaluation_steps,
            eval_scenarios=cfg.evaluator.eval_scenarios,
            algorithm_identifier = cfg.algorithm_identifier,
        )
        
        self.trainer = Trainer(
            trainer_logger=self.logger,
            agent_manager=self.agent_manager,
            episode_manager=self.episode_manager,
            evaluator=self.evaluator,
            env_manager=self.env_manager,
            cfg = cfg.trainer
        )      

    def preload(self, path: str) -> None:
        self.agent_manager.load(path, evaluate=self.eval)
        
    def envision(self, scenario_id: int ) -> None:
        self.evaluator.envision(scenario_id)
        
    def train(self) -> None:
        self.trainer.train()
        
    def evaluate(self) -> None:
        self.statistics.evaluate()
        