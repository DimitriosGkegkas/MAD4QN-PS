from typing import List, Dict, Tuple, Optional
import numpy as np

from train.AgentManager import AgentManager
from train.BaseTrainer import BaseTrainer
from train.EpisodeManager import EpisodeManager
from train.Evaluator import Evaluator

class Trainer:
    def __init__(
        self,
        trainer_logger: BaseTrainer,  # BaseTrainer instance
        agent_manager: AgentManager,   # AgentManager instance
        episode_manager: EpisodeManager,  # ScenarioManager instance
        evaluator: Evaluator,       # Evaluator instance
        env_manager,     # EnvironmentManager instance
        config
    ):
        self.logger = trainer_logger
        self.agent_manager = agent_manager
        self.episode_manager = episode_manager
        self.env_manager = env_manager
        self.evaluator = evaluator
        
        self.total_steps = config.total_steps
        self.agent_count = config.agent_count
        self.algorithm_identifier = config.algorithm_identifier
        self.evaluation_step = config.evaluation_step
        self.max_training_steps = config.max_train_steps
        
        self.n_steps = 0
        self.n_episodes = 0
        
    def train(self) -> None:
        while self.n_steps < self.total_steps:
            self._episode_train()

    def _episode_train(self) -> None:
        self.agent_manager.train()
        direction, communication, current_messages, current_raw_messages, current_state, terminate, truncated, reward, infos = self.env_manager.reset()
        ep_steps = 0
        scores = [0.0 for _ in current_state]
        
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_done(current_state, reward, terminate, infos) and ep_steps < self.max_training_steps:
            action, next_messages, next_raw_messages  = self.agent_manager.action(current_state, current_messages, terminate, truncated)
            
            next_state, reward, terminate, truncated, infos = self.env_manager.step(action)

            self.agent_manager.store_transitions(current_state, current_raw_messages, action, reward, next_state, next_raw_messages, terminate, truncated)
            self.agent_manager.update_agent(step=ep_steps, logger=self.logger)
            
            current_state = next_state
            current_messages = next_messages
            current_raw_messages = next_raw_messages
            self.n_steps += 1
            ep_steps += 1
            
            scores = [sum(r.values()) + s for r, s in zip(reward, scores)]
            self.logger.after_train_step(np.mean(scores), self.n_episodes, ep_steps)

        # Log and evaluate
        self.logger.after_episode_batch(
            episode=self.n_episodes,
            stats={
                "Avg.Reward": np.mean(scores),
                "Max.Reward": np.max(scores),
                "Min.Reward": np.min(scores),
            }
        )

        self.n_episodes += 1

        if self.evaluator.should_evaluate(self.n_episodes):
            mean_score, rewards_all = self.evaluator.evaluate(self.n_episodes, self.n_steps)
            self.env_manager.env.modify_probs(rewards_all)
