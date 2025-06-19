from typing import List, Dict, Tuple, Optional
import numpy as np

from train.AgentManager import AgentManager
from train.BaseTrainer import BaseTrainer
from train.EnvironmentManager import EnvironmentManager
from train.EpisodeManager import EpisodeManager
from train.Evaluator import Evaluator

class Trainer:
    def __init__(
        self,
        trainer_logger: BaseTrainer,  # BaseTrainer instance
        agent_manager: AgentManager,   # AgentManager instance
        episode_manager: EpisodeManager,  # ScenarioManager instance
        evaluator: Evaluator,       # Evaluator instance
        env_manager: EnvironmentManager,     # EnvironmentManager instance
        total_steps: int,
        agent_count: int,
        algorithm_identifier: str,
        evaluation_step: int,
        max_training_steps: int = 1000
    ):
        self.logger = trainer_logger
        self.agent_manager = agent_manager
        self.episode_manager = episode_manager
        self.env_manager = env_manager
        self.evaluator = evaluator
        
        self.total_steps = total_steps
        self.agent_count = agent_count
        self.algorithm_identifier = algorithm_identifier
        self.evaluation_step = evaluation_step
        self.max_training_steps = max_training_steps
        
        self.n_steps = 0
        self.n_episodes = 0
        
    def train(self) -> None:
        while self.n_steps < self.total_steps:
            self._episode_train()

    def _episode_train(self) -> None:
        self.agent_manager.train()
        current_state, terminate, truncated, reward, infos = self.env_manager.reset()
        ep_steps = 0
        scores = [0.0 for _ in current_state]

        while not self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) and ep_steps < self.max_training_steps:
            action, next_messages, next_raw_messages  = self.agent_manager.action(current_state, terminate, truncated)
            
            next_state, reward, terminate, truncated, infos = self.env_manager.step(action, next_messages, next_raw_messages)

            self.agent_manager.store_transitions(current_state, action, reward, next_state, terminate, truncated)
            self.agent_manager.update_agent(step=self.n_steps)
            
            current_state = next_state
            self.n_steps += 1
            ep_steps += 1
            
            scores = [sum(r.values()) + s for r, s in zip(reward, scores)]
            self.logger.after_train_step(np.mean(scores), self.n_episodes, ep_steps)
        


        # Log and evaluate
        self.logger.after_episode_batch(
            episode=self.n_episodes,
            stats={
                "reward/mean": np.mean(scores),
                "reward/max": np.max(scores),
                "reward/min": np.min(scores),
                "steps": ep_steps,
            }
        )

        self.n_episodes += 1
        self.env_manager.env.modify_probs(self.n_episodes)
        
        if self.evaluator.should_evaluate(self.n_episodes):
            self.evaluator.evaluate(self.n_episodes, self.n_steps)
        self.agent_manager.save(best= False)
