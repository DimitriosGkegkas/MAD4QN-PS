from typing import List, Dict, Tuple
import numpy as np
from train.AgentManager import AgentManager
from train.BaseTrainer import BaseTrainer
from train.EnvironmentManager import EnvironmentManager
from train.EpisodeManager import EpisodeManager

class Evaluator:
    def __init__(
        self,
        trainer_logger: BaseTrainer,  # BaseTrainer instance
        agent_manager: AgentManager,   # AgentManager instance
        episode_manager: EpisodeManager,  # ScenarioManager instance
        env_manager: EnvironmentManager,     # EnvironmentManager instance
        evaluation_step: int,
        eval_scenarios: List[int],
        max_evaluation_steps: int = 1000,
        checkpoint_enabled: bool = True,
    ):
        self.logger = trainer_logger
        self.agent_manager = agent_manager
        self.episode_manager = episode_manager
        self.env_manager = env_manager
        self.evaluation_step = evaluation_step
        self.checkpoint_enabled = checkpoint_enabled
        self.best_score = -np.inf
        self.eval_scenarios = eval_scenarios
        self.max_evaluation_steps = max_evaluation_steps
        self.evaluate_step = 0

    def should_evaluate(self, n_episodes: int) -> bool:
        return n_episodes % self.evaluation_step == 0

    def evaluate(self, n_episodes: int, n_steps: int) -> Tuple[float, List[float]]:
        self.agent_manager.eval()
        rewards_all: List[float] = []
        self.logger.log_percentage(0.0)
        self.env_manager.auto_reset(False)
        
        for i, scenario_ids in enumerate(self.logger.slice_list(self.eval_scenarios, self.env_manager.num_env)):
            
            if len(scenario_ids) < self.env_manager.num_env:
                continue  # Skip if not enough scenarios for the number of environments
            scores = self._episode_eval(scenario_ids)
            rewards_all.extend(scores)
            self.logger.log_percentage(len(rewards_all) / len(self.eval_scenarios))
            
        self.logger.after_evaluation(rewards_all, self.eval_scenarios[:len(rewards_all)], n_episodes, n_steps)
        self.evaluate_step += 1

        mean_score = float(np.mean(rewards_all))

        if self.checkpoint_enabled and mean_score > self.best_score:
            self.agent_manager.save()
            self.best_score = mean_score

        return

    def _average_rewards(self, reward_batch: List[Dict]) -> List[float]:
        return [np.mean(list(r.values())) if r else 0.0 for r in reward_batch]

    def _episode_eval(self, scenario_ids: List[int]) -> List[float]:
        current_state, terminate, truncated, reward, infos = self.env_manager.reset(scenario_ids)
        ep_steps = 0
        scores = [0.0 for _ in current_state]

        while True:
            action, next_messages, _ = self.agent_manager.action(current_state, terminate, truncated)
            current_state, reward, terminate, truncated, infos, _ = self.env_manager.step(action, next_messages)
            ep_steps += 1

            avg_rewards = self._average_rewards(reward)
            scores = [s + r for s, r in zip(scores, avg_rewards)]
            if self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) or ep_steps > self.max_evaluation_steps:
                break
        return scores
        
        
    def envision(self, scenario_id: int) -> None:
        self.agent_manager.eval()
        current_state, terminate, truncated, reward, infos = self.env_manager.reset([scenario_id])
        ep_steps = 0
        while True:
            action, next_messages, _  = self.agent_manager.action(current_state, terminate, truncated)
            
            current_state, reward, terminate, truncated, infos, _ = self.env_manager.step(action, next_messages)

            ep_steps += 1
            if self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) or ep_steps > self.max_evaluation_steps:
                break
