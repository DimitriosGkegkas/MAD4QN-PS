from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
from statistics.experiment_data_collector import ExperimentDataCollector
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
        total_scenarios: int,
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
        self.total_scenarios = total_scenarios
        self.max_evaluation_steps = max_evaluation_steps
        self.evaluate_step = 0

    def should_evaluate(self, n_episodes: int) -> bool:
        return n_episodes % self.evaluation_step == 0

    def evaluate(self, n_episodes: int, n_steps: int) -> Tuple[float, List[float]]:
        self.agent_manager.eval()
        rewards_all: List[float] = []
        self.logger.log_percentage(0.0)

        for i, scenario_ids in enumerate(self.logger.slice_list(list(range(self.total_scenarios)), self.env_manager.num_env)):
            scores = self._episode_eval(scenario_ids)
            rewards_all.extend(scores)
            self.logger.log_percentage(len(rewards_all) / self.total_scenarios)
            
        self.logger.after_evaluation(episode = self.evaluate_step, scenario_rewards=rewards_all)
        self.evaluate_step += 1

        mean_score = float(np.mean(rewards_all))
        self.logger.log_scalar("reward/eval", mean_score, n_episodes)
        self.logger.log_histogram("reward/eval_distribution", rewards_all, n_episodes)

        if self.checkpoint_enabled and mean_score > self.best_score:
            self.agent_manager.save()
            self.best_score = mean_score

        elapsed_time = datetime.now() - self.logger.start_time

        return mean_score, rewards_all

    def _episode_eval(self, scenario_ids: List[int]) -> List[float]:
        direction, communication, current_messages, current_raw_messages, current_state, terminate, truncated, reward, infos = self.env_manager.reset(scenario_ids)
        ep_steps = 0
        scores = [0.0 for _ in current_state]
        
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) and ep_steps < self.max_evaluation_steps:
            action, next_messages, _  = self.agent_manager.action(current_state, current_messages, terminate, truncated)
            
            next_state, reward, terminate, truncated, infos = self.env_manager.step(action)

            
            current_state = next_state
            current_messages = next_messages
            ep_steps += 1
            
            scores = [sum(r.values()) + s for r, s in zip(reward, scores)]

        return scores


    # def collect_statistics(self, collector: ExperimentDataCollector, parallel: bool = True) -> List[float]:
    #     all_scores = []
    #     if parallel:
    #         for ids in self.logger.slice_list(list(range(self.total_scenarios)), self.env.num_env):
    #             scores = self._full_eval_batch(ids, collector)
    #             all_scores.extend(scores)
    #     else:
    #         for i in range(self.total_scenarios):
    #             self._full_eval_single(i, collector)
    #     collector.save_raw_data()
    #     return all_scores

    # def _full_eval_batch(self, ids: List[int], collector: ExperimentDataCollector) -> List[float]:
    #     obs_batch, infos = self.env.reset(ids)
    #     turning_ints = self.episode.get_batch_turning_intentions(infos)
    #     collector.start_new_scenarios(ids, turning_ints)
    #     ...
    #     # Same loop as in _eval_batch, but calls collector.record_agent_data()
    #     ...

    # def _full_eval_single(self, id: int, collector: ExperimentDataCollector) -> None:
    #     obs, infos = self.env.reset_single(id)
    #     turning_int = self.episode.get_turning_intention(infos)
    #     collector.start_new_scenarios([id], [turning_int])
    #     ...
    #     # Single-agent eval loop with collector logic
    #     ...
        
        
    def envision(self, scenario_id: int) -> None:
        direction, communication, current_messages, current_raw_messages, current_state, terminate, truncated, reward, infos = self.env_manager.reset(scenario_id)
        ep_steps = 0
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) and ep_steps < self.max_evaluation_steps:
            action, next_messages, _  = self.agent_manager.action(current_state, current_messages, terminate, truncated)
            
            next_state, reward, terminate, truncated, infos = self.env_manager.step(action)

            
            current_state = next_state
            current_messages = next_messages
            ep_steps += 1

