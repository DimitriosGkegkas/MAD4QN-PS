from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
from statistics.experiment_data_collector import ExperimentDataCollector

class Evaluator:
    def __init__(
        self,
        trainer_logger,  # BaseTrainer instance
        agent_manager,   # AgentManager instance
        episode_manager,  # ScenarioManager instance
        env_manager,     # EnvironmentManager instance
        evaluation_step: int,
        total_scenarios: int,
        checkpoint_enabled: bool = True,
    ):
        self.logger = trainer_logger
        self.agents = agent_manager
        self.episode = episode_manager
        self.env_manager = env_manager
        self.evaluation_step = evaluation_step
        self.checkpoint_enabled = checkpoint_enabled
        self.best_score = -np.inf
        self.total_scenarios = total_scenarios

    def should_evaluate(self, n_episodes: int) -> bool:
        return n_episodes % self.evaluation_step == 0

    def evaluate(self, n_episodes: int, n_steps: int) -> Tuple[float, List[float]]:
        rewards_all: List[float] = []
        self.logger.log_percentage(0.0)

        for i, scenario_ids in enumerate(self.logger.slice_list(list(range(self.total_scenarios)), self.env.num_env)):
            scores = self._episode_eval(scenario_ids)
            rewards_all.extend(scores)
            self.logger.log_percentage(len(rewards_all) / self.total_scenarios)

        mean_score = float(np.mean(rewards_all))
        self.logger.log_scalar("reward/eval", mean_score, n_episodes)
        self.logger.log_histogram("reward/eval_distribution", rewards_all, n_episodes)

        if self.checkpoint_enabled and mean_score > self.best_score:
            self.agents.save_all_models()
            self.best_score = mean_score

        elapsed_time = datetime.now() - self.logger.start_time
        self.logger.scores_list.append((mean_score, str(elapsed_time), n_steps))
        self.logger.scores_per_scenario_list.append(rewards_all)
        self.logger.save_scores(self.logger.scores_list, self.logger.scores_per_scenario_list)

        return mean_score, rewards_all

    def _episode_eval(self, scenario_ids: List[int]) -> List[float]:
        direction, communication, current_state, current_messages, current_raw_messages, terminate, truncated, reward, infos = self.env_manager.reset()
        ep_steps = 0
        scores = [0.0 for _ in current_state]
        
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_batch_episode_done(current_state, reward, terminate, infos) and ep_steps < 1000:
            action, next_messages, _  = self.agent_manager.select_batch_actions(current_state, current_messages, terminate, truncated)
            
            next_state, terminate, truncated, reward, infos = self.env_manager.step(action)
            
            current_state = next_state
            current_messages = next_messages
            ep_steps += 1
            
            scores = [sum(r.values()) + s for r, s in zip(reward, scores)]
            self.logger.log_progress(np.mean(scores), self.n_episodes, ep_steps)
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
        direction, communication, current_state, current_messages, current_raw_messages, terminate, truncated, reward, infos = self.env_manager.reset(scenario_id)
        ep_steps = 0
        
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_batch_episode_done(current_state, reward, terminate, infos) and ep_steps < 1000:
            action, next_messages, _  = self.agent_manager.select_batch_actions(current_state, current_messages, terminate, truncated)
            
            next_state, terminate, truncated, reward, infos = self.env_manager.step(action)
            
            current_state = next_state
            current_messages = next_messages
            ep_steps += 1
