from typing import List, Dict, Any, Union
import numpy as np
from evaluation.experiment_data_collector import ExperimentDataCollector
from train import AgentManager, EnvironmentManager, EpisodeManager, BaseTrainer

class StatisticsCollector:
    def __init__(self,
        trainer_logger: BaseTrainer,  # BaseTrainer instance
        agent_manager: AgentManager,   # AgentManager instance
        episode_manager: EpisodeManager,  # ScenarioManager instance
        env_manager: EnvironmentManager,     # EnvironmentManager instance
        max_evaluation_steps: int,
        eval_scenarios: List[int],
        algorithm_identifier: str,
    ):
        self.logger = trainer_logger
        self.agent_manager = agent_manager
        self.episode_manager = episode_manager
        self.env_manager = env_manager
        self.max_evaluation_steps = max_evaluation_steps
        self.best_score = -np.inf
        self.eval_scenarios = eval_scenarios
        self.collector = ExperimentDataCollector(algorithm_identifier)

    def evaluate(self) -> None:
        self.agent_manager.eval()
        rewards_all: List[float] = []
        self.logger.log_percentage(0.0)
        
        for i, scenario_ids in enumerate(self.logger.slice_list(self.eval_scenarios, self.env_manager.num_env)):
            scores = self._episode_eval(scenario_ids)
            rewards_all.extend(scores)
            self.logger.log_percentage(len(rewards_all) / len(self.eval_scenarios))
        print()
        self.collector.save_data()
        return

    def _average_rewards(self, reward_batch: List[Dict]) -> List[float]:
        return [np.mean(list(r.values())) if r else 0.0 for r in reward_batch]

    def _episode_eval(self, scenario_ids: List[int]) -> List[float]:
        current_state, terminate, truncated, reward, infos = self.env_manager.reset(scenario_ids)
        self.extract_scenario_data_batch(scenario_ids, current_state, infos, reward)
        ep_steps = 0
        scores = [0.0 for _ in current_state]

        while True:
            action, next_messages, _ = self.agent_manager.action(current_state, terminate, truncated)
            current_state, reward, terminate, truncated, infos, _ = self.env_manager.step(action, next_messages)
            self.extract_scenario_data_batch(scenario_ids, current_state, infos, reward)
            ep_steps += 1

            avg_rewards = self._average_rewards(reward)
            scores = [s + r for s, r in zip(scores, avg_rewards)]
            if self.episode_manager.is_done(current_state, reward, terminate, truncated, infos) or ep_steps > self.max_evaluation_steps:
                break
        self.collector.reset()
        return scores
    

    def extract_scenario_data_batch(
        self,
        scenario_ids: List[int],
        observations_batch: List[Dict[str, Any]],
        infos_batch: List[Dict[str, Any]],
        reward_batch: List[Dict[str, Any]]
    ) -> None:
        for sid, obs, info, reward in zip(scenario_ids, observations_batch, infos_batch, reward_batch):
            self.extract_scenario_data(sid, obs, info, reward)
            

    def extract_scenario_data(
        self,
        scenario_id: int,
        observations: Dict[str, Any],
        infos: Dict[str, Any],
        reward: Dict[str, Any] = None
    ) -> None:
        for agent_id in observations:
            ego_state = infos[agent_id]['env_obs'].ego_vehicle_state
            speed = np.linalg.norm(ego_state.linear_velocity)
            jerk = np.linalg.norm(ego_state.linear_jerk)
            acc = self.get_directional_acceleration(
                ego_state.linear_velocity,
                ego_state.linear_acceleration
            )

            self.collector.record_agent_data(
                agent_id,
                speed=speed,
                acceleration=acc,
                jerk=jerk,
                dt=infos[agent_id]['env_obs'].dt,
                travel_distance=infos[agent_id]['env_obs'].distance_travelled,
                time_separation=infos[agent_id]['time_separation'] if 'time_separation' in infos[agent_id] else 0.0,
                is_waiting=(speed < 0.1),
                scenario_id=scenario_id,
                reward=reward[agent_id] if reward is not None else 0.0,
            )

            if infos[agent_id]['env_obs'].events.collisions or \
                infos[agent_id]['env_obs'].events.off_road or \
                infos[agent_id]['env_obs'].events.off_route or \
                infos[agent_id]['env_obs'].events.on_shoulder or \
                infos[agent_id]['env_obs'].events.wrong_way:
                self.collector.mark_agent_crashed(agent_id, scenario_id)

            if infos[agent_id]['env_obs'].events.reached_goal:
                self.collector.mark_agent_succeeded(agent_id, scenario_id)

        for social in infos.get("social_traffic", []):
            speed = np.linalg.norm(social["linear_velocity"])
            acc = self.get_directional_acceleration(
                np.array(social["linear_velocity"]),
                np.array(social["linear_acceleration"])
            )
            jerk = np.linalg.norm(social["linear_jerk"])

            self.collector.record_agent_data(
                agent_id=social["id"],
                speed=speed,
                acceleration=acc,
                jerk=jerk,
                dt=social["dt"],
                travel_distance=social["travel_distance"],
                time_separation=social["time_separation"] if "time_separation" in social else 0.0,
                is_waiting=(speed < 0.1),
                scenario_id=scenario_id,
            )
            self.collector.add_social_vehicle(social["id"], scenario_id)



    def get_directional_acceleration(
        self, velocity: np.ndarray, acceleration: np.ndarray
    ) -> float:
        speed = np.linalg.norm(velocity)
        if speed > 0:
            direction = velocity / speed
        else:
            direction = np.zeros_like(velocity)
        return float(np.dot(acceleration, direction))