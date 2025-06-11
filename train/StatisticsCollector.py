from typing import List, Dict, Any, Union
import numpy as np
from statistics.experiment_data_collector import ExperimentDataCollector

class StatisticsCollector:
    def __init__(self, agent_names: List[str]):
        self.agent_names = agent_names

    def get_directional_acceleration(
        self, velocity: np.ndarray, acceleration: np.ndarray
    ) -> float:
        speed = np.linalg.norm(velocity)
        if speed > 0:
            direction = velocity / speed
        else:
            direction = np.zeros_like(velocity)
        return float(np.dot(acceleration, direction))

    def extract_scenario_data(
        self,
        scenario_id: int,
        observations: Dict[str, Any],
        infos: Dict[str, Any],
        collector: ExperimentDataCollector
    ) -> None:
        for agent_id in self.agent_names:
            if agent_id not in observations:
                continue

            ego_state = infos[agent_id]['env_obs'].ego_vehicle_state
            speed = np.linalg.norm(ego_state.linear_velocity)
            jerk = np.linalg.norm(ego_state.linear_jerk)
            acc = self.get_directional_acceleration(
                ego_state.linear_velocity,
                ego_state.linear_acceleration
            )

            collector.record_agent_data(
                agent_id,
                speed=speed,
                acceleration=acc,
                jerk=jerk,
                dt=infos[agent_id]['env_obs'].dt,
                travel_distance=infos[agent_id]['env_obs'].distance_travelled,
                time_separation=infos[agent_id]['time_separation'],
                is_waiting=(speed < 0.1),
                scenario_id=scenario_id,
            )

            if infos[agent_id]['env_obs'].events.collisions:
                collector.mark_agent_crashed(agent_id, scenario_id)

            if infos[agent_id]['env_obs'].events.reached_goal:
                collector.mark_agent_succeeded(agent_id, scenario_id)

        for social in infos.get("social_traffic", []):
            speed = np.linalg.norm(social["linear_velocity"])
            acc = self.get_directional_acceleration(
                np.array(social["linear_velocity"]),
                np.array(social["linear_acceleration"])
            )
            jerk = np.linalg.norm(social["linear_jerk"])

            collector.record_agent_data(
                agent_id=social["id"],
                speed=speed,
                acceleration=acc,
                jerk=jerk,
                dt=social["dt"],
                travel_distance=social["travel_distance"],
                time_separation=social["time_separation"],
                is_waiting=(speed < 0.1),
                scenario_id=scenario_id,
            )
            collector.add_social_vehicle(social["id"], scenario_id)

    def extract_scenario_data_batch(
        self,
        scenario_ids: List[int],
        observations_batch: List[Dict[str, Any]],
        infos_batch: List[Dict[str, Any]],
        collector: ExperimentDataCollector
    ) -> None:
        for sid, obs, info in zip(scenario_ids, observations_batch, infos_batch):
            self.extract_scenario_data(sid, obs, info, collector)
