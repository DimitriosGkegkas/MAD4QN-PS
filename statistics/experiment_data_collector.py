from energy.electric_vehicle_energy_model import ElectricVehicleEnergyModel
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class ExperimentDataCollector:
    def __init__(self,  algorithm_identifier):
        """
        Initialize the collector with necessary data structures and vehicle model parameters.
        :algorithm_identifier: Identifier for the algorithm used in the experiment
        """
        self.scenarios = []  # List to store data for all scenarios
        self.current_scenarios = {}  # Store current scenario data
        self.algorithm_identifier = algorithm_identifier

    # def add_agent(self, agent_id, agent_type, scenario_id):

    def start_new_scenarios(self, scenario_ids, batch_agents_with_types):
        """
        Start a new scenario and initialize agents with their types.
        :param batch_agents_with_types: Dictionary mapping agent_id to agent_type.
        """

        # Initialize each agent with type and state
        for scenario_id, agents_with_types in zip(scenario_ids, batch_agents_with_types):
            for agent_id, agent_type in agents_with_types.items():
                self.add_agent(agent_id, agent_type, scenario_id)

    def add_agent(self, agent_id, agent_type, scenario_id):
        if scenario_id not in self.current_scenarios:
            self.current_scenarios[scenario_id] = {
            }

        if agent_id not in self.current_scenarios[scenario_id]:
            self.current_scenarios[scenario_id][agent_id] = {
                "type": agent_type,
                "distance": [],
                "speeds": [],
                "accelerations": [],
                "jerk": [],
                "dt": [],
                "waiting_time": 0,
                "state": "traveling",  # Possible states: traveling, succeeded, crashed
            }

    def add_social_vehicle(self, agent_id, scenario_id):
        self.add_agent(agent_id, "social", scenario_id)
        self.mark_agent_succeeded(agent_id, scenario_id)

    def record_agent_data(self, agent_id, speed, acceleration, jerk, dt, travel_distance, is_waiting = False, scenario_id = None):
        """
        Record speed, acceleration, and energy consumption for a given agent in the current scenario.
        :param agent_id: Unique identifier for the agent
        :param speed: Current speed of the agent
        :param acceleration: Current acceleration of the agent
        :param jerk: Current jerk of the agent
        :param dt: Time duration of this step
        :param travel_distance: Distance traveled during this step
        :param is_waiting: Boolean indicating whether the agent is waiting
        """
        if self.current_scenarios is None:
            raise ValueError("No active scenario. Start a new scenario first.")
        
        assert scenario_id is not None, "Scenario id must be provided"
        assert self.current_scenarios[scenario_id] is not None, "Scenario id must be valid"

        # Update agent's data
        agent_data = self.current_scenarios[scenario_id][agent_id]

        # Append speed and acceleration
        agent_data["distance"].append(travel_distance)
        agent_data["speeds"].append(speed)
        agent_data["accelerations"].append(acceleration)
        agent_data["jerk"].append(jerk)
        agent_data["dt"].append(dt)

        # Update travel and waiting times
        if is_waiting:
            agent_data["waiting_time"] += dt

    def mark_agent_succeeded(self, agent_id, scenario_id):
        """Mark the specified agent as succeeded."""
        self._change_agent_state(agent_id, "succeeded", scenario_id)

    def mark_agent_crashed(self, agent_id, scenario_id):
        """Mark the specified agent as crashed."""
        self._change_agent_state(agent_id, "crashed", scenario_id)

    def _change_agent_state(self, agent_id, state, scenario_id):
        """
        Internal method to change an agent's state.
        :param agent_id: Unique identifier for the agent
        :param state: New state for the agent ('succeeded' or 'crashed')
        """
        if self.current_scenarios is None:
            raise ValueError("No active scenario. Start a new scenario first.")
        
        assert scenario_id is not None, "Scenario id must be provided"
        assert self.current_scenarios[scenario_id] is not None, "Scenario id must be valid"

        self.current_scenarios[scenario_id][agent_id]["state"] = state
        
    def close_scenario(self):
        """Close the current scenario"""
        if self.current_scenarios is None:
            raise ValueError("No active scenario to close.")
        for current_scenario in self.current_scenarios.values():
            self.scenarios.append(current_scenario)
        self.current_scenarios = {}
        
    
    def get_scenarios_based_on_state(self, state):
        """Get all scenarios where all agents succeeded."""

        return [scenario for scenario in self.scenarios if all(agent["state"] == state for agent in scenario.values())]
    
    def get_success_scenarios(self):
        """Get all scenarios where all agents succeeded."""
        return self.get_scenarios_based_on_state("succeeded")
    
    def get_crashed_scenarios(self):
        """Get all scenarios where at least one agent crashed."""
        return [scenario for scenario in self.scenarios if any(agent["state"] == "crashed" for agent in scenario.values())]
    
    def get_energy_consumption(self):
        """Get energy consumption for all agents in the current scenario."""
        energy_consumption = []
        for scenario in self.scenarios:
            for agent in scenario.values():
                energy_model = ElectricVehicleEnergyModel()
                for v, a, dt in zip(agent["speeds"], agent["accelerations"], agent["dt"]):
                    energy_model.process_time_step(v, a, dt)
                energy_consumption.append(energy_model.get_energy_consumption_per_km())
        return energy_consumption
    
    def get_energy_consumption_time_series(self, agent_id, scenario_id):
        """Get energy consumption time series for a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        energy_model = ElectricVehicleEnergyModel()
        for v, a, dt in zip(agent["speeds"], agent["accelerations"], agent["dt"]):
            energy_model.process_time_step(v, a, dt)
        return energy_model.get_energy_consumption_history(), energy_model.get_energy_consumption_per_km()
    
    def get_acceleration_time_series(self, agent_id, scenario_id):
        """Get acceleration time series for a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        return agent["accelerations"]
    def get_speed_time_series(self, agent_id, scenario_id):
        """Get speed time series for a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        return agent["speeds"]
    def get_jerk_time_series(self, agent_id, scenario_id):
        """Get jerk time series for a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        return agent["jerk"]
    
    def get_total_distance(self, agent_id, scenario_id):
        """Get total distance traveled by a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        return sum(agent["distance"])
    
    def get_time_step(self, agent_id, scenario_id):
        """Get dt time series for a specific agent in the current scenario."""
        agent = self.scenarios[scenario_id][agent_id]
        return agent["dt"]
        

    def get_avg_statistics(self):
        """
        Calculate overall statistics across all scenarios and agents.
        Includes averages for speed, acceleration, energy consumption,
        travel time, waiting time, and success/crash/incomplete rates.
        :return: Dictionary with overall statistics.
        """
        total_scenarios = len(self.scenarios)
        succeeded_scenarios = self.get_success_scenarios()
        crashed_scenarios = self.get_crashed_scenarios()
        
        statistics = {
            "travel_time": np.mean([
                sum(agent["dt"])
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "waiting_time": np.mean([
                agent["waiting_time"]
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "speed": np.mean([
                np.mean(agent["speeds"])
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "acceleration": np.mean([
                np.mean(agent["accelerations"])
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "absolute_acceleration": np.mean([
                np.mean(np.abs(agent["accelerations"]))
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "absolute_jerk": np.mean([
                np.mean(agent["jerk"])
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]), 
            "distance": np.mean([
                sum(agent["distance"])
                for scenario in succeeded_scenarios
                for agent in scenario.values()
            ]),
            "energy_consumption": np.mean(self.get_energy_consumption()),
            "success_rate": len(succeeded_scenarios) / total_scenarios * 100 if total_scenarios > 0 else 0,
            "crash_rate": len(crashed_scenarios) / total_scenarios * 100 if total_scenarios > 0 else 0,
            "incomplete_rate": (total_scenarios - len(succeeded_scenarios) - len(crashed_scenarios)) / total_scenarios * 100 if total_scenarios > 0 else 0,
        }

        return statistics

    def _calculate_percentage(self, state):
        total = sum(
            1 for scenario in self.scenarios for agent in scenario.values() if agent["state"] == state
        )
        total_agents = sum(len(scenario) for scenario in self.scenarios)
        return (total / total_agents) * 100 if total_agents > 0 else 0

    def _calculate_average(self, key):
        total = sum(
            sum(agent[key]) if key in agent else 0
            for scenario in self.scenarios
            for agent in scenario.values()
        )
        count = sum(
            len(agent[key]) if key in agent else 0
            for scenario in self.scenarios
            for agent in scenario.values()
        )
        return total / count if count > 0 else 0
    

    def save_raw_data(self, base_dir = "data"):
        """
        Save raw data for all scenarios and agents to files in a structured directory format.
        """
        # Subfolder for the experiment based on the algorithm identifier
        experiment_folder = os.path.join(base_dir, self.algorithm_identifier)

        # Subfolder for the current date (DDMMYYYY format)
        current_date = datetime.now().strftime("%d%m%Y")
        date_folder = os.path.join(experiment_folder, current_date)

        # Create directories if they don't exist
        os.makedirs(date_folder, exist_ok=True)

        # Save raw data self.scenarios in one file called raw_data.npy
        scenario_file_path = os.path.join(date_folder, f"raw_data.npy")
        np.save(scenario_file_path, self.scenarios)
        print(f"Raw data saved to {scenario_file_path}")

    def load_raw_data(self, base_dir = "data", date = None):
        """
        Load raw data from files in a structured directory format.
        """
        # Subfolder for the experiment based on the algorithm identifier
        experiment_folder = os.path.join(base_dir, self.algorithm_identifier)

        # Find the most recent date folder if none is specified
        if date is None:
            dates = [
                d for d in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, d))
            ]
            if not dates:
                raise ValueError(f"No date folders found for algorithm '{self.algorithm_identifier}'.")
            date = max(dates, key=lambda d: datetime.strptime(d, "%d%m%Y"))  # Most recent date

        # Load the total.npy file
        data_dir = os.path.join(experiment_folder, date)

        # Load raw data from file
        scenario_file_path = os.path.join(data_dir, f"raw_data.npy")
        self.scenarios = np.load(scenario_file_path, allow_pickle=True)

    def save_statistics(self, base_dir = "data"):
        """
        Save statistics to files in a structured directory format.
        Uses the `self.get_statistics` method to retrieve statistics.
        """
        # Get statistics from the class
        statistics = self.get_avg_statistics()


        # Subfolder for the experiment based on the algorithm identifier
        experiment_folder = os.path.join(base_dir, self.algorithm_identifier)

        # Subfolder for the current date (DDMMYYYY format)
        current_date = datetime.now().strftime("%d%m%Y")
        date_folder = os.path.join(experiment_folder, current_date)

        # Create directories if they don't exist
        os.makedirs(date_folder, exist_ok=True)

        # Save general statistics to a single 'total.npy' file
        general_file_path = os.path.join(date_folder, "total.npy")
        np.save(general_file_path, statistics)
        print(f"Saved general statistics to {general_file_path}")

