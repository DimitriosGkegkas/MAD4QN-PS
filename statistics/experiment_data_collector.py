from energy.electric_vehicle_energy_model import ElectricVehicleEnergyModel
import os
import numpy as np
from datetime import datetime

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
                "agents_data": {},  # Dictionary to store data for each agent
                "averages": {},  # Store averages for speed, acceleration, energy, etc.
            }

        if agent_id not in self.current_scenarios[scenario_id]["agents_data"]:
            self.current_scenarios[scenario_id]["agents_data"][agent_id] = {
                "type": agent_type,
                "speeds": [],
                "accelerations": [],
                "energy_model": ElectricVehicleEnergyModel(),
                "travel_distance": 0,
                "travel_time": 0,
                "waiting_time": 0,
                "state": "traveling",  # Possible states: traveling, succeeded, crashed
            }

    def add_social_vehicle(self, agent_id, scenario_id):
        self.add_agent(agent_id, "social", scenario_id)
        self.mark_agent_succeeded(agent_id, scenario_id)

    def record_agent_data(self, agent_id, speed, acceleration, dt, travel_distance, is_waiting = False, scenario_id = None):
        """
        Record speed, acceleration, and energy consumption for a given agent in the current scenario.
        :param agent_id: Unique identifier for the agent
        :param speed: Current speed of the agent
        :param acceleration: Current acceleration of the agent
        :param dt: Time duration of this step
        :param travel_distance: Distance traveled during this step
        :param is_waiting: Boolean indicating whether the agent is waiting
        """
        if self.current_scenarios is None:
            raise ValueError("No active scenario. Start a new scenario first.")
        
        assert scenario_id is not None, "Scenario id must be provided"
        assert self.current_scenarios[scenario_id] is not None, "Scenario id must be valid"

        # Update agent's data
        agent_data = self.current_scenarios[scenario_id]["agents_data"][agent_id]

        # Append speed and acceleration
        agent_data["speeds"].append(speed)
        agent_data["accelerations"].append(acceleration)

        # Update travel and waiting times
        agent_data["travel_time"] += dt
        if is_waiting:
            agent_data["waiting_time"] += dt

        # Update travel distance
        agent_data["travel_distance"] += travel_distance

        # Update energy model
        agent_data["energy_model"].process_time_step(speed, acceleration, time_step=dt)

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

        self.current_scenarios[scenario_id]["agents_data"][agent_id]["state"] = state
    def close_scenario(self):
        """Close the current scenario and calculate averages."""
        if self.current_scenarios is None:
            raise ValueError("No active scenario to close.")
        for current_scenario in self.current_scenarios.values():
            self._close_scenario(current_scenario)
        self.current_scenarios = {}
    def _close_scenario(self, current_scenario):

        # Compute averages and finalize agent statistics
        for agent_id, data in current_scenario["agents_data"].items():
            avg_speed = sum(data["speeds"]) / len(data["speeds"]) if data["speeds"] else 0
            avg_acceleration = sum(data["accelerations"]) / len(data["accelerations"]) if data["accelerations"] else 0
            energy_consumption = data["energy_model"].get_energy_consumption_per_km()

            current_scenario["averages"][agent_id] = {
                "type": data["type"],
                "average_speed": avg_speed,
                "average_acceleration": avg_acceleration,
                "energy_consumption": energy_consumption,
                "travel_time": data["travel_time"],
                "waiting_time": data["waiting_time"],
                "state": data["state"],
            }
        
        # If at least one agent is still traveling, mark the scenario as incomplete
        # If at least one agent crashed, mark the scenario as crashed
        current_scenario["state"] = "crashed" if any(
            agent["state"] == "crashed" for agent in current_scenario["agents_data"].values()
        ) else "incomplete" if any(
            agent["state"] == "traveling" for agent in current_scenario["agents_data"].values()
        ) else "succeeded"

        self.scenarios.append(current_scenario)


    def get_statistics(self):
        """
        Calculate overall statistics across all scenarios and agents.
        Includes averages for speed, acceleration, energy consumption,
        travel time, waiting time, and success/crash/incomplete rates.
        :return: Dictionary with overall statistics.
        """
        stats_by_type = self.get_statistics_by_type()
        total = len(self.scenarios)
        overall_stats = {
            "average_travel_time": np.mean([
                data["travel_time"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if scenario["state"] == "succeeded"
            ]),
            "average_waiting_time": np.mean([
                data["waiting_time"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if scenario["state"] == "succeeded"
            ]),
            "average_speed": np.mean([
                data["average_speed"] 
                for scenario in self.scenarios 
                for agent_id, data in scenario["averages"].items() 
                if scenario["state"] == "succeeded"
            ]),
            "average_acceleration": np.mean([
                data["average_acceleration"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if scenario["state"] == "succeeded"
            ]),
            "average_energy_consumption": np.mean([
                data["energy_consumption"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if scenario["state"] == "succeeded"
            ]),
            "success_rate": sum(
                1 for scenario in self.scenarios if scenario["state"] == "succeeded"
            ) / total * 100 if total > 0 else 0,
            "crash_rate": sum(
                1 for scenario in self.scenarios if scenario["state"] == "crashed"
            ) / total * 100 if total > 0 else 0,
            "incomplete_rate": sum(
                1 for scenario in self.scenarios if scenario["state"] == "incomplete"
            ) / total * 100 if total > 0 else 0,
        }

        return {**overall_stats, "by_type": stats_by_type}

    def get_statistics_by_type(self):
        """
        Calculate statistics (average speed, acceleration, energy, etc.) grouped by agent type.
        :return: Dictionary with statistics grouped by type.
        """
        type_stats = {}

        # find all agent types unique from the scenarios
        agent_types = set(
            agent["type"]
            for scenario in self.scenarios
            for agent in scenario["agents_data"].values()
        )
        # Compute averages for each type
        for agent_type in agent_types:
            total = sum(
                1 for scenario in self.scenarios for agent in scenario["agents_data"].values()
                if agent["type"] == agent_type
            )

            type_stats[agent_type] = {}
            type_stats[agent_type]["average_travel_time"] = np.mean([
                data["travel_time"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if data["type"] == agent_type
                if data["state"] == "succeeded"

            ])
            type_stats[agent_type]["average_waiting_time"] = np.mean([
                data["waiting_time"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if data["type"] == agent_type
                if data["state"] == "succeeded"
            ])

            type_stats[agent_type]["average_speed"] = np.mean([
                data["average_speed"] 
                for scenario in self.scenarios 
                for agent_id, data in scenario["averages"].items() 
                if data["type"] == agent_type
                if scenario["state"] == "succeeded"
            ])
            type_stats[agent_type]["average_acceleration"] = np.mean([
                data["average_acceleration"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if data["type"] == agent_type
                if scenario["state"] == "succeeded"
            ])
            type_stats[agent_type]["average_energy_consumption"] = np.mean([
                data["energy_consumption"]
                for scenario in self.scenarios
                for agent_id, data in scenario["averages"].items()
                if data["type"] == agent_type
                if scenario["state"] == "succeeded"
            ])
            type_stats[agent_type]["success_rate"] = sum(
                1 for scenario in self.scenarios for agent in scenario["agents_data"].values() 
                if agent["state"] == "succeeded" and agent["type"] == agent_type
            ) / total * 100
            type_stats[agent_type]["crash_rate"] = sum(
                1 for scenario in self.scenarios for agent in scenario["agents_data"].values() 
                if agent["state"] == "crashed" and agent["type"] == agent_type
            ) /  total * 100
            type_stats[agent_type]["incomplete_rate"] = sum(
                1 for scenario in self.scenarios for agent in scenario["agents_data"].values() 
                if agent["state"] == "traveling" and agent["type"] == agent_type
            ) / total * 100


        return type_stats

    def _calculate_percentage(self, state):
        total = sum(
            1 for scenario in self.scenarios for agent in scenario["agents_data"].values() if agent["state"] == state
        )
        total_agents = sum(len(scenario["agents_data"]) for scenario in self.scenarios)
        return (total / total_agents) * 100 if total_agents > 0 else 0

    def _calculate_average(self, key):
        total = sum(
            sum(agent[key]) if key in agent else 0
            for scenario in self.scenarios
            for agent in scenario["agents_data"].values()
        )
        count = sum(
            len(agent[key]) if key in agent else 0
            for scenario in self.scenarios
            for agent in scenario["agents_data"].values()
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

    def save(self, base_dir = "data"):
        """
        Save statistics to files in a structured directory format.
        Uses the `self.get_statistics` method to retrieve statistics.
        """
        # Get statistics from the class
        statistics = self.get_statistics()


        # Subfolder for the experiment based on the algorithm identifier
        experiment_folder = os.path.join(base_dir, self.algorithm_identifier)

        # Subfolder for the current date (DDMMYYYY format)
        current_date = datetime.now().strftime("%d%m%Y")
        date_folder = os.path.join(experiment_folder, current_date)

        # Create directories if they don't exist
        os.makedirs(date_folder, exist_ok=True)

        # Separate general statistics from per-type statistics
        general_statistics = {k: v for k, v in statistics.items() if k != "by_type"}
        per_type_statistics = statistics.get("by_type", {})

        # Save general statistics to a single 'total.npy' file
        general_file_path = os.path.join(date_folder, "total.npy")
        np.save(general_file_path, general_statistics)
        print(f"Saved general statistics to {general_file_path}")

        # Save per-type statistics to individual files
        for agent_type, type_data in per_type_statistics.items():
            type_file_path = os.path.join(date_folder, f"{agent_type}.npy")
            np.save(type_file_path, type_data)
            print(f"Saved statistics for {agent_type} to {type_file_path}")
