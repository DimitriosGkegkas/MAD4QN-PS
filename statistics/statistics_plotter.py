from itertools import accumulate
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from statistics.experiment_data_collector import ExperimentDataCollector


class StatisticsPlotter:
    def __init__(self, algorithm_identifier=None):
        """
        Initialize the plotter with storage for algorithm results.
        """
        self.results = {}  # Store statistics by algorithm identifier
        self.dataCollector = None
        
        if(algorithm_identifier is not None):
            self.algorithm_identifier = algorithm_identifier
            self.dataCollector = ExperimentDataCollector(algorithm_identifier)
            self.dataCollector.load_raw_data()

    def add_algorithm(self, algorithm_identifier, date=None):
        """
        Add results for a specific algorithm. If no date is specified, the most recent date is used.
        :param algorithm_identifier: Name of the algorithm.
        :param date: Optional date in DDMMYYYY format. Defaults to the most recent.
        """
        dataCollector = ExperimentDataCollector(algorithm_identifier)
        dataCollector.load_raw_data(date=date)
        # Load statistics and store
        self.results[algorithm_identifier] = dataCollector.get_avg_statistics()

    def _save_figure(self, figure, name, algorithms):
        """
        Save a figure to the appropriate directory.
        :param figure: The matplotlib figure to save.
        :param name: Name of the figure.
        :param algorithms: List of algorithm identifiers used in the figure.
        """
        base_dir = "figures"
        sorted_algorithms = "_".join(sorted(algorithms))

        # Subfolder for the current date (DDMMYYYY format)
        current_date = datetime.now().strftime("%d%m%Y")
        date_folder = os.path.join(base_dir, current_date)
        os.makedirs(date_folder, exist_ok=True)

        # Save the figure
        file_path = os.path.join(date_folder, f"{name}_{sorted_algorithms}.png")
        figure.savefig(file_path)
        print(f"Saved figure to {file_path}")

    def plot_travel_and_waiting_time(self):
        """
        Plot travel time and waiting time as grouped bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        travel_times = [self.results[algo].get("travel_time", 0) for algo in algorithms]
        waiting_times = [self.results[algo].get("waiting_time", 0) for algo in algorithms]

        # Create a bar plot
        x = np.arange(len(algorithms))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, travel_times, width, label='Travel Time', color='skyblue')
        ax.bar(x + width/2, waiting_times, width, label='Waiting Time', color='lightgreen')
        ax.set_title('Travel and Waiting Time Comparison', fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=0, fontsize=18)
        ax.set_ylabel('Time (seconds)', fontsize=18)
        ax.legend(fontsize=18)
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "travel_and_waiting_time", algorithms)
        plt.show()

    def plot_success_crash_incomplete_rates(self):
        """
        Plot success, crash, and incomplete rates as stacked bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        success_rates = [self.results[algo].get("success_rate", 0) for algo in algorithms]
        crash_rates = [self.results[algo].get("crash_rate", 0) for algo in algorithms]
        incomplete_rates = [self.results[algo].get("incomplete_rate", 0) for algo in algorithms]

        # Create a stacked bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algorithms, success_rates, label='Success Rate', color='lightgreen')
        ax.bar(algorithms, crash_rates, bottom=success_rates, label='Crash Rate', color='salmon')
        ax.bar(algorithms, incomplete_rates,
               bottom=[success_rates[i] + crash_rates[i] for i in range(len(success_rates))],
               label='Incomplete Rate', color='lightgray')
        ax.set_title('Success, Crash, and Incomplete Rates', fontsize=22)
        ax.set_ylabel('Percentage (%)', fontsize=18)
        ax.set_xticks(range(len(algorithms)))  # Ensure tick positions match the bars
        ax.set_xticklabels(algorithms, rotation=0, ha='right', fontsize=18) # Rotate and align labels
    
    
        ax.legend(fontsize=18)
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "success_crash_incomplete_rates", algorithms)
        plt.show()

    def plot_fuel_consumption(self):
        """
        Plot fuel consumption (energy) as bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        fuel_consumptions = [self.results[algo].get("energy_consumption", 0) for algo in algorithms]

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algorithms, fuel_consumptions, color='gold')
        ax.set_title('Fuel Consumption Comparison', fontsize=22)
        ax.set_ylabel('Energy Consumption (Wh)', fontsize=18)
        ax.set_xlabel('Algorithm')
        ax.set_ylim(min(fuel_consumptions) - 50, max(fuel_consumptions) + 50)
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "fuel_consumption", algorithms)
        plt.show()

    def plot_speed_and_acceleration(self):
        """
        Plot average speed and acceleration as grouped bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        avg_speeds = [self.results[algo].get("speed", 0) for algo in algorithms]
        avg_accelerations = [self.results[algo].get("acceleration", 0) for algo in algorithms]

        # Create a bar plot
        x = np.arange(len(algorithms))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, avg_speeds, width, label='Average Speed', color='cornflowerblue')
        ax.bar(x + width/2, avg_accelerations, width, label='Average Acceleration', color='coral')
        ax.set_title('Speed and Acceleration Comparison', fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=0, fontsize=18)
        ax.set_ylabel('Values', fontsize=18)
        ax.legend(fontsize=18)
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "speed_and_acceleration", algorithms)
        plt.show()

    def plot_combined_statistics(self):
        """
        Create a combined plot with subplots comparing algorithms across multiple metrics.
        """
        algorithms = list(self.results.keys())
        plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

        travel_times = [self.results[algo].get("travel_time", 0) for algo in algorithms]
        waiting_times = [self.results[algo].get("waiting_time", 0) for algo in algorithms]
        success_rates = [self.results[algo].get("success_rate", 0) for algo in algorithms]
        crash_rates = [self.results[algo].get("crash_rate", 0) for algo in algorithms]
        incomplete_rates = [self.results[algo].get("incomplete_rate", 0) for algo in algorithms]
        fuel_consumptions = [self.results[algo].get("energy_consumption", 0) for algo in algorithms]
        avg_speeds = [self.results[algo].get("speed", 0) for algo in algorithms]
        avg_accelerations = [self.results[algo].get("acceleration", 0) for algo in algorithms]
        avg_absolute_jerk = [self.results[algo].get("absolute_jerk", 0) for algo in algorithms]
        avg_absolute_acceleration = [self.results[algo].get("absolute_acceleration", 0) for algo in algorithms]

        fig, axs = plt.subplots(3, 2, figsize=(30, 10))
        
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Top Left: Travel Time and Waiting Time
        x = np.arange(len(algorithms))
        width = 0.35
        axs[0, 0].bar(x - width/2, travel_times, width, label='Average Travel Time', color='skyblue')
        axs[0, 0].bar(x + width/2, waiting_times, width, label='Average Waiting Time', color='lightgreen')
        axs[0, 0].set_title('Travel and Waiting Time', fontsize=22)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(algorithms, rotation=0, fontsize=18)
        axs[0, 0].set_ylabel('Time (seconds)', fontsize=18)
        axs[0, 0].legend(fontsize=18)

        # Top Right: Success, Crash, and Incomplete Rates
        axs[0, 1].bar(algorithms, success_rates, label='Average Success Rate', color='lightgreen')
        axs[0, 1].bar(algorithms, crash_rates, bottom=success_rates, label='Average Crash Rate', color='salmon')
        axs[0, 1].bar(algorithms, incomplete_rates,
                       bottom=[success_rates[i] + crash_rates[i] for i in range(len(success_rates))],
                       label='Average Incomplete Rate', color='lightgray')
        axs[0, 1].set_title('Success, Crash, and Incomplete Rates', fontsize=22)
        axs[0, 1].set_ylabel('Average Percentage (%)', fontsize=18)
        axs[0, 1].legend(fontsize=18)

        # Bottom Left: Fuel Consumption
        axs[1, 0].bar(algorithms, fuel_consumptions, color='gold')
        axs[1, 0].set_title('Energy Consumption', fontsize=22)
        axs[1, 0].set_ylabel('Average Energy Consumption (Wh/Km)', fontsize=18)
        axs[1, 0].set_xlabel('Algorithm')
        axs[1, 0].set_ylim(0.9*min(fuel_consumptions), 1.1*max(fuel_consumptions) )

        # Bottom Right: Speed and Acceleration
        # Bar for Average Speed on the left y-axis
        ax1 = axs[1, 1]
        ax1.bar(x - width/2, avg_speeds, width, label='Average Speed (m/s)', color='cornflowerblue')
        ax1.set_ylabel('Average Speed (m/s)', color='cornflowerblue', fontsize=18)
        ax1.tick_params(axis='y', labelcolor='cornflowerblue', labelsize=18)
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=0, fontsize=18)
        ax1.set_title('Speed and Acceleration', fontsize=22)

        # Create a twin y-axis for Average Acceleration
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, avg_accelerations, width, label='Average Acceleration (m/s²)', color='coral')
        ax2.set_ylabel('Average Acceleration (m/s²)', color='coral', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='coral',labelsize=18)

        # Bottom Center: Absolute Jerk
        axs[2, 1].bar(algorithms, avg_absolute_jerk, color='purple', label='Average Absolute Jerk')
        axs[2, 1].bar(algorithms, avg_absolute_acceleration, color='orange', bottom=avg_absolute_jerk, label='Average Absolute Acceleration')

        axs[2, 1].set_xlabel('Algorithm', fontsize=14)
        axs[2, 1].set_ylabel('Passenger Comfort Metric (|a| + |Jerk|)', fontsize=14)
        axs[2, 1].set_title('Passenger Comfort Analysis: Absolute Acceleration & Jerk', fontsize=18)

        
        axs[2, 1].legend(fontsize=18)
        
        for ax in axs.flat:
            ax.tick_params(axis='x', labelsize=12, rotation=0) 

        # plt.tight_layout()

        # Save and show
        fig.set_size_inches(16, 20)  # Manually adjust figure size
        self._save_figure(fig, "combined_statistics", algorithms)
        
        plt.show()
        
    def plot_travel_speed_success(self):
        """
        Plot Travel & Waiting Time, Speed & Acceleration, and Success, Crash, and Incomplete Rates
        """
        algorithms = list(self.results.keys())
        plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

        travel_times = [self.results[algo].get("travel_time", 0) for algo in algorithms]
        waiting_times = [self.results[algo].get("waiting_time", 0) for algo in algorithms]
        success_rates = [self.results[algo].get("success_rate", 0) for algo in algorithms]
        crash_rates = [self.results[algo].get("crash_rate", 0) for algo in algorithms]
        incomplete_rates = [self.results[algo].get("incomplete_rate", 0) for algo in algorithms]
        avg_speeds = [self.results[algo].get("speed", 0) for algo in algorithms]
        avg_accelerations = [self.results[algo].get("acceleration", 0) for algo in algorithms]
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 13))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Travel and Waiting Time
        axs[0].bar(x - width/2, travel_times, width, label='Average Travel Time', color='skyblue')
        axs[0].bar(x + width/2, waiting_times, width, label='Average Waiting Time', color='lightgreen')
        axs[0].set_title('Travel and Waiting Time', fontsize=22)
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(algorithms, rotation=0, fontsize=18)
        axs[0].set_ylabel('Time (seconds)', fontsize=18)
        axs[0].legend(fontsize=18)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Speed and Acceleration
        ax1 = axs[1]
        ax1.bar(x - width/2, avg_speeds, width, label='Average Speed (m/s)', color='cornflowerblue')
        ax1.set_ylabel('Average Speed (m/s)', color='cornflowerblue', fontsize=18)
        ax1.tick_params(axis='y', labelcolor='cornflowerblue', labelsize=18)
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=0, fontsize=18)
        ax1.set_title('Speed and Acceleration', fontsize=22)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, avg_accelerations, width, label='Average Acceleration (m/s²)', color='coral')
        ax2.set_ylabel('Average Acceleration (m/s²)', color='coral', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='coral', labelsize=18)
        
        # Success, Crash, and Incomplete Rates
        axs[2].bar(algorithms, success_rates, label='Average Success Rate', color='lightgreen')
        axs[2].bar(algorithms, crash_rates, bottom=success_rates, label='Average Crash Rate', color='salmon')
        axs[2].bar(algorithms, incomplete_rates,
                   bottom=[success_rates[i] + crash_rates[i] for i in range(len(success_rates))],
                   label='Average Incomplete Rate', color='lightgray')
        axs[2].set_title('Success, Crash, and Incomplete Rates', fontsize=22)
        axs[2].set_ylabel('Average Percentage (%)', fontsize=18)
        axs[2].set_xlabel('Algorithm', fontsize=18)
        axs[2].legend(fontsize=18)
        axs[2].grid(True, linestyle='--', alpha=0.7)

        fig.set_size_inches(15, 15)
        self._save_figure(fig, "travel_speed_success", algorithms)
        plt.show()
    
    def plot_comfort_energy(self):
        """
        Plot Passenger Comfort Metrics and Energy Consumption
        """
        algorithms = list(self.results.keys())
        plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

        fuel_consumptions = [self.results[algo].get("energy_consumption", 0) for algo in algorithms]
        avg_absolute_jerk = [self.results[algo].get("absolute_jerk", 0) for algo in algorithms]
        avg_absolute_acceleration = [self.results[algo].get("absolute_acceleration", 0) for algo in algorithms]
        
        fig, axs = plt.subplots(2, 1, figsize=(8, 15))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        
        # Energy Consumption
        axs[0].bar(algorithms, fuel_consumptions, color='gold')
        axs[0].set_title('Average Energy Consumption', fontsize=22)
        axs[0].set_ylabel('kWh/Km', fontsize=18)
        # axs[0].set_xlabel('Algorithm', fontsize=18)
        axs[0].set_ylim(0.9*min(fuel_consumptions), 1.1*max(fuel_consumptions))
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Passenger Comfort Analysis
        axs[1].bar(algorithms, avg_absolute_jerk, color='purple', label='Average Absolute Jerk')
        axs[1].bar(algorithms, avg_absolute_acceleration, color='orange', bottom=avg_absolute_jerk, label='Average Absolute Acceleration')
        axs[1].set_xlabel('Algorithm', fontsize=18)
        # axs[1].set_ylabel('|a| + |Jerk|', fontsize=18)
        axs[1].set_title('Passenger Comfort Analysis: Absolute Acceleration & Jerk', fontsize=22)
        axs[1].legend(fontsize=18)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        fig.set_size_inches(16, 10)
        self._save_figure(fig, "comfort_energy", algorithms)
        plt.show()




    def plot_agent_motion(self, agent_id, scenario_id):
        """
        Plot the acceleration, velocity, and energy consumption of a given agent over time in subplots.

        :param agent_id: Unique identifier for the agent
        :param scenario_id: Identifier for the scenario
        """
    
        
        accelerations = self.dataCollector.get_acceleration_time_series(agent_id, scenario_id)
        velocities = self.dataCollector.get_speed_time_series(agent_id, scenario_id)
        energy_consumption , total_consumption = self.dataCollector.get_energy_consumption_time_series(agent_id, scenario_id)
        dt = self.dataCollector.get_time_step(agent_id, scenario_id)
        print(dt)
        print(self.dataCollector.get_total_distance(agent_id, scenario_id))
        
        time = list(accumulate(dt))


        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
        
        # add grid title the self.algorithm_identifier
        fig.suptitle(self.algorithm_identifier)
        

        # Plot acceleration
        axes[0].plot(time, accelerations, marker='o', linestyle='-', color='b', label="Acceleration")
        axes[0].set_ylabel("Acceleration (m/s²)")
        axes[0].set_title(f"Acceleration, Velocity, and Energy Consumption of Agent {agent_id} Over Time")
        axes[0].legend()
        axes[0].grid(True)

        # Plot velocity
        axes[1].plot(time, velocities, marker='s', linestyle='-', color='r', label="Velocity")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()
        axes[1].grid(True)

        # Plot energy consumption
        axes[2].plot(time, energy_consumption, marker='d', linestyle='-', color='g', label="Energy Consumption")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Energy Consumption (kWh)")
        axes[2].set_title(f"Total energy consumption: {total_consumption, sum(energy_consumption)} kWh/km")
        axes[2].legend()
        axes[2].grid(True)
        

        plt.show(block=False)
