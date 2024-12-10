import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class StatisticsPlotter:
    def __init__(self):
        """
        Initialize the plotter with storage for algorithm results.
        """
        self.results = {}  # Store statistics by algorithm identifier

    def add_algorithm(self, algorithm_identifier, date=None):
        """
        Add results for a specific algorithm. If no date is specified, the most recent date is used.
        :param algorithm_identifier: Name of the algorithm.
        :param date: Optional date in DDMMYYYY format. Defaults to the most recent.
        """
        base_dir = "data"
        algorithm_dir = os.path.join(base_dir, algorithm_identifier)

        if not os.path.exists(algorithm_dir):
            raise ValueError(f"No data found for algorithm '{algorithm_identifier}'.")

        # Find the most recent date folder if none is specified
        if date is None:
            dates = [
                d for d in os.listdir(algorithm_dir) if os.path.isdir(os.path.join(algorithm_dir, d))
            ]
            if not dates:
                raise ValueError(f"No date folders found for algorithm '{algorithm_identifier}'.")
            date = max(dates, key=lambda d: datetime.strptime(d, "%d%m%Y"))  # Most recent date

        # Load the total.npy file
        data_dir = os.path.join(algorithm_dir, date)
        total_file = os.path.join(data_dir, "total.npy")

        if not os.path.exists(total_file):
            raise ValueError(f"No 'total.npy' file found for algorithm '{algorithm_identifier}' on date '{date}'.")

        # Load statistics and store
        self.results[algorithm_identifier] = np.load(total_file, allow_pickle=True).item()

    def _save_figure(self, figure, name, algorithms):
        """
        Save a figure to the appropriate directory.
        :param figure: The matplotlib figure to save.
        :param name: Name of the figure.
        :param algorithms: List of algorithm identifiers used in the figure.
        """
        base_dir = "figures"
        sorted_algorithms = "_".join(sorted(algorithms))
        folder_path = os.path.join(base_dir, sorted_algorithms)

        # Subfolder for the current date (DDMMYYYY format)
        current_date = datetime.now().strftime("%d%m%Y")
        date_folder = os.path.join(folder_path, current_date)
        os.makedirs(date_folder, exist_ok=True)

        # Save the figure
        file_path = os.path.join(date_folder, f"{name}.png")
        figure.savefig(file_path)
        print(f"Saved figure to {file_path}")

    def plot_travel_and_waiting_time(self):
        """
        Plot travel time and waiting time as grouped bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        travel_times = [self.results[algo].get("average_travel_time", 0) for algo in algorithms]
        waiting_times = [self.results[algo].get("average_waiting_time", 0) for algo in algorithms]

        # Create a bar plot
        x = np.arange(len(algorithms))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, travel_times, width, label='Travel Time', color='skyblue')
        ax.bar(x + width/2, waiting_times, width, label='Waiting Time', color='lightgreen')
        ax.set_title('Travel and Waiting Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('Time (seconds)')
        ax.legend()
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
        ax.set_title('Success, Crash, and Incomplete Rates')
        ax.set_ylabel('Percentage (%)')
        ax.legend()
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "success_crash_incomplete_rates", algorithms)
        plt.show()

    def plot_fuel_consumption(self):
        """
        Plot fuel consumption (energy) as bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        fuel_consumptions = [self.results[algo].get("average_energy_consumption", 0) for algo in algorithms]

        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(algorithms, fuel_consumptions, color='gold')
        ax.set_title('Fuel Consumption Comparison')
        ax.set_ylabel('Energy Consumption (Wh)')
        ax.set_xlabel('Algorithm')
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "fuel_consumption", algorithms)
        plt.show()

    def plot_speed_and_acceleration(self):
        """
        Plot average speed and acceleration as grouped bar plots for all added algorithms.
        """
        algorithms = list(self.results.keys())
        avg_speeds = [self.results[algo].get("average_speed", 0) for algo in algorithms]
        avg_accelerations = [self.results[algo].get("average_acceleration", 0) for algo in algorithms]

        # Create a bar plot
        x = np.arange(len(algorithms))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, avg_speeds, width, label='Average Speed', color='cornflowerblue')
        ax.bar(x + width/2, avg_accelerations, width, label='Average Acceleration', color='coral')
        ax.set_title('Speed and Acceleration Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('Values')
        ax.legend()
        plt.grid(axis='y')

        # Save and show
        self._save_figure(fig, "speed_and_acceleration", algorithms)
        plt.show()

    def plot_combined_statistics(self):
        """
        Create a combined plot with subplots comparing algorithms across multiple metrics.
        """
        algorithms = list(self.results.keys())

        travel_times = [self.results[algo].get("average_travel_time", 0) for algo in algorithms]
        waiting_times = [self.results[algo].get("average_waiting_time", 0) for algo in algorithms]
        success_rates = [self.results[algo].get("success_rate", 0) for algo in algorithms]
        crash_rates = [self.results[algo].get("crash_rate", 0) for algo in algorithms]
        incomplete_rates = [self.results[algo].get("incomplete_rate", 0) for algo in algorithms]
        fuel_consumptions = [self.results[algo].get("average_energy_consumption", 0) for algo in algorithms]
        avg_speeds = [self.results[algo].get("average_speed", 0) for algo in algorithms]
        avg_accelerations = [self.results[algo].get("average_acceleration", 0) for algo in algorithms]

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Top Left: Travel Time and Waiting Time
        x = np.arange(len(algorithms))
        width = 0.35
        axs[0, 0].bar(x - width/2, travel_times, width, label='Average Travel Time', color='skyblue')
        axs[0, 0].bar(x + width/2, waiting_times, width, label='Average Waiting Time', color='lightgreen')
        axs[0, 0].set_title('Travel and Waiting Time')
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(algorithms)
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].legend()

        # Top Right: Success, Crash, and Incomplete Rates
        axs[0, 1].bar(algorithms, success_rates, label='Average Success Rate', color='lightgreen')
        axs[0, 1].bar(algorithms, crash_rates, bottom=success_rates, label='Average Crash Rate', color='salmon')
        axs[0, 1].bar(algorithms, incomplete_rates,
                       bottom=[success_rates[i] + crash_rates[i] for i in range(len(success_rates))],
                       label='Average Incomplete Rate', color='lightgray')
        axs[0, 1].set_title('Success, Crash, and Incomplete Rates')
        axs[0, 1].set_ylabel('Average Percentage (%)')
        axs[0, 1].legend()

        # Bottom Left: Fuel Consumption
        axs[1, 0].bar(algorithms, fuel_consumptions, color='gold')
        axs[1, 0].set_title('Energy Consumption')
        axs[1, 0].set_ylabel('Average Energy Consumption (Wh/Km)')
        axs[1, 0].set_xlabel('Algorithm')

        # Bottom Right: Speed and Acceleration
        axs[1, 1].bar(x - width/2, avg_speeds, width, label='Average Speed', color='cornflowerblue')
        axs[1, 1].bar(x + width/2, avg_accelerations, width, label='Average Acceleration', color='coral')
        axs[1, 1].set_title('Speed and Acceleration')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(algorithms)
        axs[1, 1].set_ylabel('Values')
        axs[1, 1].legend()

        plt.tight_layout()

        # Save and show
        self._save_figure(fig, "combined_statistics", algorithms)
        plt.show()
