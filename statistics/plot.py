from statistics.experiment_data_collector import ExperimentDataCollector
from statistics.statistics_plotter import StatisticsPlotter
import random


if __name__ == "__main__":

    algorithms = [
        "ATL1",
        "ATL2",
        "FTTL1",
        "FTTL2",
        "FTTLOPT",
        "Centralized",
        # "keepLane",
        "random",
        #     "DuelingDDQNAgentsNoParaller",
        #     "DuelingDDQNAgentsNoRandomLR",
        # "MAD4QN-PS",
        "DropOutLayer2",
        "DropOutLayer2-v1",
        # "keepLane",
        #   "keepLaneSlow",
        #   "DuelingDDQNAgents1",
        #     "DuelingDDQNAgents-v1",
        #     "DuelingDDQNAgentsNoRandom",
        #     "DuelingDDQNAgentsNoParaller"
    ]

    # Initialize the plotter
    plotter = StatisticsPlotter()

    # Add algorithms (most recent data is selected by default)
    for algo in algorithms:
        plotter.add_algorithm(algo)

    # Plot average travel time comparison
    plotter.plot_combined_statistics()
