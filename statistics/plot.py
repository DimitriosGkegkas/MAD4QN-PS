from statistics.experiment_data_collector import ExperimentDataCollector
from statistics.statistics_plotter import  StatisticsPlotter
import random


if __name__ == "__main__":

    algorithms = [
        # "ATL1", 
                    # "ATL2",
                    # "FTTL1",
                    # "FTTL2",
                    # "FTTLOPT",
                
                    # "Central",
                #     "DuelingDDQNAgentsNoParaller",
                #     "DuelingDDQNAgentsNoRandomLR",
                #   "MAD4QN-PS",
                "keepLane",
                  "keepLaneSlow",
                #   "DuelingDDQNAgents1",
                #     "DuelingDDQNAgents-v1",
                #     "DuelingDDQNAgentsNoRandom",
                #     "DuelingDDQNAgentsNoParaller"
                  ]

    for algo in algorithms:
        data_collector = ExperimentDataCollector(algo)
        try:
            data_collector.load_raw_data()
            data_collector.save()
        except:
            print(f"Could not load data for {algo}")
            continue



    # Initialize the plotter
    plotter = StatisticsPlotter()

    # Add algorithms (most recent data is selected by default)
    for algo in algorithms:
        plotter.add_algorithm(algo)

    # Plot average travel time comparison
    plotter.plot_combined_statistics()
