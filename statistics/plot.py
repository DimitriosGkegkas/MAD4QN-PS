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
        "CENTRAL",
        "RP",
        "MAD4QN",
        "MAD4QN-v1",
    ]
    # Initialize the plotter
    # plotter = StatisticsPlotter()

    # # Add algorithms (most recent data is selected by default)
    # for algo in algorithms:
    #     plotter.add_algorithm(algo)

    # # Plot average travel time comparison
    # plotter.plot_travel_speed_success()
    # # plotter.plot_combined_statistics()
    # plotter.plot_comfort_energy()

    
    plotter = StatisticsPlotter("CENTRAL")
    plotter.plot_agent_motion("car-edge-west-WE_0_10-edge-north-SN_0_60--51344-139011-1-0.0", 5)
    
        
    plotter = StatisticsPlotter("MAD4QN-v1")
    plotter.plot_agent_motion("Agent-2", 5)
        
    # car-edge-east-EW_0_10-edge-west-EW_0_60--71865--86686-0-0.0
    # car-edge-west-WE_0_10-edge-south-NS_0_60-668089--43872-1-0.0
    # algorithms = [
    #     "ATL1",
    #     "ATL2",
    #     "FTTL1",
    #     "FTTL2",
    #     "FTTLOPT",
    #     "CENTRAL",
    #     "RP",
    #     "MAD4QN",
    #     "MAD4QN-v1",
    #     "DuelingDDQNAgents",
    # ]

    # # Initialize the plotter
    # plotter = StatisticsPlotter()

    # # Add algorithms (most recent data is selected by default)
    # for algo in algorithms:
    #     plotter.add_algorithm(algo)

    # # Plot average travel time comparison
    # # plotter.plot_travel_speed_success()
    # # plotter.plot_combined_statistics()
    # plotter.plot_comfort_energy()