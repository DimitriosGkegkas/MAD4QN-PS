from os import wait

from sympy import plot
from statistics.experiment_data_collector import ExperimentDataCollector
from statistics.statistics_plotter import  StatisticsPlotter
import random
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # plotter = StatisticsPlotter("FTTL1")
    # plotter.plot_agent_motion(agent_id="car-edge-east-EW_0_10-edge-west-EW_0_60--71865--86686-0-0.0", scenario_id=0)
    

    # plotter = StatisticsPlotter("Centralized")
    # plotter.plot_agent_motion(agent_id="car-edge-east-EW_0_10-edge-west-EW_0_60--71865--86686-0-0.0", scenario_id=0)
    
    
    plotter = StatisticsPlotter("DropOutLayer2")
    plotter.plot_agent_motion(agent_id="Agent-0", scenario_id=0)
    plotter.plot_agent_motion(agent_id="Agent-1", scenario_id=0)
    
    plotter = StatisticsPlotter("keepLane")
    plotter.plot_agent_motion(agent_id="Agent-0", scenario_id=0)
    plotter.plot_agent_motion(agent_id="Agent-1", scenario_id=0)
    # plotter.plot_agent_motion(agent_id="car-edge-east-EW_0_10-edge-west-EW_0_60--71865--86686-0-0.0", scenario_id=0)
    # plotter.plot_agent_motion(agent_id="car-edge-north-NS_0_10-edge-south-NS_0_60--44830-527665-2-0.0", scenario_id=0)
    
    
    
    # plotter = StatisticsPlotter("FTTLOPT")
    # plotter.plot_agent_motion(agent_id="car-edge-north-NS_0_10-edge-south-NS_0_60--44830-527665-2-0.0", scenario_id=0)
    
    
    plt.show()



# from statistics.experiment_data_collector import ExperimentDataCollector
# from statistics.statistics_plotter import StatisticsPlotter
# import random


# if __name__ == "__main__":

#     algorithms = [
#         "ATL1",
#         "ATL2",
#         "FTTL1",
#         "FTTL2",
#         "FTTLOPT",
#         "Centralized",
#         #     "DuelingDDQNAgentsNoParaller",
#         #     "DuelingDDQNAgentsNoRandomLR",
#         # "MAD4QN-PS",
#         # "DropOutLayer2",
#         # "DropOutLayer2-v1",
#         # "keepLane",
#         #   "keepLaneSlow",
#         #   "DuelingDDQNAgents1",
#         #     "DuelingDDQNAgents-v1",
#         #     "DuelingDDQNAgentsNoRandom",
#         #     "DuelingDDQNAgentsNoParaller"
#     ]

#     # Initialize the plotter
#     plotter = StatisticsPlotter()

#     # Add algorithms (most recent data is selected by default)
#     for algo in algorithms:
#         plotter.add_algorithm(algo)

#     # Plot average travel time comparison
#     plotter.plot_combined_statistics()


# import numpy as np
# from energy.electric_vehicle_energy_model import ElectricVehicleEnergyModel


# t = ElectricVehicleEnergyModel()
    
# times = np.linspace(0, 10000, 1000)
# velocity = np.concatenate((np.linspace(0, 34, 30), 34 * np.ones(10000 - 30)))
# prev_speed = 0
# for  speed in velocity:
    
#     t.process_time_step(speed, speed - prev_speed, 0.1)
#     prev_speed = speed
# print(t.get_energy_consumption_per_km())