from statistics.experiment_data_collector import ExperimentDataCollector
from statistics.statistics_plotter import  StatisticsPlotter
import random

def run_test1():
    """
    Run the test for a given algorithm identifier.
    """
    collector = ExperimentDataCollector("algorithm_A")

    # Run 10 scenarios
    for scenario_index in range(10):
        print(f"Running scenario {scenario_index + 1} for algorithm_A...")

        # Initialize agents with types
        collector.start_new_scenario({
            1: "type_A",
            2: "type_B",
            3: "type_C",
        })

        # Simulate 100 steps for each scenario
        for step in range(100):
            dt = 0.1
            collector.record_agent_data(1, speed=random.uniform(8, 12), acceleration=random.uniform(2, 4), dt=dt, travel_distance=dt * 10, is_waiting=False)
            collector.record_agent_data(2, speed=random.uniform(8, 12), acceleration=random.uniform(2, 4), dt=dt, travel_distance=0, is_waiting=(step % 10 == 0))
            collector.record_agent_data(3, speed=random.uniform(0, 5), acceleration=random.uniform(0, 1), dt=dt, travel_distance=0, is_waiting=True)

        # Mark outcomes for agents
        collector.mark_agent_succeeded(1)
        collector.mark_agent_succeeded(2)
        collector.mark_agent_crashed(3)
        # Agent 3 remains "traveling"

        # Close the scenario
        collector.close_scenario()

    # Get and save statistics
    stats = collector.get_statistics()
    print(f"Statistics for algorithm_A: {stats}")
    collector.save()

def run_test2():
    """
    Run the test for a given algorithm identifier.
    """
    collector = ExperimentDataCollector("algorithm_B")

    # Run 10 scenarios
    for scenario_index in range(10):
        print(f"Running scenario {scenario_index + 1} for algorithm_B...")

        # Initialize agents with types
        collector.start_new_scenario({
            1: "type_A",
            2: "type_B",
            3: "type_C",
        })

        # Simulate 100 steps for each scenario
        for step in range(100):
            dt = 0.1
            collector.record_agent_data(1, speed=random.uniform(10, 12), acceleration=random.uniform(12, 14), dt=dt, travel_distance=dt * 10, is_waiting=False)
            collector.record_agent_data(2, speed=random.uniform(10, 12), acceleration=random.uniform(12, 14), dt=dt, travel_distance=0, is_waiting=(step % 10 == 0))
            collector.record_agent_data(3, speed=random.uniform(10, 12), acceleration=random.uniform(10, 11), dt=dt, travel_distance=0, is_waiting=False)

        # Mark outcomes for agents
        collector.mark_agent_succeeded(1)
        collector.mark_agent_crashed(2)
        # Agent 3 remains "traveling"
        collector.mark_agent_succeeded(1)
        collector.mark_agent_succeeded(2)
        collector.mark_agent_succeeded(3)

        # Close the scenario
        collector.close_scenario()

    # Get and save statistics
    stats = collector.get_statistics()
    print(f"Statistics for algorithm_B: {stats}")
    collector.save()



if __name__ == "__main__":
    # Run tests for two different algorithms
    run_test1()
    run_test2()



    # Initialize the plotter
    plotter = StatisticsPlotter()

    # Add algorithms (most recent data is selected by default)
    plotter.add_algorithm("random_0")
    plotter.add_algorithm("mad4qn_0")

    # Plot average travel time comparison
    plotter.plot_combined_statistics()
