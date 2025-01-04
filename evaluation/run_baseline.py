import pathlib
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from evaluation.evaluation_simulation import run_evaluation_simulation, parse_arguments

if __name__ == '__main__':
    args = parse_arguments()


    agents_spec = {
    }
    
    scenarios_path_base = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "traffic_lights" 


    for baseline_algo in ["FTTL1", "FTTL2", "FTTLOPT", "ATL1", "ATL2"]:
        print(f"Running evaluation for {baseline_algo}")
        scenarios = [str(scenario) for scenario in (scenarios_path_base / baseline_algo).iterdir() if not scenario.is_file()]
        run_evaluation_simulation(scenarios, agents_spec, baseline_algo, args.seed, True)



