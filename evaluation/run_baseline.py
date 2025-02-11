import pathlib
import argparse
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = True

    agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        )
    
    scenarios_path_base = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "traffic_lights" 


    for baseline_algo in ["FTTL1", "FTTL2", "FTTLOPT", "ATL1", "ATL2", "Centralized"]:
    # for baseline_algo in ["FTTL1"]:
        scenario_subdir = scenarios_path_base / baseline_algo
        trainer = MultiAgentTrainerParallel(args, num_env=27, agent_count=0, algorithm_identifier=baseline_algo)
        trainer.initialize_environment(
            agent_spec,
            scenario_subdir=scenario_subdir,
            parallel=False,
        )
        trainer.full_eval(parallel=False)