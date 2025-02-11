import pathlib
import argparse
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from evaluation.evaluation_simulation import run_evaluation_simulation, parse_arguments
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel

class RandomAgent(Agent):
    def __init__(self):
        """Any policy initialization matters, including loading of model,
        may be performed here.
        """
        pass

    def act(self, obs, turning_intention):
        return np.random.randint(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = True

    # AgentSpec specifying the agent's interface and policy.
    agent_spec = AgentSpec(
        # Agent's interface.
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        # Agent's policy.
        agent_builder=RandomAgent,
        agent_params=None, # Optional parameters passed to agent_builder during building.
    )
    scenarios_path = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "multi_scenario"
    trainer = MultiAgentTrainerParallel(args, num_env=27, agent_count=0, algorithm_identifier="Random")
    trainer.initialize_environment(
        agent_spec,
        scenario_subdir=scenarios_path,
        parallel=False,
    )
    trainer.full_eval(parallel=False)

