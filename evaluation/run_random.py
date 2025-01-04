import pathlib
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from evaluation.evaluation_simulation import run_evaluation_simulation, parse_arguments


class RandomAgent(Agent):
    def __init__(self):
        """Any policy initialization matters, including loading of model,
        may be performed here.
        """
        pass

    def act(self, obs, turning_intention):
        return np.random.randint(2)


if __name__ == '__main__':
    args = parse_arguments()

    # AgentSpec specifying the agent's interface and policy.
    agent_spec = AgentSpec(
        # Agent's interface.
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        # Agent's policy.
        agent_builder=RandomAgent,
        agent_params=None, # Optional parameters passed to agent_builder during building.
    )

    agents_spec = {
            f"Agent-{i}": agent_spec for i in range(4)  # Supports four agents by default
    }
    
    scenarios_path = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "multi_scenario"
    scenarios = [str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()]


    run_evaluation_simulation(scenarios, agents_spec, "random", args.seed, True)



