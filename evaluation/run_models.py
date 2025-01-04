import pathlib
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from evaluation.evaluation_simulation import run_evaluation_simulation, parse_arguments
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    class ModelAgent(Agent):
        def __init__(self):
            """Any policy initialization matters, including loading of model,
            may be performed here.
            """
            mem_size = 1
            input_dims = (9, 48, 48)
            self.agents_straight = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=input_dims,
                                            n_actions=2, mem_size=int(mem_size * 1.5), eps_min=0.01,
                                            batch_size=256, replace=1000, eps_dec=1e-6, chkpt_dir='models',
                                            algo='DuelingDDQNAgents',
                                            env_name=f'agent_straight_{args.seed}')
            self.agents_left = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=input_dims,
                                        n_actions=2, mem_size=int(mem_size * 1.5), eps_min=0.01,
                                        batch_size=256,
                                        replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                        env_name=f'agent_left_{args.seed}')
            self.agents_right = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=input_dims,
                                            n_actions=2, mem_size=mem_size, eps_min=0.01, batch_size=256,
                                            replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                            env_name=f'agent_right_{args.seed}')

            self.agents_straight.load_models()
            self.agents_left.load_models()
            self.agents_right.load_models()

        def act(self, obs, turning_intention):
            if turning_intention == 'straight':
                return self.agents_straight.choose_action(obs, evaluate=True)
            elif turning_intention == 'left':
                return self.agents_left.choose_action(obs, evaluate=True)
            elif turning_intention == 'right':
                return self.agents_right.choose_action(obs, evaluate=True)

    # AgentSpec specifying the agent's interface and policy.
    agent_spec = AgentSpec(
        # Agent's interface.
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        # Agent's policy.
        agent_builder=ModelAgent,
        agent_params=None, # Optional parameters passed to agent_builder during building.
    )

    agents_spec = {
            f"Agent-{i}": agent_spec for i in range(4)  # Supports four agents by default
    }
    
    scenarios_path = pathlib.Path(__file__).absolute().parent.parent / "scenarios" / "sumo" / "multi_scenario"
    scenarios = [str(scenario) for scenario in scenarios_path.iterdir() if not scenario.is_file()]
    scenarios.sort()


    run_evaluation_simulation(scenarios, agents_spec, "mad4qn", args.seed, False)



