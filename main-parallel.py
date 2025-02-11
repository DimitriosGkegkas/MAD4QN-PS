import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = True
    

    trainer = MultiAgentTrainerParallel(args, num_env=1, algorithm_identifier='DuelingDDQNAgents',evaluation_step=100)
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        ),
    )
    trainer.initialize_agents()
    trainer.train()
