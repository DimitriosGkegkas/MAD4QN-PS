import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel



class MultiAgentTrainer_v1 (MultiAgentTrainerParallel):
    def format_action(self, action):
        if action:
            return (0, 0)
        else:
            return (15, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = True
    

    trainer = MultiAgentTrainerParallel(args, num_env=9, algorithm_identifier='DuelingDDQNAgents-v1',evaluation_step=10)
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=None, top_down_rgb=True),
        )
    )
    trainer.initialize_agents(
        replace = 10000,
        eps_dec=1e-5,
        batch_size=2*256,
        eps_min=0.1,
    )
    trainer.train()