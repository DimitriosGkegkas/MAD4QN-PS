import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainer import MultiAgentTrainer


class MultiAgentTrainer_v1 (MultiAgentTrainer):
    def format_action(self, action):
        if action:
            return (0, 0)
        else:
            return (54, 0)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()

    trainer = MultiAgentTrainer_v1(args, algorithm_identifier='DuelingDDQNAgents-v1')
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=None, top_down_rgb=True),
        )
    )
    trainer.initialize_agents()
    trainer.train()
