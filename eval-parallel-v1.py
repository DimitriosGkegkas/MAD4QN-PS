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
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = False
    

    trainer = MultiAgentTrainer_v1(args, num_env=9, algorithm_identifier='DuelingDDQNAgents-v1')
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=None, top_down_rgb=True),
        )
    )
    trainer.initialize_agents(
        base_dir='models/DuelingDDQNAgents-v1/26012025',
        evaluation=True
    )
    # test, test1 = trainer._envision_episode(1)

    trainer.collect_statistics(parallel=True)

    # exit the program
    exit(0)
