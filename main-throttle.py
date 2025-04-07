import pathlib
import argparse
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel
from smarts.core.controllers import ActionSpaceType

throttleLevel = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class MultiAgentTrainer_v1 (MultiAgentTrainerParallel):
    def format_action(self, action):
        return throttleLevel[action]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = False
    

    trainer = MultiAgentTrainer_v1(args, num_env=1, algorithm_identifier='throttle')
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.RawThrottle,
                max_episode_steps=None, 
                top_down_rgb=True
            )

        )
    )
    trainer.initialize_agents(
        batch_size=256,
        n_actions=len(throttleLevel),
    )
    trainer.train()
    # trainer.collect_statistics(parallel=True)
    # trainer.envision(16)
    # test, test = trainer._envision_episode(30)

