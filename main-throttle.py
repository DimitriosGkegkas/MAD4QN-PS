import pathlib
import argparse
from re import A
import numpy as np
from smarts.core.agent_interface import AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerThrottle import MultiAgentTrainerThrottle
from smarts.core.controllers import ActionSpaceType

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = False
    

    trainer = MultiAgentTrainerThrottle(args, num_env=9, algorithm_identifier='throttle', evaluation_step=10, agent_count = 4)
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.RawThrottle,
                max_episode_steps=None, 
                top_down_rgb=True
            ),

        )
    )
    trainer.initialize_agents(
        batch_size=64,
    )  
    trainer.preload("models/maneu")
    trainer.train()
    # trainer.envision(0)
    