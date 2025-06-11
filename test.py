import pathlib
import argparse
from re import A
import numpy as np
from smarts.core.agent_interface import AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainer import MultiAgentTrainerParallel, TrainerConfig
from smarts.core.controllers import ActionSpaceType

if __name__ == '__main__':
    config = TrainerConfig(
        total_steps=2_000_000,
        agent_count=6,
        algorithm_identifier="MyCustomAgent",
        evaluate=True,
        num_env=4,
        agent_spec=AgentSpec(
            interface=AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.RawThrottle,
                max_episode_steps=None, 
                top_down_rgb=True
            ),
        ),
    )

    trainer = MultiAgentTrainerParallel(config)

    trainer.train()
    