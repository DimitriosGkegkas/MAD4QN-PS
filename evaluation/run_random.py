import pathlib
import argparse
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel

class MultiAgentTrainer_v1 (MultiAgentTrainerParallel):
    def act(self, observation, turning_intention):
        return np.random.randint(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--headless', action='store_true', help='Not visualize the simulation')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = True

    trainer = MultiAgentTrainer_v1(args, num_env=9, algorithm_identifier='random')
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None, top_down_rgb=True),
        )
    )
    trainer.collect_statistics(parallel=True)



