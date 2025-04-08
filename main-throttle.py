import pathlib
import argparse
from re import A
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel
from smarts.core.controllers import ActionSpaceType
import os

from ddpg.ddpg_agent import DDPGAgent


class MultiAgentTrainer_v1 (MultiAgentTrainerParallel):
    def format_action(self, action):
        # print(action)
        return action[0]
    def initialize_agents(
        self,
        batch_size=256/4,
        gamma=0.99,
alpha=0.0001, beta=0.001, 
         tau=0.001,
        mem_size_factor=1.5,
        n_actions=1,
        base_dir='models',
    ):
        mem_size = 1 if self.evaluate else 1e5
        if self.evaluate:
            chkpt_dir = base_dir
            assert os.path.exists(chkpt_dir), f"Checkpoint directory {chkpt_dir} does not exist"
        else:
            chkpt_dir = os.path.join(
                base_dir, self.algorithm_identifier, self.timestamp
            )
            os.makedirs(chkpt_dir, exist_ok=True)

        input_dims = self.env.observation_space.shape
        agent_params = {
            'alpha': alpha,
            'beta': beta,
            'input_dims': input_dims,   
            'tau': tau,    
            'n_actions': n_actions,
            'gamma': gamma,
            'max_size': int(mem_size * mem_size_factor),
            'batch_size': batch_size,
            'algo': self.algorithm_identifier,
            'chkpt_dir': chkpt_dir,
            'training_stats_path': self.training_stats_path,
        }
        self.agents = {
            'straight': DDPGAgent(
                **agent_params,
                env_name=f'agent_straight'
            ),
            'left': DDPGAgent(
                **agent_params,
                env_name=f'agent_left'
            ),
            'right': DDPGAgent(
                **agent_params,
                env_name=f'agent_right'
            )
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = False
    

    trainer = MultiAgentTrainer_v1(args, num_env=27, algorithm_identifier='throttle', evaluation_step=5)
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
    )
    
    trainer.train()
    # trainer.envision(10)