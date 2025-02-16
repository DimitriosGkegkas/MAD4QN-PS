import pathlib
import argparse
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent
from smarts.zoo.agent_spec import AgentSpec
from train.MultiAgentTrainerParallel import MultiAgentTrainerParallel

class MultiAgentTrainer_v1 (MultiAgentTrainerParallel):
    def _select_actions(self, turning_intentions, observations, terminated, truncated):
        agent_actions = {}
        now_go = len(observations.keys())  -1
        
        for idx, agent_name in enumerate(self.agent_names):
            if agent_name in observations and not terminated[agent_name] and not truncated[agent_name]:
                go = str(now_go) in agent_name
                if(go):
                    agent_actions[agent_name] = self.act(observations[agent_name], 0)
                elif (str(now_go - 1) in agent_name):
                    agent_actions[agent_name] = self.act(observations[agent_name], 1)
                else:
                    agent_actions[agent_name] = self.act(observations[agent_name], 2)
        return agent_actions
    
    def act(self, observation, go):
        if( go == 0):
            return (13,0)
        elif (go == 1):
            return (5,0)
        else:
            return (1,0)
    
    def format_action(self, action):
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
    args = parser.parse_args()
    args.headless = False
    

    trainer = MultiAgentTrainer_v1(args, num_env=3, algorithm_identifier='keepLaneSlow')
    trainer.initialize_environment(
        AgentSpec(
            interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=None, top_down_rgb=True),
        )
    )
    trainer.full_eval(parallel=True)
    # test, test = trainer._envision_episode(30)

