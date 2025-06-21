from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from omegaconf import DictConfig
from agent import Agent

from .BaseTrainer import BaseTrainer

from environment.types import ObservationType


class AgentManager:
    def __init__(
        self,
        agent_names: List[str],
        cfg: DictConfig,
        logger: BaseTrainer,
    ):
        self.agent_names = agent_names
        self.evaluate = cfg.evaluate
        self.message_dim = cfg.message_dim
        self.parallel = cfg.parallel
        self.logger = logger
        

        self.agent = Agent(cfg)
    
    
    def action(
        self,
        state: Union[Dict[str, ObservationType], Tuple[Dict[str, np.ndarray]]],
        terminate: Union[Dict[str, bool], Tuple[Dict[str, bool]]],
        truncated: Union[Dict[str, bool], Tuple[Dict[str, bool]]],
    ):
        is_batch = not isinstance(state, Dict)

        if is_batch:
            return self.select_batch_actions(state, terminate, truncated)
        else:
            return self.select_actions(state, terminate, truncated)


    def select_batch_actions(
        self,
        state: List[Dict[str, np.ndarray]],
        terminate: List[Dict[str, bool]],
        truncated: List[Dict[str, bool]]
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:

        batch_actions, batch_new_messages, batch_raw_inputs = [], [], []
        for  index, (obs, term, trunc) in enumerate(zip(
          state, terminate, truncated
        )):
            actions, next_messages, next_raw_messages = self.select_actions(obs, term, trunc, index)
            batch_actions.append(actions)
            batch_new_messages.append(next_messages)
            batch_raw_inputs.append(next_raw_messages)
        return batch_actions, batch_new_messages, batch_raw_inputs



    def select_actions(
        self,
        observations: Dict[str, ObservationType],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        env_index: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        actions, new_messages, raw_inputs = {}, {}, {}
        for agent in observations:
            if terminated[agent] or truncated[agent]:
                continue
            
            action, message, raw_input = self.agent.choose_action(
                observations[agent][0], 
                observations[agent][1], 
                observations[agent][2], 
                agent + str(env_index), 
                self.evaluate
                )
            
            actions[agent] = action
            new_messages[agent] = message
            raw_inputs[agent] = raw_input
            
        return actions, new_messages, raw_inputs
    
            
    #-------------------------------------------
    # Transition storage and update methods
    #-------------------------------------------        
    def store_transition(
        self,
        observations: Dict[str, ObservationType],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, ObservationType],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool]
    ) -> None:
        for agent in self.agent_names:
            if agent in observations and agent in next_observations:
                done = terminated[agent] or truncated[agent]
                self.agent.store_transition(
                    current_state=observations[agent][0],
                    current_messages=observations[agent][3],
                    direction=observations[agent][1],
                    action=actions[agent],
                    reward=rewards[agent],
                    next_state=next_observations[agent][0],
                    next_messages=next_observations[agent][3],
                    done=done
                )
    
    def store_transitions(
        self,
        current_state: List[Dict[str, ObservationType]],
        action: List[Dict[str, np.ndarray]],
        reward: List[Dict[str, float]],
        next_state: List[Dict[str, ObservationType]],
        terminate: List[Dict[str, bool]],
        truncated: List[Dict[str, bool]]
    ) -> None:
        if not isinstance(current_state, Dict):
            self.store_transition(
                current_state,
                action,
                reward,
                next_state,
                terminate,
                truncated
            )
        for i in range(len(current_state)):
            self.store_transition(
                current_state[i],
                action[i],
                reward[i],
                next_state[i],
                terminate[i],
                truncated[i]
            )


    
    def update_agent(self, step: int) -> None:
        self.agent.learn(self.logger)

    def save(self, best = True) -> None:
        if best:
            self.agent.save("best_checkpoint.pth")
        else:
            self.agent.save("checkpoint.pth")

    def load(self, path: str, evaluate: bool = False) -> None:
        self.agent.load(path, evaluate)
        
    def eval(self):
        """
        Set the agent to evaluation mode.
        This is a placeholder method, as the actual implementation may vary.
        """
        self.evaluate = True
        
    def train(self):
        """
        Set the agent to training mode.
        This is a placeholder method, as the actual implementation may vary.
        """
        self.evaluate = False
    