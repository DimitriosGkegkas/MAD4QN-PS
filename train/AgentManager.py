from typing import Dict, List, Any, Optional, Tuple
import os
import numpy as np
import torch
from Agent import Agent, agent
from dataclasses import dataclass

from energy import config


class AgentManager:
    def __init__(
        self,
        agent_names: List[str],
        agent_config: agent.AgentConfig,
        evaluate: bool = False,
        parallel: bool = True
    ):
        self.agent_names = agent_names
        self.evaluate = evaluate
        self.message_dim = agent_config.message_dim
        self.parallel = parallel
        

        self.agent = Agent(agent_config)
        
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

    def choose_action(
        self,
        state: np.ndarray,
        direction: np.ndarray,
        messages: List[np.ndarray],
        agent: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        return self.agent.choose_action(state, direction, messages, agent, self.evaluate)
    
    
    def create_communication_message(self, agent: str, messages, communication: Dict[str, List[str]]) -> torch.Tensor:
        agent_msg = [
                        messages[communicating_agent] if communicating_agent in messages else None
                        for communicating_agent in communication[agent]
                    ]
        agent_msg = [
            np.zeros(self.message_dim) if msg is None else msg
            for msg in agent_msg
        ]
        aggregated_message = np.concatenate(agent_msg)
        return aggregated_message
    
    def create_communication_raw_message(self, agent: str, messages, communication: Dict[str, List[str]]) -> torch.Tensor:
        agent_msg = [
                        messages[communicating_agent] if communicating_agent in messages else None
                        for communicating_agent in communication[agent]
                    ]
        return agent_msg

    def select_actions(
        self,
        direction: Dict[str, np.ndarray],
        communication: Dict[str, List[str]],
        observations: Dict[str, np.ndarray],
        messages: Dict[str, List[np.ndarray]],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        actions, new_messages, raw_inputs = {}, {}, {}
        for agent in self.agent_names:
            if agent in observations and not terminated[agent] and not truncated[agent]:
                action, message, raw_input = self.choose_action(
                    observations[agent],
                    direction[agent],
                    self.create_communication_message(agent, messages, communication),
                    agent=agent
                )
                actions[agent] = action
                new_messages[agent] = message
                raw_inputs[agent] = raw_input
        return actions, new_messages, raw_inputs
    
    
    def action(
        self,
        state: Dict[str, np.ndarray],
        messages: Dict[str, List[np.ndarray]],
        terminate: Dict[str, bool],
        truncated: Dict[str, bool]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if self.parallel:
            return self.select_batch_actions(
                state, messages, terminate, truncated
            )
        return self.select_actions(
            self.direction, self.communication, state, messages, terminate, truncated
        )

    def select_batch_actions(
        self,
        state: List[Dict[str, np.ndarray]],
        messages: List[Dict[str, List[np.ndarray]]],
        terminate: List[Dict[str, bool]],
        truncated: List[Dict[str, bool]]
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, np.ndarray]],
        List[Dict[str, np.ndarray]]
    ]:
        batch_actions, batch_new_messages, batch_raw_inputs = [], [], []
        for  dir_, com_, obs, msg, term, trunc in zip(
         self.direction, self.communication, state, messages, terminate, truncated
        ):
            actions, next_messages, next_raw_messages = self.select_actions(dir_, com_, obs, msg, term, trunc)
            batch_actions.append(actions)
            batch_new_messages.append(next_messages)
            batch_raw_inputs.append(next_raw_messages)
        return batch_actions, batch_new_messages, batch_raw_inputs

            
            
            
    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        messages: Dict[str, List[np.ndarray]],
        direction: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        next_messages: Dict[str, List[np.ndarray]],
        communication: Dict[str, List[str]],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool]
    ) -> None:
        for agent in self.agent_names:
            # TODO check if next_observations is necessary
            if agent in observations and agent in next_observations:
                done = terminated[agent] or truncated[agent]
                # if done:
                #     print("hi")
                #     self.agent.embedded.visualize_head_output(torch.Tensor(observations[agent]).unsqueeze(0))
                self.agent.store_transition(
                    current_state=observations[agent],
                    current_messages=self.create_communication_raw_message(
                        agent, messages, communication
                    ),
                    direction=direction[agent],
                    action=actions[agent],
                    reward=rewards[agent],
                    next_state=next_observations[agent],
                    next_messages= self.create_communication_raw_message(
                        agent, next_messages, communication
                    ),
                    done=done
                )
    def store_transitions(
        self,
        current_state: List[Dict[str, np.ndarray]],
        current_messages: List[Dict[str, List[np.ndarray]]],
        action: List[Dict[str, np.ndarray]],
        reward: List[Dict[str, float]],
        next_state: List[Dict[str, np.ndarray]],
        next_messages: List[Dict[str, List[np.ndarray]]],
        terminate: List[Dict[str, bool]],
        truncated: List[Dict[str, bool]]
    ) -> None:
        for i in range(len(current_state)):
            self.store_transition(
                current_state[i],
                current_messages[i],
                self.direction[i],
                action[i],
                reward[i],
                next_state[i],
                next_messages[i],
                self.communication[i],
                terminate[i],
                truncated[i]
            )

    def set_communication(self, communication: Dict[str, Any]) -> None:
        """
        Set the communication for the agent.
        This is a placeholder method, as the actual implementation may vary.
        """
        self.communication = communication
        
    def set_direction(self, direction: Dict[str, np.ndarray]) -> None:
        """
        Set the direction for the agent.
        This is a placeholder method, as the actual implementation may vary.
        """
        self.direction = direction
        
    def update_agent(self, step: int, logger: Optional[Any] = None) -> None:
        self.agent.learn(logger)

    def save(self, best = True) -> None:
        if best:
            self.agent.save("best_checkpoint.pth")
        else:
            self.agent.save("checkpoint.pth")

    def load(self, path: str, evaluate: bool = False) -> None:
        self.agent.load(path, evaluate)
