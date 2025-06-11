from typing import Dict, List, Any, Optional, Tuple
import os
import numpy as np
from Agent import Agent
from dataclasses import dataclass

@dataclass
class AgentConfig:
    input_dim: Any
    n_actions: int = 1
    gamma: float = 0.99
    lr: float = 1e-4
    tau: float = 1e-3
    batch_size: int = 64
    mem_size_factor: float = 1.5

class AgentManager:
    def __init__(
        self,
        agent_names: List[str],
        algorithm_identifier: str,
        agent_config: AgentConfig,
        evaluate: bool = False,
        base_dir: str = "models",
    ):
        self.agent_names = agent_names
        self.algorithm_identifier = algorithm_identifier
        self.evaluate = evaluate
        self.base_dir = base_dir

        self.agent: Agent = None
        
        mem_size = 1 if self.evaluate else int(1e5)
        chkpt_dir = self.base_dir if self.evaluate else os.path.join(
            self.base_dir, self.algorithm_identifier
        )
        os.makedirs(chkpt_dir, exist_ok=True)

        self.agent = Agent(
            input_dim=agent_config.input_dim,
            n_actions=agent_config.n_actions,
            gamma=agent_config.gamma,
            lr=agent_config.lr,
            tau=agent_config.tau,
            batch_size=agent_config.batch_size,
            max_size=int(mem_size * agent_config.mem_size_factor),
            chkpt_dir=chkpt_dir,
        )

    def choose_action(
        self,
        state: np.ndarray,
        direction: np.ndarray,
        messages: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.agent.choose_action(state, direction, messages, self.evaluate)

    def select_actions(
        self,
        direction: Dict[str, np.ndarray],
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
                    messages[agent]
                )
                actions[agent] = action
                new_messages[agent] = message
                raw_inputs[agent] = raw_input
        return actions, new_messages, raw_inputs

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
        for  dir_, obs, msg, term, trunc in zip(
         self.direction, state, messages, terminate, truncated
        ):
            actions, next_messages, next_raw_messages = self.select_actions(dir_, obs, msg, term, trunc)
            batch_actions.append(actions)
            batch_new_messages.append(next_messages)
            batch_raw_inputs.append(next_raw_messages)
        return batch_actions, batch_new_messages, batch_raw_inputs

    def store_transitions(
        self,
        observations: Dict[str, np.ndarray],
        messages: Dict[str, List[np.ndarray]],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        next_messages: Dict[str, List[np.ndarray]],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool]
    ) -> None:
        for agent in self.agent_names:
            # TODO check if next_observations is necessary
            if agent in observations and agent in next_observations:
                done = terminated[agent] or truncated[agent]
                self.agent.store_transition(
                    current_state=observations[agent],
                    current_messages=messages[agent],
                    direction=self.direction[agent],
                    action=actions[agent],
                    reward=rewards[agent],
                    next_state=next_observations[agent],
                    next_messages=next_messages[agent],
                    done=done
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
        critic, recon, smooth, policy, entropy = self.agent.learn()
        if logger:
            logger.log_scalar("loss/critic", critic, step)
            logger.log_scalar("loss/reconstruction", recon, step)
            logger.log_scalar("loss/smoothness", smooth, step)
            logger.log_scalar("loss/policy", policy, step)
            logger.log_scalar("loss/entropy", entropy, step)

    def save(self) -> None:
        self.agent.save()

    def load(self, path: str, evaluate: bool = False) -> None:
        self.agent.load(path, evaluate)
