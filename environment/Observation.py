import gymnasium as gym
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple

from utils.debug import debug_observation, debug_save_any_img


class Observation(gym.ObservationWrapper):
    def __init__(
        self, 
        shape: Tuple[int, int, int], 
        env: gym.Env, 
        agent_names: List[str] = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
    ) -> None:
        """
        Initializes the Observation wrapper.

        Args:
            shape (Tuple[int, int, int]): The desired shape of the observation (channels, height, width).
            env (gym.Env): The environment to wrap.
            agent_names (List[str], optional): List of agent names to process. Defaults to ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3'].
        """
        super().__init__(env)
        self.shape: Tuple[int, int, int] = (shape[2], shape[0], shape[1])  # (channels, height, width)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )
        self.agent_names: List[str] = agent_names

    def normalize_top_down_rgb(self, top_down_rgb: np.ndarray) -> np.ndarray:
        """
        Formats the top-down RGB observation.

        Args:
            top_down_rgb (np.ndarray): The top-down RGB observation.

        Returns:
            np.ndarray: The formatted top-down RGB observation.
        """
        # Resize the observation to the target shape
        resized_obs = cv2.resize(top_down_rgb, self.shape[1:], interpolation=cv2.INTER_AREA)
        
        # Normalize the resized observation
        normalized_obs = np.array(resized_obs, dtype=np.uint8).reshape(self.shape) / 255.0
        return normalized_obs

    def observation(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Processes the observation from the environment.

        Args:
            obs (Dict[str, Dict[str, Any]]): Raw observations from the environment. Each agent's observation 
                                            is expected to have a "top_down_rgb" key.

        Returns:
            Dict[str, np.ndarray]: Processed observations where values are resized and normalized images.
        """
        # Process observations for each agent
        new_obs = {
            agent_name: self.normalize_top_down_rgb(agent_data["top_down_rgb"])
                for agent_name, agent_data in obs.items()
                if "top_down_rgb" in agent_data
        }
        return new_obs

