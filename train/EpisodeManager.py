from typing import Dict, List, Any

class EpisodeManager:
    def __init__(self, agent_names: List[str], parallel: bool = True):
        self.agent_names = agent_names
        self.parallel = parallel
        
        
    def is_done(
        self,
        observations: Dict[str, Any],
        rewards: Dict[str, float],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, Any],
    ) -> bool:
        """
        Determines if the episode is done based on reward signals,
        observation presence, termination flags, and custom info flags.
        """
        if self.parallel:
            return self.is_batch_episode_done(observations, rewards, terminated, truncated, info)
        return self.is_episode_done(observations, rewards, terminated, truncated, info)

    def is_episode_done(
        self,
        observations: Dict[str, Any],
        rewards: Dict[str, float],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, Any],
    ) -> bool:
        """
        Determines if a single episode is done based on reward signals,
        observation presence, termination flags, and custom info flags.
        """
        return self._crashed(truncated) or self._is_episode_terminated(observations, terminated, info)

    def is_batch_episode_done(
        self,
        batch_observations: List[Dict[str, Any]],
        batch_rewards: List[Dict[str, float]],
        batch_terminated: List[Dict[str, bool]],
        batch_truncated: List[Dict[str, bool]],
        batch_infos: List[Dict[str, Any]],
    ) -> bool:
        """
        Determines if all episodes in a batch are done.
        """
        return all(
            self.is_episode_done(obs, rew, term, trun, inf)
            for obs, rew, term, trun, inf in zip(batch_observations, batch_rewards, batch_terminated, batch_truncated, batch_infos)
        )

    def _crashed(self, truncated: Dict[str, bool]) -> bool:
        """
        Checks if any agent has crashed based on the truncated flags.
        """
        return any(truncated.values())

    def _is_episode_terminated(
        self,
        observations: Dict[str, Any],
        terminated: Dict[str, bool],
        info: Dict[str, Any],
    ) -> bool:
        no_observations = len(observations) == 0
        terminated_all = terminated.get("__all__", False)
        no_social_traffic = not info.get("social_traffic", True)
        return (no_observations or terminated_all) and no_social_traffic
