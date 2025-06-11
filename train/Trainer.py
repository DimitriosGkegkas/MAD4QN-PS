from typing import List, Dict, Tuple, Optional
import numpy as np

class Trainer:
    def __init__(
        self,
        trainer_logger,  # BaseTrainer instance
        agent_manager,   # AgentManager instance
        episode_manager,  # ScenarioManager instance
        env_manager,     # EnvironmentManager instance
        config
    ):
        self.logger = trainer_logger
        self.agent_manager = agent_manager
        self.episode_manager = episode_manager
        self.env_manager = env_manager
        
        self.total_steps = config.total_steps
        self.agent_count = config.agent_count
        self.algorithm_identifier = config.algorithm_identifier
        self.evaluation_step = config.evaluation_step
        
        self.n_steps = 0
        self.n_episodes = 0
        
    def train(self) -> None:
        while self.n_steps < self.total_steps:
            self._episode()

    def _episode_train(self) -> None:
        direction, communication, current_state, current_messages, current_raw_messages, terminate, truncated, reward, infos = self.env_manager.reset()
        ep_steps = 0
        scores = [0.0 for _ in current_state]
        
        self.agent_manager.set_communication(communication)
        self.agent_manager.set_direction(direction)

        while not self.episode_manager.is_batch_episode_done(current_state, reward, terminate, infos) and ep_steps < 1000:
            action, next_messages, next_raw_messages  = self.agent_manager.select_batch_actions(current_state, current_messages, terminate, truncated)
            
            next_state, terminate, truncated, reward, infos = self.env_manager.step(action)

            self.agent_manager.store_transitions(current_state, current_raw_messages, action, reward, next_state, next_raw_messages, terminate, truncated)
            self.agent_manager.update_agents(step=ep_steps, logger=self.logger)
            
            current_state = next_state
            current_messages = next_messages
            current_raw_messages = next_raw_messages
            self.n_steps += 1
            ep_steps += 1
            
            scores = [sum(r.values()) + s for r, s in zip(reward, scores)]
            self.logger.log_progress(np.mean(scores), self.n_episodes, ep_steps)

        # Log and evaluate
        self.logger.log_scalar("reward/train", np.mean(scores), self.n_episodes)
        self.n_episodes += 1

        if self.evaluator.should_evaluate(self.n_episodes):
            mean_score, rewards_all = self.evaluator.evaluate(self.n_episodes, self.n_steps)
            self.env_manager.env.modify_probs(rewards_all)
