from typing import List, Tuple, Any, Optional, Dict
import os
import sys
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, algorithm_identifier: str, enable_tensorboard: bool = True, evaluate: bool = False):
        self.scores_list: List[Tuple[float, str, int]] = []
        self.scores_per_scenario_list: List[Any] = []
        self.start_time = datetime.now()

        self.run_name = os.path.join(algorithm_identifier, self.start_time.strftime("%d%m%Y_%H%M%S"))
        self.training_stats_path = None if evaluate else f"training_stats/{self.run_name}"

        self.writer: Optional[SummaryWriter] = None
        if self.training_stats_path is not None and enable_tensorboard:
            tensorboard_path = os.path.join("training_stats", "tensorboard", self.run_name)
            os.makedirs(tensorboard_path, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_path)

    # --- Console Logging ---
    def log_progress(self, score: float, episode: int, ep_steps: int) -> None:
        elapsed = datetime.now() - self.start_time
        total_seconds = int(elapsed.total_seconds())
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)

        sys.stdout.write(
            f"\r Epi: {episode} | St.: {ep_steps} | Re.: {score:.2f} | Elapsed Time: {hours:02}:{minutes:02}:{seconds:02}"
        )
        sys.stdout.flush()

    def log_percentage(self, percentage: float) -> None:
        progress_bar = int(percentage * 20) * "-" + int((1 - percentage) * 20) * "_"
        sys.stdout.write(f"\r [{progress_bar}] {percentage * 100:.2f}%")
        sys.stdout.flush()

    # --- TensorBoard Logging ---
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        if self.writer:
            self.writer.add_histogram(tag, np.array(values), step)

    def close_writer(self) -> None:
        if self.writer:
            self.writer.close()

    # --- Public Training Hooks ---

    def after_train_step(self, reward: float, episode: int, step: int) -> None:
        """Called after each training step: log to console."""
        self.log_progress(score=reward, episode=episode, ep_steps=step)

    def after_episode_batch(self, episode: int, stats: Dict[str, float]) -> None:
        """Called after a batch of episodes: log summary stats to TensorBoard."""
        print(f"\nEpisode {episode} complete with stats: {stats}")
        for key, value in stats.items():
            self.log_scalar(tag=f"episode/{key}", value=value, step=episode)

    def after_evaluation(self, episode: int, scenario_rewards: List[float]) -> None:
        """Called after evaluation: log histogram of scenario scores."""
        print(f"\nEvaluation complete for episode {episode} with rewards: {np.mean(scenario_rewards)}")
        self.log_histogram(tag="Evaluation/Scenario_Rewards", values=scenario_rewards, step=episode)

    # --- Misc Utils ---
    def slice_list(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
