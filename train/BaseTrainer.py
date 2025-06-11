from typing import List, Tuple, Any, Optional
import os
import sys
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, algorithm_identifier: str,   enable_tensorboard: bool = True, evaluate: bool = False):
        self.scores_list: List[Tuple[float, str, int]] = []
        self.scores_per_scenario_list: List[Any] = []
        self.start_time = datetime.now()
        
        self.run_name = f"{algorithm_identifier}_{self.start_time.strftime("%d%m%Y")}"
        self.training_stats_path = None if evaluate else f"training_stats/{self.run_name}"

        self.writer: Optional[SummaryWriter] = None
        if enable_tensorboard:
            tensorboard_path = os.path.join(self.training_stats_path, "tensorboard")
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

    # --- File-Based Score Saving ---
    def save_scores(
        self,
        avg_rewards: List[Tuple[float, str, int]],
        avg_per_scenario: List[Any]
    ) -> None:
        np.save(os.path.join(self.training_stats_path, "avg_reward.npy"), np.array(avg_rewards, dtype=object))
        np.save(os.path.join(self.training_stats_path, "avg_reward_per_scenario.npy"), np.array(avg_per_scenario, dtype=object))

    def load_scores(self) -> Tuple[List[Tuple[float, str, int]], List[Any]]:
        try:
            rewards = np.load(os.path.join(self.training_stats_path, "avg_reward.npy"), allow_pickle=True).tolist()
            per_scenario = np.load(os.path.join(self.training_stats_path, "avg_reward_per_scenario.npy"), allow_pickle=True).tolist()
            return rewards, per_scenario
        except Exception:
            return [], []

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
    
    

    # --- Misc Utils ---
    def slice_list(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
