import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import os


# Define the base directory
base_dir = "training_stats"

# Model names
# models = [ "decay-preload", "decay-v1" , "decay-v2", "DuelingDDQNAgents92", "DuelingDDQNAgents-v1" ]
models = [ "DropOutLayer2-v1", "DropOutLayer2" ]


# Function to get the latest date folder
def get_latest_folder(model_name):
    model_path = os.path.join(base_dir, model_name)
    if not os.path.exists(model_path):
        return None
    
    # Get all folder names that look like a date
    date_folders = [
        f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))
    ]

    if not date_folders:
        return None

    # Convert folder names from DDMMYYYY to actual datetime objects
    def parse_date(folder_name):
        try:
            return datetime.strptime(folder_name, "%d%m%Y")
        except ValueError:
            return None  # Ignore folders that are not in DDMMYYYY format

    date_folders = [f for f in date_folders if parse_date(f) is not None]  # Filter valid dates
    date_folders.sort(key=lambda x: parse_date(x), reverse=True)  # Sort by date (latest first)

    if not date_folders:
        return None

    latest_folder = date_folders[0]  # Pick the latest date folder
    print(f"Latest folder for {model_name}: {latest_folder}")
    return os.path.join(model_path, latest_folder)


# Function to downsample data
def downsample_data(data, step_size=1000):
    if len(data) < step_size:
        return data  # Avoid downsampling if not enough data
    return [np.mean(data[i:i+step_size]) for i in range(0, len(data), step_size)]

# Collect data from each model
reward_data = {}
learning_curve_data = {}

for model in models:
    latest_folder = get_latest_folder(model)
    if latest_folder:
        reward_path = os.path.join(latest_folder, "avg_reward.npy")
        learning_curve_path = os.path.join(latest_folder, "agent_straight_0_learning_curve.npy")

        if os.path.exists(reward_path):
            reward_data[model] = np.load(reward_path, allow_pickle=True)
        if os.path.exists(learning_curve_path):
            learning_curve_data[model] = np.load(learning_curve_path, allow_pickle=True)

# Create subplots for each model
fig, axes = plt.subplots(len(models), 2, figsize=(14, 5 * len(models)), constrained_layout=True)


for i, model in enumerate(models):
    if model in reward_data:
        rdata = [r[0] for r in reward_data[model]]
        axes[i, 0].plot(rdata, label=f"{model} - Reward", alpha=0.6)

        # Add horizontal lines for min and max rewards
        axes[i, 0].axhline(y=150, color='r', linestyle='--', label="Crash Reward (150)")
        axes[i, 0].axhline(y=300, color='g', linestyle='--', label="Success Reward (300)")

        axes[i, 0].set_xlabel("Episodes")
        axes[i, 0].set_ylabel("Reward")
        axes[i, 0].set_title(f"Reward Curve - {model}")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

    if model in learning_curve_data:
        # Extract loss, epsilon, and steps
        loss_values = [d["loss"] for d in learning_curve_data[model] if d is not None]
        epsilon = [d["epsilon"] for d in learning_curve_data[model] if d is not None]
        steps = [d["learn_step_counter"] for d in learning_curve_data[model] if d is not None]

        # Plot loss and epsilon
        axes[i, 1].plot(steps, loss_values, label=f"{model} - Loss", alpha=0.6)
        axes[i, 1].plot(steps, epsilon, label=f"{model} - Epsilon", alpha=0.6)
        axes[i, 1].plot(downsample_data(steps), downsample_data(loss_values), label=f"{model} - Downsampled Loss", linestyle="--")

        axes[i, 1].set_xlabel("Episodes")
        axes[i, 1].set_ylabel("Loss")
        axes[i, 1].set_title(f"Loss & Epsilon Curve - {model}")
        axes[i, 1].legend()
        axes[i, 1].grid(True)

# Adjust layout and show the plot
# plt.tight_layout()
# fig.subplots_adjust(hspace=0.4)
plt.show()
