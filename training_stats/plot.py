from genericpath import exists
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Set a consistent font size for all plot elements
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

# Define fixed colors for consistency across all plots using ColorBrewer Set1 palette
# Define fixed colors for consistency across all plots
reward_color       = "cornflowerblue"  # Reward curves
loss_color         = "teal"            # Loss curves
epsilon_color      = "goldenrod"       # Epsilon curves
crash_line_color   = "crimson"         # Horizontal line for crash reward
success_line_color = "forestgreen"     # Horizontal line for success reward


# Define the base directory
base_dir = "training_stats"

# Model names
models = [  "CheckT0", "CheckT" ]

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

    def parse_date(folder_name):
        try:
            return datetime.strptime(folder_name, "%d%m%Y")
        except ValueError:
            return None

    date_folders = [f for f in date_folders if parse_date(f) is not None]
    date_folders.sort(key=lambda x: parse_date(x), reverse=True)
    if not date_folders:
        return None

    latest_folder = date_folders[0]
    print(f"Latest folder for {model_name}: {latest_folder}")
    return os.path.join(model_path, latest_folder)

# Function to downsample data
def downsample_data(data, step_size=50):
    if len(data) < step_size:
        return data
    return [np.mean(data[i:i+step_size]) for i in range(0, len(data), step_size)]

# Collect data from each model
reward_data = {}
learning_curve_data = {}

for model in models:
    latest_folder = os.path.join(base_dir, model)
    if not os.path.exists(os.path.join(latest_folder, "avg_reward.npy")):
        latest_folder = get_latest_folder(model)
    if latest_folder:
        reward_path = os.path.join(latest_folder, "avg_reward.npy")
        learning_curve_path = os.path.join(latest_folder, "agent_straight_0_learning_curve.npy")

        if os.path.exists(reward_path):
            reward_data[model] = np.load(reward_path, allow_pickle=True)
        if os.path.exists(learning_curve_path):
            learning_curve_data[model] = np.load(learning_curve_path, allow_pickle=True)

# Create subplots for each model without using constrained_layout.
fig, axes = plt.subplots(len(models), 2, figsize=(15, 4 * len(models)))
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Add extra space between plots

for i, model in enumerate(models):
    if model in reward_data:
        reward_data[model] = reward_data[model][:10]
        rdata = [r[0] for r in reward_data[model]]
        steps = []
        for r in reward_data[model]:
            if len(r) > 2:
                steps.append(r[2] / 1000)
            else:
                steps.append(0)
        axes[i, 0].plot(steps, rdata, label="Reward", alpha=0.8)
        axes[i, 0].axhline(y=150, color=crash_line_color, linestyle='--', label="Crash Reward")
        axes[i, 0].axhline(y=300, color=success_line_color, linestyle='--', label="Success Reward")

        axes[i, 0].set_xlabel("Learning Steps (x1000)")
        axes[i, 0].set_ylabel("Reward")
        axes[i, 0].set_title(f"Reward Curve - {model}")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

    if model in learning_curve_data:
        learning_curve_data[model] = learning_curve_data[model][:100]
        loss_values = [d["loss"] for d in learning_curve_data[model] if d is not None]
        epsilon = [d["epsilon"] for d in learning_curve_data[model] if d is not None]
        steps = [d["learn_step_counter"] / 1000 for d in learning_curve_data[model] if d is not None]

        axes[i, 1].plot(steps, loss_values, label="Loss", alpha=0.8, )
        axes[i, 1].plot(steps, epsilon, label="Epsilon", alpha=0.8,)
        axes[i, 1].plot(downsample_data(steps), downsample_data(loss_values),
                        linestyle="--",)

        axes[i, 1].set_xlabel("Learning Steps (x1000)")
        axes[i, 1].set_ylabel("Loss")
        axes[i, 1].set_title(f"Loss & Epsilon Curve - {model}")
        axes[i, 1].legend()
        axes[i, 1].grid(True)

# Create subfolder for the current date (DDMMYYYY format)
current_date = datetime.now().strftime("%d%m%Y")
date_folder = os.path.join(base_dir, current_date)
os.makedirs(date_folder, exist_ok=True)

# Save the figure
file_path = os.path.join(date_folder, "learning_curve_training.png")
fig.savefig(file_path)
print(f"Saved figure to {file_path}")
