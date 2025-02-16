import numpy as np
import os
import time
import subprocess
import matplotlib.pyplot as plt

# Remote server details
REMOTE_USER = "gkegkas"
REMOTE_HOST = "dgx-01.tail4ddd5c.ts.net"

# Remote directories containing both sets of files
REMOTE_LEARNING_CURVE_DIRS = {
    "DuelingDDQN_1": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents9/12022025/",
    "DuelingDDQN_2": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents91/12022025/",
    "DuelingDDQN_3": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents/12022025/",
    "DuelingDDQN_4": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents1/12022025/",
    "DuelingDDQN_5": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents92/12022025/",
    "DuelingDDQN_6": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents93/12022025/",
}

REMOTE_REWARD_DIRS = {
    "DuelingDDQN_1": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents9/12022025/",
    "DuelingDDQN_2": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents91/12022025/",
    "DuelingDDQN_3": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents/12022025/",
    "DuelingDDQN_4": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents1/12022025/",
    "DuelingDDQN_5": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents92/12022025/",
    "DuelingDDQN_6": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents93/12022025/",
}

# The specific file names
LEARNING_CURVE_FILE = "agent_straight_0_learning_curve.npy"
REWARD_FILE = "avg_reward.npy"

# Local directory to store fetched files
LOCAL_PATH = "remote_models/"
os.makedirs(LOCAL_PATH, exist_ok=True)

# Create a figure for learning curves
fig1, axs1 = plt.subplots(3, 2, figsize=(12, 8))  # 2x2 grid for 4 models
fig1.suptitle("Downsampled Learning Curve Comparison")

# Create a figure for avg rewards
fig2, axs2 = plt.subplots(3, 2, figsize=(12, 8))  # 2x2 grid for 4 models
fig2.suptitle("Average Reward Comparison")

for i, (name, _) in enumerate(REMOTE_LEARNING_CURVE_DIRS.items()):
    # Load learning curve data
    learning_file_path = os.path.join(LOCAL_PATH, f"{name}_learning.npy")
    if os.path.exists(learning_file_path):
        try:
        
            data = np.load(learning_file_path, allow_pickle=True)
            if data is not None and len(data) > 0:
                loss_values = [d["loss"] for d in data if d is not None]
                epsilon_values = [d["epsilon"] for d in data if d is not None]

                # Apply downsampling (every 50 steps)
                loss_downsampled = loss_values
                epsilon_downsampled = epsilon_values

                # Select subplot for learning curve
                ax1 = axs1[i // 2, i % 2]
                ax1.plot(loss_downsampled, label="Loss (Downsampled)", alpha=0.5)
                ax1.plot(epsilon_downsampled, label="Epsilon (Downsampled)", alpha=0.5)

                ax1.set_title(f"{name} (Downsampled Learning Curve)")
                ax1.set_xlabel("Timesteps (Downsampled)")
                ax1.set_ylabel("Value")
                ax1.legend()
        except Exception as e:
            print(e)
            print("Error in loading data")

    # Load reward data
    reward_file_path = os.path.join(LOCAL_PATH, f"{name}_reward.npy")
    if os.path.exists(reward_file_path):
        try:
            reward_data = np.load(reward_file_path, allow_pickle=True)
            if reward_data is not None and len(reward_data) > 0:
                # pad reward_data to have the same length 200
                reward_data = np.pad(reward_data, ((0, 200 - len(reward_data)), (0, 0)), mode="constant")
                # Select subplot for rewards
                ax2 = axs2[i // 2, i % 2]
                ax2.plot([reward[0] for reward in reward_data], label="Avg Reward", color="tab:orange")
                ax2.set_title(f"{name} (Average Reward)")
                ax2.set_xlabel("Timesteps")
                ax2.set_ylabel("Reward")
                ax2.legend()
        except Exception as e:
            print(e)
            print("Error in loading data")

# Display the plots
plt.show()

# Update every 30 seconds