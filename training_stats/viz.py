import streamlit as st
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
    "DuelingDDQN_1": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents92/14022025/",
    "DuelingDDQN_2": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents-v1/15022025/",
    # "DuelingDDQN_3": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents/12022025/",
    # "DuelingDDQN_4": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents1/12022025/",
    # "DuelingDDQN_5": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents92/12022025/",
    # "DuelingDDQN_6": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents93/12022025/",
    # "DuelingDDQN_7": "/home/gkegkas/Research/MAD4QN-PS/models/DuelingDDQNAgents94/12022025/",
}

REMOTE_REWARD_DIRS = {
    "DuelingDDQN_1": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents92/14022025/",
    "DuelingDDQN_2": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents-v1/15022025/",
    # "DuelingDDQN_3": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents/12022025/",
    # "DuelingDDQN_4": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents1/12022025/",
    # "DuelingDDQN_5": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents92/12022025/",
    # "DuelingDDQN_6": "/home/gkegkas/Research/MAD4QN-PS/training_stats/DuelingDDQNAgents93/12022025/",
}

# The specific file names
LEARNING_CURVE_FILE = "agent_right_0_learning_curve.npy"
REWARD_FILE = "avg_reward.npy"

# Local directory to store fetched files
LOCAL_PATH = "remote_models/"
os.makedirs(LOCAL_PATH, exist_ok=True)

# Downsampling function (averages over `step_size` intervals)
def downsample_data(data, step_size=50):
    if len(data) < step_size:
        return data  # Avoid downsampling if not enough data
    return [np.mean(data[i:i + step_size]) for i in range(0, len(data), step_size)]

# Function to sync remote files (Parallel `scp` execution for speed)
def sync_files():
    processes = []
    
    # Sync learning curve files
    for name, remote_dir in REMOTE_LEARNING_CURVE_DIRS.items():
        remote_file = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_dir}{LEARNING_CURVE_FILE}"
        local_file = os.path.join(LOCAL_PATH, f"{name}_learning.npy")
        proc = subprocess.Popen(["scp", remote_file, local_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(proc)
    
    # Sync reward files
    for name, remote_dir in REMOTE_REWARD_DIRS.items():
        remote_file = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_dir}{REWARD_FILE}"
        local_file = os.path.join(LOCAL_PATH, f"{name}_reward.npy")
        proc = subprocess.Popen(["scp", remote_file, local_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(proc)
    
    # Wait for all file transfers to complete
    for proc in processes:
        proc.communicate()

# Streamlit UI
st.title("Live Visualization of Downsampled Learning Curves & Rewards")

# Create placeholders for the plots
learning_curve_placeholder = st.empty()
reward_placeholder = st.empty()

while True:
    sync_files()  # Fetch latest data from remote server
    
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
                    loss_downsampled = downsample_data(loss_values, step_size=50)
                    epsilon_downsampled = downsample_data(epsilon_values, step_size=50)

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

    # Replace the old plots with the new ones
    learning_curve_placeholder.pyplot(fig1, clear_figure=True)
    reward_placeholder.pyplot(fig2, clear_figure=True)

    time.sleep(60)  # Update every 30 seconds