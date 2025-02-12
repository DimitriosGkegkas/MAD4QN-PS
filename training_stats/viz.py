import streamlit as st
import numpy as np
import os
import time

# Set the file paths
file_paths = {
    "DuelingDDQN_1": "models/DuelingDDQNAgents/11022025/agent_straight_0_learning_curve.npy",
    "DuelingDDQN_2": "models/DuelingDDQNAgents9/11022025/agent_straight_0_learning_curve.npy",
    "DropOutLayer": "models/DropOutLayer/11022025/agent_straight_0_learning_curve.npy"
}

# Define the moving average function
def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data  # Avoid computing if not enough data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Downsampling function (takes mean over step_size intervals)
def downsample_data(data, step_size=1000):
    if len(data) < step_size:
        return data  # Avoid downsampling if not enough data
    return [np.mean(data[i:i+step_size]) for i in range(0, len(data), step_size)]

# Streamlit UI
st.title("Live Visualization of .npy File with Moving Average and Downsampling")

# Create placeholders for the charts
chart_placeholders = {key: st.empty() for key in file_paths}

while True:
    for name, path in file_paths.items():
        if os.path.exists(path):
            # Load data
            data = np.load(path, allow_pickle=True)
            if data is not None and len(data) > 0:
                loss_values = [d["loss"] for d in data if d is not None]
                epsilon_values = [d["epsilon"] for d in data if d is not None]
            

                # Apply downsampling (every 1000 steps)
                loss_values_down = downsample_data(loss_values, step_size=1000)
                epsilon_values_down = downsample_data(epsilon_values, step_size=1000)

                # Update original data chart with downsampled values
                chart_placeholders[name].line_chart({
                    "Loss (Downsampled)": loss_values_down,
                    "Epsilon (Downsampled)": epsilon_values_down
                })
        else:
            st.write(f"Waiting for file: {path}")

    # Sleep before refreshing
    time.sleep(1000)  # Faster updates
