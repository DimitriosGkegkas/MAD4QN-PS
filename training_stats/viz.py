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
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Streamlit UI
st.title("Live Visualization of .npy File with Moving Average")

# Create placeholders for the charts
chart_placeholders = {key: st.empty() for key in file_paths}
chart_placeholders_avg = {key: st.empty() for key in file_paths}  # For moving averages

while True:
    for name, path in file_paths.items():
        if os.path.exists(path):
            # Load data
            data = np.load(path, allow_pickle=True)
            if data is not None and len(data) > 0:
                loss_values = [d["loss"] for d in data if d is not None]
                epsilon_values = [d["epsilon"] for d in data if d is not None]
                
                # Compute moving averages
                loss_avg = moving_average(loss_values)

                # Update original data chart
                chart_placeholders[name].line_chart({"Loss": loss_values, "Loss (Moving Avg)": loss_avg, "Epsilon": epsilon_values})
        else:
            st.write(f"Waiting for file: {path}")

    # Sleep before refreshing
    time.sleep(100)  # Reduce the wait time to make it more responsive
