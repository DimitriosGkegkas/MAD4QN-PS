import streamlit as st
import numpy as np
import os
import time

file_path = "training_stats/DuelingDDQNAgents/16012025/avg_reward.npy"

st.title("Live Visualization of .npy File")

# Create a placeholder for the chart
chart_placeholder = st.empty()
chart_placeholder1 = st.empty()


while True:
    if os.path.exists(file_path):
        # Load data from the .npy file
        data = np.load("models/DuelingDDQNAgents/11022025/agent_straight_0_learning_curve.npy", allow_pickle=True)
        data2 = np.load("models/DuelingDDQNAgents9/11022025/agent_straight_0_learning_curve.npy", allow_pickle=True)
        # Update the line chart with the new data
        chart_placeholder.line_chart([[d["loss"],d["epsilon"]] for d in data if d is not None])

        # Update the line chart with the new data
        chart_placeholder1.line_chart([[d["loss"],d["epsilon"]] for d in data2 if d is not None])

    else:
        st.write("Waiting for file...")
    
    # Sleep for a while before refreshing
    time.sleep(10)