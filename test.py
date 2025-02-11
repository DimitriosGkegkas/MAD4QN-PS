import numpy as np

# Define the file path
file_path = 'training_stats/DuelingDDQNAgents-v1/09012025/avg_reward.npy'

# Load the .npy file
try:
    data = np.load(file_path, allow_pickle=True)  # allow_pickle=True in case the file contains objects
    print("Data loaded successfully.")
    scenario = data[0]
except Exception as e:
    print(f"Error loading the file: {e}")
