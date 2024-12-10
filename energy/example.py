from electric_vehicle_energy_model import ElectricVehicleEnergyModel
import numpy as np
from matplotlib import pyplot as plt

# Initialize the model with constant parameters
vehicle_model = ElectricVehicleEnergyModel(
    mass=1500,  # kg
    g=9.8066,  # m/s^2
    c_r=0.015,  # rolling resistance coefficient
    c1=0.01,  # speed coefficient for rolling resistance
    c2=0.1,  # constant coefficient for rolling resistance
    rho_air=1.225,  # air density [kg/m^3]
    A_f=2.2,  # frontal area [m^2]
    C_d=0.28,  # drag coefficient
    eta_driveline=0.92,  # driveline efficiency
    eta_motor=0.91,  # motor efficiency
    alpha=0.9 # regenerative braking efficiency parameter
)


# Experiment 1: Accelerate to a constant speed and travel 1 km
velocities_exp1 = [0]  # Initial velocity
energies_exp1 = []
accelerations_exp1 = []

distance_exp1 = 0
while distance_exp1 < 1000:  # Target distance of 1 km
    if velocities_exp1[-1] < 15:  # Accelerate to 15 m/s
        acceleration = 1  # m/s^2
    else:
        acceleration = 0  # Constant speed

    energy, distance = vehicle_model.process_time_step(
        velocity=velocities_exp1[-1], acceleration=acceleration, time_step=1
    )
    velocities_exp1.append(velocities_exp1[-1] + acceleration)
    energies_exp1.append(energy)
    accelerations_exp1.append(acceleration)
    distance_exp1 += velocities_exp1[-1] * 1  # Update distance

energy_consumption_exp1 = vehicle_model.get_energy_consumption_per_km()
vehicle_model.clear_data()

# Experiment 2: Accelerate and decelerate 3-4 times to reach 1 km
velocities_exp2 = [0]
energies_exp2 = []
accelerations_exp2 = []

distance_exp2 = 0
step = 0
while distance_exp2 < 1000:  # Target distance of 1 km
    if step % 20 < 10:  # Alternate between acceleration and deceleration
        acceleration = 2 if velocities_exp2[-1] < 20 else -2
    else:
        acceleration = 0  # Constant speed between cycles

    energy, distance = vehicle_model.process_time_step(
        velocity=velocities_exp2[-1], acceleration=acceleration, time_step=1
    )
    velocities_exp2.append(velocities_exp2[-1] + acceleration)
    energies_exp2.append(energy)
    accelerations_exp2.append(acceleration)
    distance_exp2 += velocities_exp2[-1] * 1  # Update distance
    step += 1

energy_consumption_exp2 = vehicle_model.get_energy_consumption_per_km()
vehicle_model.clear_data()

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Experiment 1 Plot
axs[0].plot(energies_exp1, label='Energy Consumption (Wh)')
axs[0].plot(velocities_exp1, label='Velocity (m/s)')
axs[0].plot(accelerations_exp1, label='Acceleration (m/s²)')
axs[0].plot(np.cumsum(energies_exp1), label='Cumulative Energy Consumption (Wh)')
axs[0].set_title('Experiment 1: Accelerate to Constant Speed')
axs[0].legend()
axs[0].grid()

# Experiment 2 Plot
axs[1].plot(energies_exp2, label='Energy Consumption (Wh)')
axs[1].plot(velocities_exp2, label='Velocity (m/s)')
axs[1].plot(accelerations_exp2, label='Acceleration (m/s²)')
axs[1].plot(np.cumsum(energies_exp2), label='Cumulative Energy Consumption (Wh)')
axs[1].set_title('Experiment 2: Accelerate and Decelerate Multiple Times')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

# Print Results
print("Energy Consumption Experiment 1 (Wh/km):", energy_consumption_exp1)
print("Energy Consumption Experiment 2 (Wh/km):", energy_consumption_exp2)
