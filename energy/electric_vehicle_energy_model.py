import math
from energy.config import vehicle_params

class ElectricVehicleEnergyModel:
    def __init__(self):
        """
        Initialize the vehicle parameters.
        """
        self.mass = vehicle_params["mass"]  # Vehicle mass [kg]
        self.g = vehicle_params["g"]  # Gravitational constant [m/s^2]
        self.c_r = vehicle_params["c_r"]  # Rolling resistance coefficient
        self.c1 = vehicle_params["c1"]  # Rolling resistance speed coefficient
        self.c2 = vehicle_params["c2"]  # Rolling resistance constant coefficient
        self.rho_air = vehicle_params["rho_air"]  # Air density [kg/m^3]
        self.A_f = vehicle_params["A_f"]  # Frontal area [m^2]
        self.C_d = vehicle_params["C_d"]  # Drag coefficient
        self.eta_driveline = vehicle_params["eta_driveline"]  # Driveline efficiency
        self.eta_motor = vehicle_params["eta_motor"]  # Motor efficiency
        self.alpha = vehicle_params["alpha"]  # Regenerative braking efficiency parameter
        self.energy_consumption = 0.0  # Total energy consumption [J]
        self.total_distance = 0.0  # Total distance traveled [m]
        self.energy_history = []  # List to track energy consumption over time

    def clear_data(self):
        """
        Reset the energy consumption, distance, and history for a new simulation.
        """
        self.energy_consumption = 0.0
        self.total_distance = 0.0
        self.energy_history = []  # Reset energy history

    def compute_regenerative_braking_efficiency(self, acceleration):
        """
        Calculate the regenerative braking efficiency based on acceleration.
        """
        if acceleration < 0:
            return math.exp(-self.alpha * abs(acceleration))
        return 0.0

    def process_time_step(self, velocity, acceleration, time_step):
        """
        Process a single time step to compute power at the wheels and motor, and update total energy consumption.
        :param velocity: Current vehicle velocity [m/s]
        :param acceleration: Current vehicle acceleration [m/s^2]
        :param time_step: Duration of the time step [s]
        """
        # Calculate power at the wheels
        rolling_resistance = (self.mass * self.g * self.c_r / 1000) * (self.c1 * velocity + self.c2)
        aerodynamic_drag = 0.5 * self.rho_air * self.A_f * self.C_d * velocity**2
        slope_force = 0  # Assuming flat road; include mg * sin(theta) if grade is considered
        inertial_force = self.mass * acceleration

        power_wheels = (inertial_force + rolling_resistance + aerodynamic_drag + slope_force) * velocity

        # Calculate power at the motor
        if power_wheels > 0:  # Traction mode
            power_motor = power_wheels / (self.eta_driveline * self.eta_motor)
        else:  # Regenerative braking mode
            eta_rb = self.compute_regenerative_braking_efficiency(acceleration)
            power_motor = power_wheels * self.eta_driveline * self.eta_motor * eta_rb

        # Convert power to energy and update total consumption
        energy_step = power_motor * time_step / 3600  # Convert J to Wh
        self.energy_consumption += energy_step
        self.total_distance += velocity * time_step  # Update total distance traveled

        # Save energy consumption per time step
        self.energy_history.append(energy_step)

        return energy_step, self.total_distance

    def get_energy_consumption_per_km(self):
        """
        Compute and return energy consumption per kilometer in kWh/km.
        """
        if self.total_distance == 0:
            return 0.0  # Avoid division by zero
        return (self.energy_consumption / 1000)/ (self.total_distance / 1000)  # Convert m to km

    def get_energy_consumption_history(self):
        """
        Return the recorded energy consumption history.
        """
        return self.energy_history
