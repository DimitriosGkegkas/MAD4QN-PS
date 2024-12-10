import math

class ElectricVehicleEnergyModel:
    def __init__(self, mass, g, c_r, c1, c2, rho_air, A_f, C_d, eta_driveline, eta_motor, alpha):
        """
        Initialize the vehicle parameters.
        """
        self.mass = mass  # Vehicle mass [kg]
        self.g = g  # Gravitational constant [m/s^2]
        self.c_r = c_r  # Rolling resistance coefficient
        self.c1 = c1  # Rolling resistance speed coefficient
        self.c2 = c2  # Rolling resistance constant coefficient
        self.rho_air = rho_air  # Air density [kg/m^3]
        self.A_f = A_f  # Frontal area [m^2]
        self.C_d = C_d  # Drag coefficient
        self.eta_driveline = eta_driveline  # Driveline efficiency
        self.eta_motor = eta_motor  # Motor efficiency
        self.alpha = alpha  # Regenerative braking efficiency parameter
        self.energy_consumption = 0.0  # Total energy consumption [J]
        self.total_distance = 0.0  # Total distance traveled [m]
    
    def clear_data(self):
        """
        Reset the energy consumption and distance for a new simulation.
        """
        self.energy_consumption = 0.0
        self.total_distance = 0.0
    
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

        
        
        # Update energy consumption
        self.energy_consumption += power_motor * time_step  # Integrate over time
        self.total_distance += velocity * time_step  # Update total distance traveled
        return power_motor / 3600, self.total_distance
    
    def get_energy_consumption_per_km(self):
        """
        Compute and return energy consumption per kilometer in Wh/km.
        """
        if self.total_distance == 0:
            return 0.0  # Avoid division by zero
        energy_in_wh = self.energy_consumption / 3600  # Convert J to Wh
        return energy_in_wh / (self.total_distance / 1000)  # Convert m to km
