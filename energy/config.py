vehicle_params = {
    "mass": 1680,  # Vehicle mass [kg] (Tesla Model 3 Standard Range Plus)
    "g": 9.8066,  # Gravitational constant [m/s^2]
    "c_r": 0.009,  # Rolling resistance coefficient (estimated from tire data)
    "c1": 0.01,  # Rolling resistance speed coefficient (approximate)
    "c2": 0.1,  # Rolling resistance constant coefficient (approximate)
    "rho_air": 1.225,  # Air density [kg/m^3] (at sea level, 15°C)
    "A_f": 2.2,  # Frontal area [m^2] (Model 3 estimated)
    "C_d": 0.23,  # Drag coefficient (Tesla Model 3 official spec)
    "eta_driveline": 0.95,  # Driveline efficiency (electric vehicles are highly efficient)
    "eta_motor": 0.96,  # Motor efficiency (Tesla motors are very efficient)
    "alpha": 0.5,  # Regenerative braking efficiency parameter (estimated)
}
