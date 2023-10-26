"""
Physical parameters for a quadrotor. Values parameterize the inertia, motor dynamics, 
rotor aerodynamics, parasitic drag, and rotor placement. 
Additional sources:
    https://bitcraze.io/2015/02/measuring-propeller-rpm-part-3
    https://wiki.bitcraze.io/misc:investigations:thrust
    https://commons.erau.edu/cgi/viewcontent.cgi?article=2057&context=publication
Notes:
    k_thrust is inferred from 14.5 g thrust at 2500 rad/s
    k_drag is mostly made up
"""
import numpy as np

d = 0.043

quad_params = {

    # Inertial properties
    'mass': 0.03,       # kg
    'Ixx':  1.43e-5,    # kg*m^2
    'Iyy':  1.43e-5,    # kg*m^2
    'Izz':  2.89e-5,    # kg*m^2
    'Ixy':  0.0,        # kg*m^2
    'Iyz':  0.0,        # kg*m^2 
    'Ixz':  0.0,        # kg*m^2

    # Geometric properties, all vectors are relative to the center of mass.
    'num_rotors': 4,                        # for looping over each actuator
    # 'rotor_pos': {  
    #                 'r1': np.array([0.046,  0,      0]),    # Location of Rotor 1, meters
    #                 'r2': np.array([0,      0.046,  0]),    # Location of Rotor 2, meters
    #                 'r3': np.array([-0.046, 0,      0]),    # Location of Rotor 3, meters
    #                 'r4': np.array([0,      -0.046, 0]),    # Location of Rotor 4, meters
    #              },
    'rotor_pos': {  
                    'r1': d*np.array([ 0.70710678118, 0.70710678118, 0]),    # Location of Rotor 1, meters
                    'r2': d*np.array([ 0.70710678118,-0.70710678118, 0]),    # Location of Rotor 2, meters
                    'r3': d*np.array([-0.70710678118,-0.70710678118, 0]),    # Location of Rotor 3, meters
                    'r4': d*np.array([-0.70710678118, 0.70710678118, 0]),    # Location of Rotor 4, meters
                 },

    'rotor_directions': np.array([1,-1,1,-1]),  # This dictates the direction of the torque for each motor. 

    'rI': np.array([0,0,0]), # location of the IMU sensor, meters

    # Frame aerodynamic properties
    'c_Dx': 0.5e-2,  # parasitic drag in body x axis, N/(m/s)**2
    'c_Dy': 0.5e-2,  # parasitic drag in body y axis, N/(m/s)**2
    'c_Dz': 1e-2,  # parasitic drag in body z axis, N/(m/s)**2

    # Rotor properties
    # See "System Identification of the Crazyflie 2.0 Nano Quadrocopter", Forster 2015.
    'k_eta': 2.3e-08,           # thrust coefficient N/(rad/s)**2
    'k_m':   7.8e-10,           # yaw moment coefficient Nm/(rad/s)**2
    'k_d':   10.2506e-07,       # rotor drag coefficient N/(rad*m/s**2) = kg/rad
    'k_z':   7.553e-07,         # induced inflow coefficient N/(rad*m/s**2) = kg/rad
    'k_flap': 0.0,              # Flapping moment coefficient Nm/(rad*m/s**2) = kg*m/rad

    # Motor properties
    'tau_m': 0.005,           # motor response time, seconds
    'rotor_speed_min': 0,       # rad/s
    'rotor_speed_max': 2500,    # rad/s
    'motor_noise_std': 0,     # rad/s

}