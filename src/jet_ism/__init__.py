""" load libraries """
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path

__all__ = [
    'outflows', 'maps', 'gas', 'selfconsistent', 'propagation', 'styles'
    'get_time_from_snap',
    'megayear', 'h', 'unit_velocity', 'unit_length', 'unit_mass', 'unit_energy',
    'unit_density', 'unit_time', 'unit_time_in_megayr', 'mu',
    'PROTONMASS', 'BOLTZMANN', 'GRAVITY', 'LIGHTSPEED', 'rho_to_numdensity', 'GAMMA'
]

# Global constants
global megayear, h, unit_velocity, unit_length, unit_mass, unit_energy
global unit_density, unit_time, unit_time_in_megayr, mu
global PROTONMASS, BOLTZMANN, GRAVITY, LIGHTSPEED, rho_to_numdensity, GAMMA
global get_time_from_snap

# Physical constants and unit conversions
megayear = 3.15576e13
h = 1#float(0.67)
unit_velocity = np.float64(100000)
unit_length = np.float64(3.08568e+18 / h)
unit_mass = np.float64(1.989e+33 / h)
unit_energy = unit_mass * unit_velocity * unit_velocity
unit_density = unit_mass / unit_length / unit_length / unit_length 
unit_time = unit_length / unit_velocity
unit_time_in_megayr = unit_time / megayear

mu = np.float64(0.6165) # mean molecular weight 

PROTONMASS = np.float64(1.67262178e-24)
BOLTZMANN = np.float64(1.38065e-16)
GRAVITY = np.float64(6.6738e-8) # G in cgs
LIGHTSPEED = np.float64(3e10)  #cm/s
rho_to_numdensity = 1. * unit_density / (mu * PROTONMASS) # to cm^-3

GAMMA = 5 / 3

def get_time_from_snap(snap_data):
    """
    Find what time the snapshot has in its header.
    Input: snap_data (snapshot)
    Output: time in sim units
    
    """
    return float(snap_data["Header"].attrs["Time"])
