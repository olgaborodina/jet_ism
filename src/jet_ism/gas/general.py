import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_density, rho_to_numdensity, unit_time_in_megayr)


def get_temp(snapshot, gamma=GAMMA, approach='local'):
    """
    Calculate temperature for cells in file
    Input: file (filename), 
           gamma (adiabatic index), 
           approach (uniform or local, depending on how we want to estimate mean molecular weight mu)
    Output: temperature in Kelvin
    
    """
    X = 0.76
    n_electrons = snapshot['PartType0/ElectronAbundance'][:]
    utherm = snapshot['PartType0/InternalEnergy'][:]
    if approach == 'uniform':
        return (gamma - 1) * utherm  * PROTONMASS * mu / BOLTZMANN * unit_velocity * unit_velocity
    if approach == 'local':
        mu_local = 4 / (3 * X + 1 + 4 * X * n_electrons)
        return (gamma - 1) * utherm  * PROTONMASS * mu_local / BOLTZMANN * unit_velocity * unit_velocity

def get_soundspeed(snapshot, gamma=GAMMA):
    """
    Calculate speed of sound for all cells in file
    Input: file (filename), 
           gamma (adiabatic index), 
    Output: speed of sound in cm/s
    
    """
    return np.sqrt(gamma * (gamma - 1) * snapshot['PartType0/InternalEnergy'][:]) * unit_velocity

def get_kinetic_energy(snapshot):
    """
    Calculate kinetic energy for all cells in file
    Input: file (filename)
    Output: kinetic energy in simulation units
    
    """
    velocity_sq = (snapshot['PartType0/Velocities'][:][:,0] ** 2 +
               snapshot['PartType0/Velocities'][:][:,1] ** 2 +
               snapshot['PartType0/Velocities'][:][:,2] ** 2)
    return velocity_sq * snapshot['PartType0/Masses'][:] / 2

def get_thermal_energy(snapshot):
    """
    Calculate thermal energy for all cells in file assuming there is no other internal energy source
    Input: file (filename)
    Output: thermal energy in simulation units
    
    """    
    return snapshot['PartType0/InternalEnergy'][:] * snapshot['PartType0/Masses'][:]

def get_entropy(snapshot):
    """
    Calculate entropy for all cells in file
    Input: file (filename)
    Output: entropy in cgs
    
    """        
    temperatures = get_temp(snapshot, GAMMA)
    densities = snapshot['PartType0/Density'][:] * rho_to_numdensity
    return temperatures * densities ** (-2/3) * snapshot['PartType0/Masses'][:] * unit_mass

def get_mass_of_phases(output_directory, onekpc=False):
    """
    Calculate evolution of total mass of cells 
    Input: directory (string) that stores the output files, 
           onekpc (boolean) controls if we focus on the central 1kpc of the box, default is True
    Output: numpy array with columns: time in Myr, masses of the cells in cold phase, of warm phase, and hot phase in simulation units.
    
    """
    i_file = 0  # skip snap 0
    mass_phases = []
    while True:
        i_file += 1
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory + filename, "r")
        except:
            break
    
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        temperatures = get_temp(snap_data, GAMMA)
        x,y,z = snap_data['PartType0/Coordinates'][:].T
        masses = snap_data['PartType0/Masses'][:]
        center = snap_data['Header'].attrs['BoxSize'] / 2
        
        if onekpc == True:
            mask_hot = (temperatures > 10**4.4) & ((x - center) > 500) & ((x - center) < 1500) & ((y - center) > 500) & ((y - center) < 1500) & ((z - center) > 500) & ((z - center) < 1500)
            mask_warm = (temperatures < 10**4.4) & (temperatures > 10**3.7) & ((x - center) > 500) & ((x - center) < 1500) & ((y - center) > 500) & ((y - center) < 1500) & ((z - center) > 500) & ((z - center) < 1500)
            mask_cold = (temperatures < 10**3.7) & ((x - center) > 500) & ((x - center) < 1500) & ((y - center) > 500) & ((y - center) < 1500) & ((z - center) > 500) & ((z - center) < 1500)

        elif onekpc == False:
            mask_hot = (temperatures > 10**4.4)
            mask_warm = (temperatures < 10**4.4) & (temperatures > 10**3.7) 
            mask_cold = (temperatures < 10**3.7)
        
        mass_phases.append([time, masses[mask_cold].sum(), masses[mask_warm].sum(), masses[mask_hot].sum()])
    return np.array(mass_phases).T


def calculate_ram_pressure(snap_data):
    vx, vy, vz = snap_data['PartType0/Velocities'][:].T
    density = snap_data['PartType0/Density'][:]

    center = 0.5 * snap_data["Header"].attrs["BoxSize"]
    v_total_sq = vx ** 2 + vy ** 2 + vz ** 2

    ram_pressure = density * v_total_sq * unit_density * unit_velocity * unit_velocity
    return ram_pressure
    
    
def calculate_thermal_pressure(snap_data):
    density = snap_data['PartType0/Density'][:]
    temperature = get_temp(snap_data, GAMMA)
    number_density = density * rho_to_numdensity 
    thermal_pressure = temperature * number_density * BOLTZMANN

    return thermal_pressure
    