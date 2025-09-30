import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
import random

from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, LIGHTSPEED, GRAVITY, sigma_T, get_time_from_snap, unit_time_in_megayr, unit_mass, unit_length, unit_time, unit_density, get_output_directory)

from ..gas import general


def tng2cgs_energy(energy):
    """
    Convert simulation units of energy to cgs
    Input: energy in simulation units (number or array)
    Output: energy in cgs (same format as input)
    """
    return energy * unit_mass * unit_length ** 2 / unit_time **2

def cgs2solarperyear_accretion(accretion_rate):
    """
    Convert accretion rate in cgs units to solar mass per year
    Input: accretion rate in cgs (number or array)
    Output: accretion rate in M_solar / year (same format as input)
    """
    return accretion_rate / unit_mass * 3600 * 24 * 365

def cumenergy(output_directory, i_file=13):
    """
    Read cumulative energies from the files in directory
    Input: output_directory (directory where snap_datas are stored),
           i_file(initial file to start from, default is 13)
    Output: cumulative energy (array with 2 columns, time in Myr and energy in cgs units)
    """    
    figures_directory = output_directory + "/figures/"
    
    threshold_tracer = 1e-3
    cumenergy_array = []
    time = 0
    
    while True:
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory + filename, "r+")
        except:
            try:
                i_file += 1
                filename = "snap_%03d.hdf5" % (i_file)
                snap_data = h5py.File(output_directory + filename, "r+")
            except:
                break
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        cumenergy_array.append([time, tng2cgs_energy(snap_data['PartType5/BH_CumEgyInjection_RM'][:][0])])
        i_file += 1
    return np.array(cumenergy_array).T

def bondi_accretion(M_BH, density, soundspeed, alpha):
    """
    Calculate accretion rate in g/s from the Bondi formula
    Input: M_BH (black hole mass in solar masses)
           density (density of surrounding gas at Bondi radius in simulation units)
           soundspeed (sound speed in cgs units)
           alpha (accretion fraction, unitless)
    Output: Bondi accretion rate in cgs
    """
    return  4 * alpha * np.pi * GRAVITY ** 2 * M_BH ** 2 * density / soundspeed ** 3 * unit_density * unit_mass ** 2


def eddington_accretion(M_BH, eps_r=0.2):
    """
    Calculate Eddington accretion rate in g/s
    Input: M_BH (black hole mass in solar masses)
           eps_r (accretion fraction, unitless, default is 0.2)
    Output: Eddington accretion rate in g/s
    """    
    return  4 * np.pi * GRAVITY *  M_BH * PROTONMASS / eps_r / sigma_T / LIGHTSPEED  * unit_mass


def jetpower_from_cumenergy(time, cumenergy_from_snapshot):
    """
    Calculate jet power in erg/s from the cumulative energy injected
    Input: time (time corrsponding to the cumulative energy measurement, in Myr)
           cumenergy (cumulative energy in erg)
    Output: time, corresponding to the jet power measurements
            jet power in erg/s
    """    
    return time[:-1], np.diff(cumenergy_from_snapshot) / np.diff(time) / unit_time * unit_time_in_megayr

def jetpower_from_snapshot(output_directory, i_file=13):
    """
    Read cumulative energies from the files in directory
    Input: output_directory (directory where snap_datas are stored),
           i_file(initial file to start from, default is 13)
    Output: time, corresponding to the jet power measurements
            jet power in erg/s
    """    
    time, cumenergy_from_snapshot = cumenergy(output_directory, i_file)
    return time[:-1], np.diff(cumenergy_from_snapshot) / np.diff(time) / unit_time * unit_time_in_megayr

def jetpower_from_bondi_formula(M_BH, density, soundspeed, alpha):
    """
    Calculate jet power in erg/s from the Bondi formula, assuming a factor 0.1 of the accretion rate is converted to jet power
    Input: M_BH (black hole mass in solar masses)
           density (density of surrounding gas at Bondi radius in simulation units)
           soundspeed (sound speed in cgs units)
           alpha (accretion fraction, unitless)
    Output: jet power in erg/s
    """ 
    return 0.1 * bondi_accretion(M_BH, density, soundspeed, alpha) * LIGHTSPEED ** 2

def jetpower_from_accretion(M_dot, factor=0.1):
    """
    Calculate jet power in erg/s from the accretion rate, assuming a factor 0.1 of the accretion rate is converted to jet power
    Input: M_dot (accretion rate in g/s)
           factor (factor of the accretion rate that is converted to jet power, unitless, default is 0.1)
    Output: jet power in erg/s
    """
    return factor * M_dot * LIGHTSPEED ** 2
    
def calculate_state_around_black_hole(snap_data, black_hole_position=None, h=None, gamma=GAMMA):
    """
    Calculate density and sound speed for a black hole using Rainer's kernel approach
    Input: snap_data (snapshot data)
           black_hole_position (position of the black hole in simulation units, default is the center of the box, otherwise
           has to be a list of 3 elements, x,y,z coordinates)
           h (smoothing length in simulation units, default is the smoothing length of the black hole, otherwise specify a number in pc)
           gamma (adiabatic index, unitless, default is 5/3)
    Output: density (in simulation units)
            sound speed (in cgs units)
    """

    if black_hole_position == None:
        black_hole_position = snap_data['PartType5/Coordinates'][:].T
    if h == None:    
        h = snap_data['PartType5/BH_Hsml'][:] 
    
    # Extract kernel coefficients
    KERNEL_COEFF_1 = 2.546479089470
    KERNEL_COEFF_2 = 15.278874536822
    KERNEL_COEFF_5 = 5.092958178941
    
    # Precompute h factors
    hinv = 1.0 / h
    hinv3 = hinv**3
    
    part0 = snap_data['PartType0']
    masses = part0['Masses'][:]
    densities = part0['Density'][:]
    therm_energy = part0['InternalEnergy'][:]
    ids = part0['ParticleIDs'][:]
    coordinates = part0['Coordinates'][:] 
    
    volumes = masses / densities
    dx = coordinates[:, 0] - black_hole_position[0]
    dy = coordinates[:, 1] - black_hole_position[1]
    dz = coordinates[:, 2] - black_hole_position[2]
    
    mask = (
        (np.abs(dx) < h) &
        (np.abs(dy) < h) &
        (np.abs(dz) < h) & (masses > 0) & (ids > 0)
    )
    
    # Apply mask *before* expensive calculations
    dx, dy, dz = dx[mask], dy[mask], dz[mask]
    masses = masses[mask]
    densities = densities[mask]
    therm_energy = therm_energy[mask]
    ids = ids[mask]
    coordinates = coordinates[mask]
    
    # Now compute volumes and distances
    volumes = masses / densities
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    u = distances * hinv
    
    mask_distances = ((distances < h) & (distances > h * 0.33))
    full_mask1 = mask_distances & (u < 0.5)
    full_mask2 = mask_distances & (u >= 0.5) 
    
    wk = np.zeros_like(distances)
    wk[full_mask1] = hinv3 * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * (u[full_mask1] - 1.0) * u[full_mask1] ** 2)
    wk[full_mask2] = hinv3 * KERNEL_COEFF_5 * (1.0 - u[full_mask2]) ** 3
    
    smoothU = masses * wk * therm_energy 
    BH_densities = masses * wk
    volume_weight = volumes * wk

    return sum(BH_densities) / sum(volume_weight), np.sqrt(gamma * (gamma - 1) * sum(smoothU) / sum(BH_densities)) * unit_velocity

def accretion_factor(time, final_eps):
    """
    Calculate the accretion factor as a function of time and the final accretion factor, assuming a sigmoid function
    Input: time (time array or number)
           final_eps (final accretion factor, unitless)
    Output: accretion factor, unitless (array or number corresponding to the time input)
    """
    eps = 1 * (1 / (1 + np.exp(- 20 * (time-0.5)))) * final_eps 
    return eps

def sample_BH(snap_data, N=10, h = 90.0, result='all', seed=42):
    """
    Sample N black holes from the snap_data and calculate the density and sound speed for each black hole
    Input: snap_data (snapshot data)
           N (number of black holes to sample, default is 10)
           h (smoothing length in simulation units, default is 90.0 pc)
           result (result to return, default is 'all', otherwise 'mean')
           seed (seed for the random number generator, default is 42)
    Output: densities (array of densities in simulation units)
            csnds (array of sound speeds in cgs units)
    """
    random.seed(seed)
    x_coords = np.random.uniform(0, snap_data['Header'].attrs['BoxSize'], N)
    y_coords = np.random.uniform(0, snap_data['Header'].attrs['BoxSize'], N)
    z_coords = np.random.uniform(0, snap_data['Header'].attrs['BoxSize'], N)
    
    densities = []
    csnds = []
    for i in range(len(x_coords)):
        density, csnd = calculate_state_around_black_hole(snap_data, black_hole_position=[x_coords[i], y_coords[i], z_coords[i]], h=h)
        densities.append(density)
        csnds.append(csnd)
    if result=='all':
        return np.array([densities, csnds])
    elif result=='mean':    
        return np.mean(densities), np.mean(csnds)
    else:
        raise ValueError("Invalid result option, must be 'all' or 'mean'")