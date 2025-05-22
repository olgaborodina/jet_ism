import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path

from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, LIGHTSPEED, GRAVITY, get_time_from_snap, unit_time_in_megayr, unit_mass, unit_length, unit_time, unit_density)

from ..gas import general


def tng2cgs_energy(energy):
    return energy * unit_mass * unit_length ** 2 / unit_time **2

def cumenergy(storage, box, density, machnumber, jetpower, start=15, i_file=12):
    simulation_directory = str(f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{box}/turb_jet_d{density}_m{machnumber}/jet{jetpower}_{start}')
    output_directory = simulation_directory + "/output/"
    figures_directory = simulation_directory + "/output/figures/"
    
    threshold_tracer = 1e-3
    cumenergy_array = []
    time = 0
    
    while True:
        i_file += 1
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
    return np.array(cumenergy_array).T

def bondi_accretion(M_BH, densities, soundspeed):
    return  4 * np.pi * GRAVITY ** 2 * M_BH ** 2 * densities / soundspeed ** 3


def eddington_accretion(M_BH, eps_r=0.2):
    return  4 * np.pi * GRAVITY *  M_BH * PROTONMASS / eps_r / sigma_T / LIGHTSPEED  


def jetpower_from_cumenergy(time, cumenergy):
    return time[:-1], np.diff(cumenergy) / np.diff(time) / unit_time * unit_time_in_megayr


def jetpower_from_bondi(storage, box, density, machnumber, M_BH, epsilon, i_file=12):
    simulation_directory = str(f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{box}/turb_jet_d{density}_m{machnumber}/turb')
    output_directory = simulation_directory + "/output/"
    figures_directory = simulation_directory + "/output/figures/"
    
    threshold_tracer = 1e-3
    jetpower_array = []
    time = 0
    
    while True:
        i_file += 1
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
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        center = snap_data['Header'].attrs['BoxSize'] / 2
        distance_sq = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
        mask_center = (distance_sq  < 90 **2) & (distance_sq  > 30 **2)
        jetpower = 0.1 * epsilon * np.mean(bondi_accretion(M_BH, snap_data['PartType0/Density'][:], general.get_soundspeed(snap_data, GAMMA))[mask_center]) * LIGHTSPEED ** 2 * unit_density * unit_mass ** 2
        jetpower_array.append([time, jetpower])
    return np.array(jetpower_array).T