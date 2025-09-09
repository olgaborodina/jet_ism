import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_density, rho_to_numdensity, unit_time_in_megayr)


def get_mass(output_directory, onekpc=False):
    """
    Calculate evolution of total mass of star particles 
    Input: directory (string) that stores the output files, 
           onekpc (boolean) controls if we focus on the central 1kpc of the box, default is True
    Output: numpy array with columns: time in Myr, masses of the cells in cold phase, of warm phase, and hot phase in simulation units.
    """
    i_file = 0  # skip snap 0
    masses_stars = []
    while True:
        i_file += 1
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory + filename, "r")
        except:
            break
    
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        try: 
            x,y,z = snap_data['PartType4/Coordinates'][:].T
            masses = snap_data['PartType4/Masses'][:]
            center = snap_data['Header'].attrs['BoxSize'] / 2
            
            if onekpc == True:
                mask = ((x - center) > 500) & ((x - center) < 1500) & ((y - center) > 500) & ((y - center) < 1500) & ((z - center) > 500) & ((z - center) < 1500)
                masses_stars.append([time, sum(masses[mask])])
            elif onekpc == False:
                masses_stars.append([time, sum(masses)])
        except: ## no stars in PartType4
            masses_stars.append([time, 0])
    return np.array(masses_stars).T

def get_full_info(output_directory, onekpc=False):
    """
    Calculate evolution of mass, number of star particles as well as total star formation rate
    Input: directory (string) that stores the output files, 
           onekpc (boolean) controls if we focus on the central 1kpc of the box, default is True
    Output: numpy array with columns: time in Myr, total star formation rate in Msun/yr, number of star particles, total mass of star particles in Msun, total gas mass in Msun
    """
    i_file = 2  # skip snap 0
    stars = []
    while True: #and time < 20: #True or a number for a set length
        i_file += 1
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory + filename, "r+")
            x_star, y_star, z_star = snap_data['PartType4/Coordinates'][:].T
        except:
            try:
                i_file += 1
                filename = "snap_%03d.hdf5" % (i_file)
                snap_data = h5py.File(output_directory + filename, "r+")
                x_star, y_star, z_star = snap_data['PartType4/Coordinates'][:].T
            except:
                break
    
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        if onekpc == False:
            stars.append([time, sum(snap_data['PartType0/StarFormationRate'][:]), len(snap_data['PartType4/Masses'][:]), sum(snap_data['PartType4/Masses'][:]), sum(snap_data['PartType0/Masses'][:])])
        else:
            mask_gas = ((x > 500) & (x < 1500) & (y > 500) & (y < 1500) & (z > 500) & (z < 1500))
            mask_star = ((x_star > 500) & (x_star < 1500) & (y_star > 500) & (y_star < 1500) & (z_star > 500) & (z_star < 1500))
            stars.append([time, sum(snap_data['PartType0/StarFormationRate'][:][mask_gas]), len(snap_data['PartType4/Masses'][:][mask_star]), sum(snap_data['PartType4/Masses'][:][mask_star]), sum(snap_data['PartType0/Masses'][mask_gas])])
    stars = np.array(stars).T
    return stars