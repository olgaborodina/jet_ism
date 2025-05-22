from PIL import Image
from natsort import natsorted
import glob

import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity)

from .general import (get_temp)


def calculate_distribution(output_directory, snapshot_number, velmin=-900, velmax=900):
    bins = np.linspace(velmin, velmax, 91)
    bins_step = bins[1] - bins[0]
    
    velocities_hot = []
    velocities_warm = []
    velocities_cold = []
    

    filename = "snap_%03d.hdf5" % (snapshot_number)
    snap_data = h5py.File(output_directory + filename, "r")
    time = get_time_from_snap(snap_data) * unit_time_in_megayr

    x, y, z = snap_data['PartType0/Coordinates'][:].T
    vx, vy, vz = snap_data['PartType0/Velocities'][:].T
    center = snap_data['Header'].attrs['BoxSize'] / 2
    v_r = (vx * (x - center) + vy * (y - center) + vz * (z - center)) / \
                np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

    temperatures = get_temp(snap_data, GAMMA, 'local')
    masses = snap_data['PartType0/Masses'][:]
    densities = snap_data['PartType0/Density'][:]
    volumes = masses / densities
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)

    mask_hot  = ((temperatures > 10 ** 4.4) & (radius > 400) & (radius < 500))
    mask_warm = ((temperatures > 10 ** 3.7) & (temperatures < 10 ** 4.4) & (radius > 400) & (radius < 500))
    mask_cold = ((temperatures < 10 ** 3.7) & (radius > 400) & (radius < 500))

    velocities_hot = np.histogram((v_r[mask_hot]), bins=bins, weights=masses[mask_hot] / bins_step)[0])
    velocities_warm = np.histogram((v_r[mask_warm]), bins=bins, weights=masses[mask_warm] / bins_step)[0])
    velocities_cold = np.histogram((v_r[mask_cold]), bins=bins, weights=masses[mask_cold] / bins_step)[0])
    
    return bins, velocities_hot, velocities_warm, velocities_cold

def calculate_distribution_average(output_directory, t_start, velmin=-900, velmax=900):
    bins = np.linspace(velmin, velmax, 91)
    bins_step = bins[1] - bins[0]
    
    velocities_hot = []
    velocities_warm = []
    velocities_cold = []
    
    i_file = 11
    while True:
        i_file += 1
    
        try:
            filename = "snap_%03d.hdf5" % (i_file)
            snap_data = h5py.File(output_directory + filename, "r")
            time = get_time_from_snap(snap_data) * unit_time_in_megayr
            if time < t_start:
                continue
            if time > t_start + 1:
                break
        except:
            break
        
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        vx, vy, vz = snap_data['PartType0/Velocities'][:].T
        center = snap_data['Header'].attrs['BoxSize'] / 2
        v_r = (vx * (x - center) + vy * (y - center) + vz * (z - center)) / \
                    np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
    
        temperatures = get_temp(snap_data, GAMMA, 'local')
        masses = snap_data['PartType0/Masses'][:]
        densities = snap_data['PartType0/Density'][:]
        volumes = masses / densities
        radius = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
    
        mask_hot  = ((temperatures > 10 ** 4.4) & (radius > 400) & (radius < 500))
        mask_warm = ((temperatures > 10 ** 3.7) & (temperatures < 10 ** 4.4) & (radius > 400) & (radius < 500))
        mask_cold = ((temperatures < 10 ** 3.7) & (radius > 400) & (radius < 500))
    
        velocities_hot.append(np.histogram((v_r[mask_hot]), bins=bins, weights=masses[mask_hot] / bins_step)[0])
        velocities_warm.append(np.histogram((v_r[mask_warm]), bins=bins, weights=masses[mask_warm] / bins_step)[0])
        velocities_cold.append(np.histogram((v_r[mask_cold]), bins=bins, weights=masses[mask_cold] / bins_step)[0])
    
    velocities_average_hot = np.mean(velocities_hot, axis=0)
    velocities_average_warm = np.mean(velocities_warm, axis=0)
    velocities_average_cold = np.mean(velocities_cold, axis=0)
    
    return bins, velocities_average_hot, velocities_average_warm, velocities_average_cold