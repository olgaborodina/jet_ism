import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_density, rho_to_numdensity, unit_time_in_megayr)


def get_mass(output_directory, onekpc=False):
    """
    Calculate evolution of total mass of cells 
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
