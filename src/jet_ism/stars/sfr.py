import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_density, rho_to_numdensity, unit_time_in_megayr, get_output_directory)


def get_sfr(output_directory, onekpc=False):
    """
    Calculate evolution of star formation rate from the snapshot files
    Input: directory (string) that stores the output files, 
           onekpc (boolean) controls if we focus on the central 1kpc of the box, default is True
    Output: numpy array with columns: time in Myr, total star formation rate in gas cells in simulation units
    
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

        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        if onekpc == False:
            stars.append([time, sum(snap_data['PartType0/StarFormationRate'][:])])
        else:
            mask_star = ((x_star > 500) & (x_star < 1500) & (y_star > 500) & (y_star < 1500) & (z_star > 500) & (z_star < 1500))
            stars.append([time, sum(snap_data['PartType0/StarFormationRate'][:][mask_gas])])
    stars = np.array(stars).T
    return stars

def sfr_output(output_directory):
    """
    Read the sfr.txt file that is output by the simulation
    Input: directory (string) that stores the output files, 
    Output: pandas dataframe with columns: time in Myr, total star formation rate in gas cells in simulation units, total star formation rate in Msun/yr, total mass of star particles in Msun, cumulative mass of star particles in Msun
    """
    sfr = pd.read_csv(output_directory + 'sfr.txt', sep='\s+', names=['time', 'total_sm', 'totsfrrate', 'rate_in_msunperyear', 'total_sum_mass_stars', 'cum_mass_stars'])
    return sfr