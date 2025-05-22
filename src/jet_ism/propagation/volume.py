import numpy as np
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_time_in_megayr)


def volume_fraction(storage, boxsize, density, machnumber, jetpower, start=15, i_file=11):
    """
    Create a table with jet tracer volume fraction and central density evolution with time
    Input: density    (number density of the simulation)
           machnumber (mach number of the simulation)
           jetpower   (jet power of the simulation)
           start      (initial snapshot from the turbulent box)
           
    Output: array with three columns: time un Myr, jet tracer volume fraction, and mean number density within central 50pc
    
    """    
    simulation_directory_jet = str(f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{boxsize}/turb_jet_d{density}_m{machnumber}/jet{jetpower}_{start}')
    output_directory_jet = simulation_directory_jet + "/output/"
    figures_directory_jet = simulation_directory_jet + "/output/figures/"
    
    threshold_tracer = 1e-3
    result = []
    while True: #True or a number for a set length
        i_file += 1
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory_jet + filename, "r+")
        except:
            try:
                i_file += 1
                filename = "snap_%03d.hdf5" % (i_file)
                snap_data = h5py.File(output_directory_jet + filename, "r+")
            except:
                break
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        center = snap_data['Header'].attrs['BoxSize'] / 2
        
        mask = (np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) < 500)
        
        volume = snap_data['PartType0/Masses'][:][mask]/snap_data['PartType0/Density'][:][mask]
        tracer = snap_data['PartType0/Jet_Tracer'][:][mask]
        jet_volume = volume[tracer > threshold_tracer]
        jet_volume_fraction = np.sum(jet_volume)/np.sum(volume)
        result.append([time, jet_volume_fraction])
    return np.array(result).T