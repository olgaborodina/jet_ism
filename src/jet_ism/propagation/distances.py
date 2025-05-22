import numpy as np
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_time_in_megayr)


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

def percentile_distance_3(storage, box, density, machnumber, jetpower, start=15, i_file=12):
    """
    Create a table with jet tracer volume fraction and central density evolution with time
    Input: box (type of the simulation: '2kpc', 'self-consistent', 'three-phases')
           density    (number density of the simulation)
           machnumber (mach number of the simulation)
           jetpower   (jet power of the simulation)
           start      (initial snapshot from the turbulent box)
           i_file     (initial snapshot for the calculation)
    Output: array with three columns: time in Myr, jet tracer volume fraction, and mean number density within central 50pc
    
    """    
    simulation_directory = str(f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{box}/turb_jet_d{density}_m{machnumber}/jet{jetpower}_{start}')
    output_directory = simulation_directory + "/output/"
    figures_directory = simulation_directory + "/output/figures/"
    
    threshold_tracer = 1e-3
    distance_array = []
    time = 0
    while True: #and time < 20: #True or a number for a set length
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
                
        center = snap_data['Header'].attrs['BoxSize'] / 2       
        x = snap_data['PartType0/Coordinates'][:, 0] - center
        y = snap_data['PartType0/Coordinates'][:, 1] - center
        z = snap_data['PartType0/Coordinates'][:, 2] - center
        jet_tracer = snap_data['PartType0/Jet_Tracer'][:]
        mass = snap_data['PartType0/Masses'][:]

        mask_jet = jet_tracer > threshold_tracer
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        distance_sq = x ** 2 + y ** 2 + z ** 2

        if len(distance_sq[mask_jet]) > 0:
            distance_100 = np.sqrt(weighted_percentile(distance_sq[mask_jet], jet_tracer[mask_jet] * mass[mask_jet], 1))
            distance_80 = np.sqrt(weighted_percentile(distance_sq[mask_jet], jet_tracer[mask_jet] * mass[mask_jet], 0.8))
            distance_50 = np.sqrt(weighted_percentile(distance_sq[mask_jet], jet_tracer[mask_jet] * mass[mask_jet], 0.5))
        else:
            distance_100 = 0
            distance_80 = 0
            distance_50 = 0

        distance_array.append([time, distance_100, distance_80, distance_50])
    return np.array(distance_array).T

def percentile_evolution(storage, boxsize, density, machnumber, jetpower, start=15):
    """
    Create a table with jet tracer volume fraction and central density evolution with time
    Input: density    (number density of the simulation)
           machnumber (mach number of the simulation)
           jetpower   (jet power of the simulation)
           start      (initial snapshot from the turbulent box)
           a
    Output: array with three columns: time un Myr, jet tracer volume fraction, and mean number density within central 50pc
    
    """    
    simulation_directory = str(f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{boxsize}/turb_jet_d{density}_m{machnumber}/jet{jetpower}_{start}')
    output_directory = simulation_directory + "/output/"
    figures_directory = simulation_directory + "/output/figures/"
    distances = np.linspace(750, 0, 501)
    fractions = []
    i_file = 0
    time = 0
    if type(start) == str:
        start = 15
    
    while True and time < start + 5: #True or a number for a set length
        i_file += 1
        filename = "snap_%03d.hdf5" % (i_file)
        try:
            snap_data = h5py.File(output_directory + filename, "r")
            if get_time_from_snap(snap_data) * unit_time_in_megayr < start:
                continue
        except:
            try:
                i_file += 1
                filename = "snap_%03d.hdf5" % (i_file)
                snap_data = h5py.File(output_directory + filename, "r")
            except:
                break
        center = snap_data['Header'].attrs['BoxSize'] / 2
        x = snap_data['PartType0/Coordinates'][:, 0] - center
        y = snap_data['PartType0/Coordinates'][:, 1] - center
        z = snap_data['PartType0/Coordinates'][:, 2] - center
        jet_tracer = snap_data['PartType0/Jet_Tracer'][:]
        mass = snap_data['PartType0/Masses'][:]
    
        mask_jet = jet_tracer > 1e-3
        time = get_time_from_snap(snap_data) * unit_time_in_megayr
        distance_jet = np.sqrt(x[mask_jet] ** 2 + y[mask_jet] ** 2 + z[mask_jet] ** 2)
        mass_jet = mass[mask_jet] * jet_tracer[mask_jet]
        fractions_i = np.zeros_like(distances)
        for i, d in enumerate(distances):
            if distance_jet.max() > d:
                fractions_i[i] = np.sum(mass_jet[distance_jet < d]) / np.sum(mass_jet)
            else:
                fractions_i[i] = 0
        fractions.append(fractions_i)
    return np.array(fractions).T, time 