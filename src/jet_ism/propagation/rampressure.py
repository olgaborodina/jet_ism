import numpy as np
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, unit_mass, unit_time_in_megayr, unit_density)
from ..gas.general import calculate_ram_pressure

import tqdm
from multiprocessing import Pool
from multiprocessing import shared_memory
import time


def x_loop(i):    
    X = x_axis[i]
    dx = x_axis[1] - x_axis[0]
    dist = (x - center[0] - X) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    index = (dist < 30) & (x < X - dx) & (x > X + dx)  
    
    if (index).sum() > 0:
        return np.mean(ram_pressure[index]), np.mean(jet_tracer[index])
    else:
        index = np.argmin(dist)    
        return ram_pressure[index], jet_tracer[index]

def x_loop_rp_only(i):
    dist = (x - center - x_axis[i]) ** 2 + (y - center) ** 2 + (z - center) ** 2
    index = (dist < 15) & (x < x_axis[i] - dx) & (x > x_axis[i] + dx)  
    
    if (index).sum() > 0:
        return np.mean(ram_pressure[index])
    else:
        index = np.argmin(dist)    
        return ram_pressure[index]
        
def calculate_rampressure_axis(output_directory, i_file, f_file, x_axis_size=101, nkernels=32, chunk=1):
    ram_pressure_list_parallel = []
    jet_tracer_fall_x_parallel_left = []
    jet_tracer_fall_x_parallel_right = []
    time_parallel = []

    global x_axis
    x_axis = np.linspace(-500, 500, x_axis_size)
    mask_left = (x_axis < -30)
    mask_right = (x_axis > 30)

        
    for file in tqdm.trange(i_file, f_file):
        try:
            filename = "snap_%03d.hdf5" % (file)
            snap_data = h5py.File(output_directory + filename, "r")
        except:
            break
        global x, y, z, ram_pressure, jet_tracer, center
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        vx, vy, vz = snap_data['PartType0/Velocities'][:].T
        density = snap_data['PartType0/Density'][:]
        ram_pressure = calculate_ram_pressure(snap_data)
        jet_tracer = snap_data['PartType0/Jet_Tracer'][:]
        center = np.array([0.5 * snap_data["Header"].attrs["BoxSize"]] * 3)
        
        pool = Pool(nkernels)

        ram_pressure_axis = []
        jet_tracer_axis = []
        for ram_pressure_axis_1, jet_tracer_axis_1 in pool.map(x_loop, np.arange(len(x_axis)), chunksize=chunk):
            jet_tracer_axis.append(jet_tracer_axis_1)
            ram_pressure_axis.append(ram_pressure_axis_1)
    
        pool.close()

        if sum(np.array(jet_tracer_axis)[mask_left]<2e-3) > 0:            
            jet_tracer_fall_x_parallel_left.append(x_axis[mask_left][np.array(jet_tracer_axis)[mask_left]<2e-3][-1])
        else: jet_tracer_fall_x_parallel_left.append(-501)
        if sum(np.array(jet_tracer_axis)[mask_right]<2e-3) > 0: 
            jet_tracer_fall_x_parallel_right.append(x_axis[mask_right][np.array(jet_tracer_axis)[mask_right]<2e-3][0])
        else: jet_tracer_fall_x_parallel_right.append(501)
        ram_pressure_list_parallel.append(np.array(ram_pressure_axis))
    
        time_parallel.append(get_time_from_snap(snap_data) * unit_time_in_megayr - 15)
        
    time_parallel = np.array(time_parallel)
    return ram_pressure_list_parallel, jet_tracer_fall_x_parallel_left, jet_tracer_fall_x_parallel_right, x_axis, time_parallel


def P_ISM_model(density, c_s, mach, b):
     return (1 + b * mach) * mach ** 2 * density * c_s ** 2 * 1e10 * (mu * PROTONMASS)


def calculate_rampressure_full():

    i_file = 13 # skip snap 0
    ram_pressure_list_parallel = []
    time_parallel = []
    
    x_axis = np.concatenate((np.linspace(500, 1000, 81), np.linspace(-1000, -500, 81)))
    mask_left = (x_axis < -30)
    mask_right = (x_axis > 30)
    
    dx = x_axis[1] - x_axis[0]
    
    #for i_file in tqdm.trange(12, 42):
        
    for i_file in tqdm.trange(60, 90):
        filename = "snap_%03d.hdf5" % (i_file)
    
        snap_data = h5py.File(output_directory + filename, "r")
        
        x, y, z = snap_data['PartType0/Coordinates'][:].T
        vx, vy, vz = snap_data['PartType0/Velocities'][:].T
        density = snap_data['PartType0/Density'][:]
        ram_pressure = calculate_ram_pressure(filename)
        center = 0.5 * snap_data["Header"].attrs["BoxSize"]
    
        pool = Pool(32)
        ram_pressure_axis = []
        for ram_pressure_axis_1 in pool.map(x_loop_media, np.arange(len(x_axis)), chunksize=1):
            ram_pressure_axis.append(ram_pressure_axis_1)
    
        pool.close()
        ram_pressure_list_parallel.append(np.array(ram_pressure_axis))
    
        time_parallel.append(get_time_from_snap(snap_data) * unit_time_in_megayr - 15)
    
    time_parallel = np.array(time_parallel)
    return ram_pressure_list_parallel, x_axis, time_parallel



def test_worker(i, x, y, z):
    # Simulate CPU-bound work: compute a large vector norm
    return np.sum((x - i)**2 + (y - i)**2 + (z - i)**2)

def test_parallel_speed(pool_size=4, n_jobs=100):
    # Generate large data arrays (simulate a big simulation snapshot)
    size = 100000
    x = np.random.rand(size)
    y = np.random.rand(size)
    z = np.random.rand(size)

    # Prepare the argument list
    args_list = [(i, x, y, z) for i in range(n_jobs)]

    start = time.time()

    with Pool(pool_size) as pool:
        results = pool.starmap(test_worker, args_list)

    elapsed = time.time() - start
    print(f"Pool size {pool_size}: Time elapsed = {elapsed:.2f} seconds")
    return results