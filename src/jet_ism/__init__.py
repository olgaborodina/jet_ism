""" load libraries """
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from PIL import Image
from natsort import natsorted
import glob

__all__ = [
    'outflows', 'maps', 'gas', 'stars', 'blackholes', 'propagation', 'styles',
    'get_time_from_snap', 'get_output_directory',
    'megayear', 'h', 'unit_velocity', 'unit_length', 'unit_mass', 'unit_energy',
    'unit_density', 'unit_time', 'unit_time_in_megayr', 'mu',
    'PROTONMASS', 'BOLTZMANN', 'GRAVITY', 'LIGHTSPEED', 'sigma_T', 'rho_to_numdensity', 'GAMMA'
]

# Global constants
global megayear, h, unit_velocity, unit_length, unit_mass, unit_energy
global unit_density, unit_time, unit_time_in_megayr, mu
global PROTONMASS, BOLTZMANN, GRAVITY, LIGHTSPEED, rho_to_numdensity, GAMMA, sigma_T
global get_time_from_snap

# Physical constants and unit conversions
megayear = 3.15576e13
h = 1#float(0.67)
unit_velocity = np.float64(100000)
unit_length = np.float64(3.08568e+18 / h)
unit_mass = np.float64(1.989e+33 / h)
unit_energy = unit_mass * unit_velocity * unit_velocity
unit_density = unit_mass / unit_length / unit_length / unit_length 
unit_time = unit_length / unit_velocity
unit_time_in_megayr = unit_time / megayear

mu = np.float64(0.6165) # mean molecular weight 

PROTONMASS = np.float64(1.67262178e-24)
BOLTZMANN = np.float64(1.38065e-16)
GRAVITY = np.float64(6.6738e-8) # G in cgs
LIGHTSPEED = np.float64(3e10)  #cm/s
sigma_T = 6.6524e-25                      # cm^2, Thomson scattering cross-section for the electron 
rho_to_numdensity = 1. * unit_density / (mu * PROTONMASS) # to cm^-3

GAMMA = 5 / 3

def get_time_from_snap(snap_data):
    """
    Find what time the snapshot has in its header.
    Input: snap_data (snapshot)
    Output: time in sim units
    
    """
    return float(snap_data["Header"].attrs["Time"])


def get_output_directory(case, density, mach, jetpower, time, storage='holystore01'):
    """
    Find what time the snapshot has in its header.
    Input: snap_data (snapshot)
    Output: time in sim units
    
    """
    output_directory = f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{case}/turb_jet_d{density}_m{mach}/jet{jetpower}_{time}/output/'
    if jetpower=='turb':
        output_directory = f'/n/{storage}/LABS/hernquist_lab/Users/borodina/{case}/turb_jet_d{density}_m{mach}/{jetpower}_{time}/output/'
    return output_directory

def _crop_img(im):
    """
    Crop the image to remove the black bars
    Input: im (image)
    Output: cropped image
    """
    width, height = im.size
    left = 9
    top =  3
    right = width - 3
    bottom = height - 9
    im = im.crop((left, top, right, bottom))
    return im

def _make_gif(ifilename, ofilename, timestep=4):
   """
    Create a gif from the images in the given directory
    Input: ifilename (string, location of the input files with asterisks, e.g. 'figures/*.png')
           ofilename (string, location of the output file, e.g. 'figures/animation.gif')
           timestep (integer, time between frames in the gif in ms, default is 4)
    Output: None
    """
    imgs = natsorted(glob.glob(ifilename))
    
    frames = []
    
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(_crop_img(new_frame))
    
    frames[0].save(ofilename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=len(imgs) * timestep, loop=0)

def weighted_std(values, weights=None):
    """
    Return the weighted standard deviation.
    Input: values (numpy array of floats)
           weights (numpy array of floats, same shape as values, default is None)
    Output: weighted standard deviation
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)