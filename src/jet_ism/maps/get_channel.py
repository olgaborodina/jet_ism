from PIL import Image
from natsort import natsorted
import glob

import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path


import matplotlib.pyplot as plt    ## plot stuff
from matplotlib import animation
import matplotlib as mpl

from matplotlib.colors import ListedColormap,colorConverter
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .. import (unit_velocity, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity)
from .base import *
from ..gas.general import *


def get_channel_gas_rhotemp(file, center, lbox, slab_width=None,
                    imsize=2000, smlfac=1.0, orientation='xy'):
    """
    Get channels of the gas density and mass weighted temperature
    Input: file (filename, including directory)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
    Output: channels (2, imsize, imsize), channels[0]: column density, channels[1]: mass weighted temperature
    """
    if slab_width is None:
        slab_width = lbox


    part = h5py.File(file,'r')
    mask, gaspos= cut(part['PartType0'], center, lbox, slab_width, orientation)
    print ("num gas:", len(gaspos))
    
    gasm = part['PartType0/Masses'][:][mask]
    gast = get_temp(part, GAMMA)[mask]
    gasvol = gasm / part['PartType0/Density'][:][mask]
    gassl = smlfac * (gasvol ** (1 / 3))
        
    p_resolution = lbox/imsize
    psml = gassl/p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------    
    gasdev = pos2device(gaspos, lbox, slab_width, imsize, orientation) 
    channels = painter.paint(gasdev, psml, [gasm, gasm*gast], (imsize, imsize), np=8)
    channels[1] /= channels[0]
    
    return channels

def get_channel_jet_tracer(file, center, lbox, slab_width=None, jetcolumn=-1,
                           imsize=2000, smlfac=1.0, orientation='xy', weight='tracer'):
    """
    Get channels of the jet tracer
    Input: file (filename, including directory)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           jetcolumn (int, if jet tracer is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
           weight (string, 'mass' or 'tracer', default is 'tracer', whether to return mass weighted jet tracer or just jet tracer)
    Output: channels (2, imsize, imsize), channels[0]: column density, channels[1]: weighted jet tracer
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask, gaspos= cut(part, center, lbox, slab_width, orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm / part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gasjet = part['Jet_Tracer'][:][mask]
    if len(gasjet.shape) > 1:
        gasjet = gasjet[:, jetcolumn]    
    gasjet[gasjet < 1e-20] = 1e-20 # avoid log(0)
        
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm * gasjet], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight=='tracer':
        channels = painter.paint(gasdev, psml, [gasm, gasjet], (imsize, imsize), np=8)
    else:
        raise NotImplementedError
    return channels

def get_channel_shock(file, center, lbox, slab_width=None, shockcolumn=-1,
                           imsize=2000, smlfac=1.0, orientation='xy'):
    """
    Get channels of the mach number in shock regions
    Input: file (filename, including directory)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           shockcolumn (int, if shock tracer is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
    Output: channels (1, imsize, imsize), channels[0]: mach number
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask, gaspos= cut(part, center, lbox, slab_width, orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm / part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gasshock = part['Machnumber'][:][mask]
    if len(gasshock.shape) > 1:
        gasshock = gasshock[:,shockcolumn] 
    gasshock[gasshock < 1e-20] = 1e-20 # avoid log(0)
    
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox, slab_width, imsize,orientation) 
    channels = painter.paint(gasdev, psml, [gasshock], (imsize, imsize), np=8)
    return channels

def get_channel_soundspeed(file,center, lbox, slab_width=None, sscolumn=-1,
                           imsize=2000, smlfac=1.0, orientation='xy'):
    """
    Get channels of the sound speed
    Input: file (filename, including directory)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           sscolumn (int, if sound speed is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
    Output: channels (1, imsize, imsize), channels[0]: sound speed
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')
    mask,gaspos= cut(part['PartType0'],center,lbox,slab_width,orientation)
    
    gasm = part['PartType0/Masses'][:][mask]
    gasvol = gasm / part['PartType0/Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gassoundspeed= get_soundspeed(part, GAMMA)[mask]
    if len(gassoundspeed.shape) > 1:
        gassoundspeed = gassoundspeed[:, sscolumn]  
    gassoundspeed[gassoundspeed < 1e-5] = 1e-5 # avoid log(0)
    
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox, slab_width, imsize, orientation) 
    channels = painter.paint(gasdev, psml, [gassoundspeed], (imsize, imsize), np=8)
    return channels



def get_channel_sigma_velocity(file,center,lbox,slab_width=None, imsize=2000,smlfac=1.0,orientation='xy',weight='density_square'):
    """
    TODO: test this
    get channels of the jet tracer, if weight='mass', return mass weighted jet tracer
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask,gaspos= cut(part,center,lbox,slab_width,orientation)
    
    gasm = part['Masses'][:][mask]
    gasdens = part['Density'][:][mask]
    gasvol = gasm/part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gasjet = part['Jet_Tracer'][:][mask]
    if len(gasjet.shape)>1:
        gasjet = gasjet[:,jetcolumn]    
    gasjet[gasjet<1e-20] = 1e-20

    x, y, z = part['Coordinates'][:].T
    vx, vy, vz = part['Velocities'][:].T

    if orientation == 'xy':
        gasvel = weighted_std(vz[mask], weights=gasdens * gasdens)
    if orientation == 'xz':
        gasvel = weighted_std(vy[mask], weights=gasdens * gasdens)
    if orientation == 'yz':
        gasvel = weighted_std(vx[mask], weights=gasdens * gasdens)
    
    p_resolution = lbox/imsize
    psml = gassl/p_resolution
    psml[psml<1]=1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm*gasvel], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight=='density_square':
        channels = painter.paint(gasdev, psml, [gasjet, gasvel], (imsize, imsize), np=8)
    else:
        raise NotImplementedError
    return channels


def get_channel_sfr(file, center, lbox, slab_width=None, sfrcolumn=-1,
                    imsize=2000, smlfac=1.0, orientation='xy', weight='sfr'):
    """
    Get channels of the star formation rate (SFR)
    Input: file (filename, including directory)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           jetcolumn (int, if jet tracer is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
           weight (string, 'mass' or 'sfr', default is 'sfr', whether to return mass weighted jet tracer or just jet tracer)
    Output: channels (2, imsize, imsize), channels[0]: column density, channels[1]: weighted sfr
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask, gaspos= cut(part, center, lbox, slab_width, orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm / part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gassfr = part['StarFormationRate'][:][mask]
    if len(gassfr.shape) > 1:
        gassfr = gassfr[:, sfrcolumn]    
    gassfr[gassfr < 1e-20] = 1e-20  # avoid log(0)
        
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm * gassfr], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight=='sfr':
        channels = painter.paint(gasdev, psml, [gasm, gassfr], (imsize, imsize), np=8)
    else:
        raise NotImplementedError
    return channels



def get_channel_random(file, mask_random, center, lbox, slab_width=None,
                           imsize=2000, smlfac=1.0, orientation='xy', weight='mass'):
    """
    Get channels of a random mask
    Input: file (filename, including directory)
           mask_random (boolean array, same length as number of gas particles, True for particles to include)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           jetcolumn (int, if jet tracer is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
           weight (string, 'mass' or 'sfr', default is 'sfr', whether to return mass weighted jet tracer or just jet tracer)
    Output: channels (2, imsize, imsize), channels[0]: column density, channels[1]: weighted sfr
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask, gaspos= cut(part, center, lbox, slab_width, orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm / part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gas_random = np.ones_like(part['Masses'][:])
    gas_random[mask_random] = 1e-20

    gas_random_cut = gas_random[mask]
        
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm * gas_random_cut], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight==None:
        channels = painter.paint(gasdev, psml, [gasm, gas_random_cut], (imsize, imsize), np=8)     
    else:
        raise NotImplementedError
    return channels



def get_channel_stars(file, center, lbox, slab_width=None,
                           imsize=2000, smlfac=1.0, orientation='xy', weight='mass', age_cut=False):
    """
    Get channels of stars
    Input: file (filename, including directory, snapshot has to include star particles)
           center (3-vector, center of the box)
           lbox (float, box length) 
           slab_width (float, projection depth, default is lbox)
           jetcolumn (int, if jet tracer is a vector, which column to use, default is -1, the last column)
           imsize (int, image size, default is 2000)  
           smlfac (float, smoothing length factor, default is 1.0)
           orientation (string, projection of 'xy','yz','xz', default is 'xy')
           weight (string, 'mass' or 'sfr', default is 'sfr', whether to return mass weighted jet tracer or just jet tracer)
           age_cut (boolean, whether to only include stars formed within the last 1 Myr)
    Output: coordinates (2, N), mass(N), where N is the number of star particles in the slab
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')
    mask, gaspos= cut(part['PartType4'], center, lbox, slab_width, orientation)

    if age_cut == True:
        current_time = part['Header'].attrs['Time']
        sf_time = part['PartType4/GFM_StellarFormationTime'][:]
        add_mask = current_time - sf_time < 1
        mask_new = mask & add_mask
        mask = mask_new
    
    mass = part['PartType4/Masses'][:][mask]

    x, y, z = part['PartType4/Coordinates'][mask].T
    if orientation == 'xy':
        coords = [x, y]
    if orientation == 'yz':
        coords = [y, z]
    if orientation == 'xz':
        coords = [y, z]

    return coords, mass