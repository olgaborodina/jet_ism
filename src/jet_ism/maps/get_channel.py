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


def get_channel_gas_rhotemp(file,center, lbox,slab_width=None,
                    imsize=2000, smlfac=1.0, orientation='xy'):
    """
    get channels of the gas density and mass weighted temperature 
    """
    if slab_width is None:
        slab_width = lbox


    part = h5py.File(file,'r')
    mask,gaspos= cut(part['PartType0'],center,lbox,slab_width,orientation)
    print ("num gas:",len(gaspos))

    
    gasm = part['PartType0/Masses'][:][mask]
    gast = get_temp(part, GAMMA)[mask]
    gasvol = gasm / part['PartType0/Density'][:][mask]
    gassl = smlfac * (gasvol ** (1 / 3))
        
    p_resolution = lbox/imsize
    psml = gassl/p_resolution
    psml[psml<1]=1
    
    #---------------------------------------    
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    channels = painter.paint(gasdev, psml, [gasm, gasm*gast], (imsize, imsize), np=8)
    channels[1] /= channels[0]
    
    return channels

def get_channel_jet_tracer(file,center, lbox,slab_width=None, jetcolumn=-1,
                           imsize=2000, smlfac=1.0,orientation='xy', weight='tracer'):
    """
    get channels of the jet tracer, if weight='mass', return mass weighted jet tracer
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
    gasjet[gasjet<1e-20] = 1e-20
        
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox,slab_width,imsize,orientation) 
    if weight=='mass':
        channels = painter.paint(gasdev, psml, [gasm, gasm * gasjet], (imsize, imsize), np=8)    
        channels[1] /= channels[0]
    elif weight=='tracer':
        channels = painter.paint(gasdev, psml, [gasjet, gasjet], (imsize, imsize), np=8)
    else:
        raise NotImplementedError
    return channels

def get_channel_shock(file,center,lbox, slab_width=None, shockcolumn=-1,
                           imsize=2000, smlfac=1.0, orientation='xy'):
    """
    get channels of the mach number in shocks, if weight='mass', return mass weighted mach numbers
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')['PartType0']
    mask,gaspos= cut(part,center,lbox,slab_width,orientation)
    
    gasm = part['Masses'][:][mask]
    gasvol = gasm / part['Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    
    gasshock = part['Machnumber'][:][mask]
    if len(gasshock.shape)>1:
        gasshock = gasshock[:,shockcolumn] 
    gasshock[gasshock<1e-20] = 1e-20
    
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
    get channels of the mach number in shocks, if weight='mass', return mass weighted mach numbers
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
    gassoundspeed[gassoundspeed<1e-5] = 1e-5
    
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos,lbox, slab_width, imsize, orientation) 
    channels = painter.paint(gasdev, psml, [gassoundspeed], (imsize, imsize), np=8)
    return channels



def get_channel_bowshock(file,center,lbox,slab_width=None,jetcolumn=-1,
                           imsize=2000,smlfac=1.0,orientation='xy'):
    """
    get channels of hot areas around jet, if weight='mass', return mass weighted mach numbers
    """
    if slab_width is None:
        slab_width = lbox
    
    part = h5py.File(file,'r')
    mask, gaspos= cut(part['PartType0'], center, lbox, slab_width, orientation)
    
    gasm = part['PartType0/Masses'][:][mask]
    gasvol = gasm/part['PartType0/Density'][:][mask]
    gassl = smlfac * (gasvol ** (1/3))
    gast = get_temp(part, GAMMA)[mask]
    gasjet = part['PartType0/Jet_Tracer'][:][mask]
    
    hot_nonjet = np.zeros_like(gasjet)
    if len(hot_nonjet.shape)>0:
        mask_hot_nonjet = ((gast > 10 ** 4.4) & (gasjet < 1e-5))
        hot_nonjet[mask_hot_nonjet] = 1
        
    p_resolution = lbox / imsize
    psml = gassl / p_resolution
    psml[psml < 1] = 1
    
    #---------------------------------------
    gasdev = pos2device(gaspos, lbox, slab_width, imsize, orientation) 
    channels = painter.paint(gasdev, psml, [hot_nonjet], (imsize, imsize), np=8)
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