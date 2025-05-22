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

import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl

def histogram(output_directory, snapshot_numbers, xmin=3.3, xmax=9, savefig=False):
    
    t_bins = np.linspace(xmin, xmax, 151)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i_file in snapshot_numbers:
        filename = "snap_%03d.hdf5" % (i_file)
        snapshot = h5py.File(output_directory + filename,'r')
        
        time = get_time_from_snap(snapshot)
        masses = snapshot['PartType0/Masses'][:]
        temperatures = get_temp(snapshot, GAMMA)
        
        plt.hist(np.log10(temperatures), weights=masses, density=False, 
             bins=t_bins, alpha=0.9, linewidth=2., label="t=%.2f Myr"%(time * unit_time_in_megayr), histtype='step')

    plt.axvline(4.45, c='black', label='log T = 4.45')
    plt.legend(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(1, 1e10)
    
    plt.grid(ls='--', c='gray', alpha=0.2, zorder=100)
    plt.yscale('log')
    plt.xlabel("log T [K]", fontsize=15)
    plt.ylabel(r"Mass [$M_\odot$]", fontsize=15)
    if len(snapshot_numbers) == 1 and savefig == True:
        plt.savefig(output_directory + f'figures/hist_temperature_{snapshot_numbers[0]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def histogram_different_dirs(output_directories, names, snapshot_number, xmin=3.3, xmax=9):
    
    t_bins = np.linspace(xmin, xmax, 151)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, output_directory in enumerate(output_directories):
        filename = "snap_%03d.hdf5" % (snapshot_number)
        snapshot = h5py.File(output_directory + filename,'r')
        
        time = get_time_from_snap(snapshot)
        masses = snapshot['PartType0/Masses'][:]
        temperatures = get_temp(snapshot, GAMMA)
        
        plt.hist(np.log10(temperatures), weights=masses, density=False, 
             bins=t_bins, alpha=0.9, linewidth=2., histtype='step', label=names[i])
    plt.title("t=%.2f Myr"%(time * unit_time_in_megayr))
    plt.axvline(4.45, c='black', label='log T = 4.45')
    plt.legend(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(1, 1e10)
    
    plt.grid(ls='--', c='gray', alpha=0.2, zorder=100)
    plt.yscale('log')
    plt.xlabel("log T [K]", fontsize=15)
    plt.ylabel(r"Mass [$M_\odot$]", fontsize=15)

def _crop_img(im):
    width, height = im.size
    left = 9
    top =  3
    right = width - 3
    bottom = height - 9
    im = im.crop((left, top, right, bottom))
    return im

def histogram_gif(figures_directory, timestep=4):
        
    ifilename = figures_directory + '/hist_temperature*.png'
    ofilename = figures_directory + '/hist_temperature.gif'
    imgs = natsorted(glob.glob(ifilename))
    
    frames = []
    
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(_crop_img(new_frame))
    
    frames[0].save(ofilename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=len(imgs) * timestep, loop=0)

def phase_diagram(output_directory, snapshot_number, xmin=-3, xmax=8, ymin=0, ymax=9, vmin=1e-5, vmax=0.3e0, savefig=False):
    
    density_bins = np.linspace(xmin, xmax, 500)
    temperature_bins = np.linspace(ymin, ymax, 500)
    filename = "snap_%03d.hdf5" % (snapshot_number)
    snapshot = h5py.File(output_directory + filename, "r")
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    time = get_time_from_snap(snapshot)
    masses = snapshot['PartType0/Masses'][:]
    densities = snapshot['PartType0/Density'][:] * rho_to_numdensity
    temperatures = get_temp(snapshot, 5/3)
    
    volumes = masses / densities * rho_to_numdensity
    h = ax.hist2d(np.log10(densities), np.log10(temperatures), weights=masses, 
                  bins=[density_bins, temperature_bins], density=True, norm=mpl.colors.LogNorm(vmin, vmax))
    
    ax.set_xlabel(r'log $n_H$')
    ax.set_ylabel('log T')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(snapshot) * unit_time_in_megayr))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(ls='--', alpha=0.5, zorder=0)
    plt.colorbar(h[3], ax=ax, label=r'Mass, $M_\odot$')
    
    if savefig == True:
        plt.savefig(output_directory + f'figures/phasediagram_{snapshot_number}.png', dpi=300, bbox_inches='tight')

def phasediagram_gif(figures_directory, timestep=4):
        
    ifilename = figures_directory + '/hist_phasediagram*.png'
    ofilename = figures_directory + '/hist_phasediagram.gif'
    imgs = natsorted(glob.glob(ifilename))
    
    frames = []
    
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(_crop_img(new_frame))
    
    frames[0].save(ofilename, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=len(imgs) * timestep, loop=0)
