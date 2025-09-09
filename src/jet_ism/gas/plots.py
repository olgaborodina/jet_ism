
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity, _crop_img, _make_gif)

from .general import (get_temp)

import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl

def t_histogram(output_directory, snapshot_numbers, colors=None, xmin=3.3, xmax=9, savefig=False):
    """
    Plot the temperature histogram for the given snapshot numbers
    Input: output_directory (string, location of the output files)
           snapshot_numbers (list of integers, snapshot numbers to plot)
           colors (list of strings same length as snapshot_numbers, colors for the different snapshots, default is None)
           xmin (float, minimum temperature in log10(K))
           xmax (float, maximum temperature in log10(K))
           savefig (boolean, save the figure if only one snapshot is provided, default is False)
    Output: None
    """    
    t_bins = np.linspace(xmin, xmax, 151)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, i_file in enumerate(snapshot_numbers):
        filename = "snap_%03d.hdf5" % (i_file)
        snap_data = h5py.File(output_directory + filename,'r')
        
        time = get_time_from_snap(snap_data)
        masses = snap_data['PartType0/Masses'][:]
        temperatures = get_temp(snap_data, GAMMA)
        if colors is None:
            plt.hist(np.log10(temperatures), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., label="t=%.2f Myr"%(time * unit_time_in_megayr), histtype='step')
        else:
            plt.hist(np.log10(temperatures), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., label="t=%.2f Myr"%(time * unit_time_in_megayr), histtype='step', color=colors[i])

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

def n_histogram(output_directory, snapshot_numbers, colors=None, xmin=-5, xmax=5, savefig=False):
    """
    Plot the number density histogram for the given snapshot numbers
    Input: output_directory (string, location of the output files)
           snapshot_numbers (list of integers, snapshot numbers to plot)
           colors (list of strings same length as snapshot_numbers, colors for the different snapshots, default is None)
           xmin (float, minimum number density in log10(cm^-3))
           xmax (float, maximum number density in log10(cm^-3))
           savefig (boolean, save the figure if only one snapshot is provided, default is False)
    Output: None
    """    
    t_bins = np.linspace(xmin, xmax, 151)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, i_file in enumerate(snapshot_numbers):
        filename = "snap_%03d.hdf5" % (i_file)
        snap_data = h5py.File(output_directory + filename,'r')
        
        time = get_time_from_snap(snap_data)
        masses = snap_data['PartType0/Masses'][:]
        densities = snap_data['PartType0/Density'][:] * rho_to_numdensity
        if colors is None:
            plt.hist(np.log10(densities), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., label="t=%.2f Myr"%(time * unit_time_in_megayr), histtype='step')
        else:
            plt.hist(np.log10(densities), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., label="t=%.2f Myr"%(time * unit_time_in_megayr), histtype='step', color=colors[i])

    plt.legend(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(1, 1e10)
    
    plt.grid(ls='--', c='gray', alpha=0.2, zorder=100)
    plt.yscale('log')
    plt.xlabel(r"log n [cm $^{-3}$]", fontsize=15)
    plt.ylabel(r"Mass [$M_\odot$]", fontsize=15)
    if len(snapshot_numbers) == 1 and savefig == True:
        plt.savefig(output_directory + f'figures/hist_numdensity_{snapshot_numbers[0]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def t_histogram_different_dirs(output_directories, snapshot_number, labels, colors=None, xmin=3.3, xmax=9):
    """
    Plot the temperature histogram for the given snapshot number in different directories
    Input: output_directories (list of strings, locations of the output files)
           snapshot_number (integer, snapshot number to plot)
           colors (list of strings same length as output_directories, colors for the different directories, default is None)
           labels (list of strings, labels for the different directories for plot legend)
           xmin (float, minimum temperature in log10(K))
           xmax (float, maximum temperature in log10(K))
    Output: None
    """    
    t_bins = np.linspace(xmin, xmax, 151)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, output_directory in enumerate(output_directories):
        filename = "snap_%03d.hdf5" % (snapshot_number)
        snap_data = h5py.File(output_directory + filename,'r')
        
        time = get_time_from_snap(snap_data)
        masses = snap_data['PartType0/Masses'][:]
        temperatures = get_temp(snap_data, GAMMA)
        if colors is None:
            plt.hist(np.log10(temperatures), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., histtype='step', label=labels[i])
        else:
            plt.hist(np.log10(temperatures), weights=masses, density=False, 
                 bins=t_bins, alpha=0.9, linewidth=2., histtype='step', label=labels[i], color=colors[i])

    plt.title("t=%.2f Myr"%(time * unit_time_in_megayr))
    plt.axvline(4.45, c='black', label='log T = 4.45')
    plt.legend(fontsize=12)
    plt.xlim(xmin, xmax)
    plt.ylim(1, 1e10)
    
    plt.grid(ls='--', c='gray', alpha=0.2, zorder=100)
    plt.yscale('log')
    plt.xlabel("log T [K]", fontsize=15)
    plt.ylabel(r"Mass [$M_\odot$]", fontsize=15)



def t_histogram_gif(figures_directory, timestep=4):
    """
    Create a gif from the temperature histogram snapshots in the given directory
    Input: figures_directory (string, location of the figure files)
           timestep (integer, time between frames in the gif in ms, default is 4)
    Output: None
    """
    ifilename = figures_directory + '/hist_temperature*.png'
    ofilename = figures_directory + '/hist_temperature.gif'
    _make_gif(ifilename, ofilename, timestep)


def phase_diagram(output_directory, snapshot_number, xmin=-3, xmax=8, ymin=0, ymax=9, vmin=1e-5, vmax=0.3e0, savefig=False):
    """
    Plot the phase diagram for the given snapshot number and in a given output directory
    Input: output_directory (string, location of the output files)
           snapshot_number (integer, snapshot number to plot)
           xmin (float, minimum number density in log10(cm^-3))
           xmax (float, maximum number density in log10(cm^-3))
           ymin (float, minimum temperature in log10(K))
           ymax (float, maximum temperature in log10(K))
           vmin (float, minimum value for the color scale, default is 1e-5)
           vmax (float, maximum value for the color scale, default is 0.3e0)
           savefig (boolean, save the figure, default is False)
    Output: None
    """
    density_bins = np.linspace(xmin, xmax, 500)
    temperature_bins = np.linspace(ymin, ymax, 500)
    filename = "snap_%03d.hdf5" % (snapshot_number)
    snap_data = h5py.File(output_directory + filename, "r")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    time = get_time_from_snap(snap_data)
    masses = snap_data['PartType0/Masses'][:]
    densities = snap_data['PartType0/Density'][:] * rho_to_numdensity
    temperatures = get_temp(snap_data, GAMMA)
    
    volumes = masses / densities * rho_to_numdensity
    h = ax.hist2d(np.log10(densities), np.log10(temperatures), weights=masses, 
                  bins=[density_bins, temperature_bins], density=True, norm=mpl.colors.LogNorm(vmin, vmax))
    
    ax.set_xlabel(r'log $n_H$')
    ax.set_ylabel('log T')
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(snap_data) * unit_time_in_megayr))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(ls='--', alpha=0.5, zorder=0)
    plt.colorbar(h[3], ax=ax, label=r'Mass, $M_\odot$')
    
    if savefig == True:
        plt.savefig(output_directory + f'figures/phasediagram_{snapshot_number}.png', dpi=300, bbox_inches='tight')

def phasediagram_gif(figures_directory, timestep=4):
    """
    Create a gif from the phase diagram images in the given directory
    Input: figures_directory (string, location of the figure files)
           timestep (integer, time between frames in the gif in ms, default is 4)
    Output: None
    """       
    ifilename = figures_directory + '/hist_phasediagram*.png'
    ofilename = figures_directory + '/hist_phasediagram.gif'
    _make_gif(ifilename, ofilename, timestep)
