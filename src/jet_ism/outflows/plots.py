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
from distributions import *

import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl

def histogram(output_directory, snapshot_number, velmin=-900, velmax=900, title='', savefig=False):
    bins, velocities_hot, velocities_warm, velocities_cold = calculate_distribution(output_directory, snapshot_number, velmin, velmax)
    
    fig, ax = plt.subplots(figsize=(7,4), sharey=True)
    
    ax.step(bins[:-1], velocities_cold, where='mid', label='cold gas', color='tab:blue', linewidth=1.5)
    ax.step(bins[:-1], velocities_warm, where='mid', label='warm gas', color='tab:green', linewidth=1.5)
    ax.step(bins[:-1], velocities_hot, where='mid', label='hot gas', color='tab:red', linewidth=1.5)
    ax.grid(ls='--', zorder = 100, alpha=0.5)
    ax.set_xlim(velmin, velmax)
    ax.set_ylim(1e-2, 1e6)
    ax.set_xlabel(r"$v_r$ [km s$^{-1}$]", fontsize=15)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\frac{M_\odot}{\text{km s}^{-1}}$ of cells with ' + '\n'+ r'$r_\text{cell} \in [400; 500]$ pc', fontsize=12)
    plt.legend()
    plt.title(title)
    if savefig == True:
        plt.savefig(output_directory + f'figures/outflows_{snapshot_number}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def histogram_average(output_directory, snapshot_number, velmin=-900, velmax=900, title='average', savefig=False):
    bins, velocities_hot, velocities_warm, velocities_cold = calculate_distribution_average(output_directory, snapshot_number, velmin, velmax)
    
    fig, ax = plt.subplots(figsize=(7,4), sharey=True)
    
    ax.step(bins[:-1], velocities_cold, where='mid', label='cold gas', color='tab:blue', linewidth=1.5)
    ax.step(bins[:-1], velocities_warm, where='mid', label='warm gas', color='tab:green', linewidth=1.5)
    ax.step(bins[:-1], velocities_hot, where='mid', label='hot gas', color='tab:red', linewidth=1.5)
    ax.grid(ls='--', zorder = 100, alpha=0.5)
    ax.set_xlim(velmin, velmax)
    ax.set_ylim(1e-2, 1e6)
    ax.set_xlabel(r"$v_r$ [km s$^{-1}$]", fontsize=15)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\frac{M_\odot}{\text{km s}^{-1}}$ of cells with ' + '\n'+ r'$r_\text{cell} \in [400; 500]$ pc', fontsize=12)
    plt.legend()
    plt.title(title)
    if savefig == True:
        plt.savefig(output_directory + f'figures/outflows_{snapshot_number}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    