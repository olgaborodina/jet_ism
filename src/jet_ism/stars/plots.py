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

from ..gas.general import (get_temp)

import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl



def sfr_diagram(output_directory, snapshot_number, xmin=1e-3, xmax=1e8, ymin=-1e-6, ymax=2e-5, vmin=1, vmax=1e6, nbins=500, xlog=True, ylog=True, savefig=False):
    
    density_bins = np.linspace(xmin, xmax, nbins)
    sfr_bins = np.linspace(ymin, ymax, nbins)

    if xlog == True:
        density_bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins)
    if ylog == True:
        sfr_bins = np.logspace(np.log10(ymin), np.log10(ymax), nbins) 
    filename = "snap_%03d.hdf5" % (snapshot_number)
    snapshot = h5py.File(output_directory + filename, "r")
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    time = get_time_from_snap(snapshot)
    masses = snapshot['PartType0/Masses'][:]
    densities = snapshot['PartType0/Density'][:] * rho_to_numdensity
    sfr = snapshot['PartType0/StarFormationRate'][:]
    
    h = ax.hist2d(densities, sfr, weights=masses, 
                  bins=[density_bins, sfr_bins], density=True, norm=mpl.colors.LogNorm(vmin, vmax))
    
    ax.set_xlabel(r'$n_H$')
    ax.set_ylabel('SFR')

    if xlog == True:
        ax.set_xscale('log')
    if ylog == True:
        ax.set_yscale('log')       
    ax.set_title("t=%.2f Myr"%(get_time_from_snap(snapshot) * unit_time_in_megayr))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(ls='--', alpha=0.5, zorder=0)
    plt.colorbar(h[3], ax=ax, label=r'Mass, $M_\odot$')
    
    if savefig == True:
        plt.savefig(output_directory + f'figures/sfrdiagram_{snapshot_number}.png', dpi=300, bbox_inches='tight')
