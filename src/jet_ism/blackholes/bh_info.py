import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp

from .accretion import *
from .. import (unit_time_in_megayr)

def blackholes(output_directory):
    """
    Read the black hole information from the blackholes.txt file in the output directory
    Input: output_directory (directory where snapshots are stored)
    Output: black_hole_info (pandas dataframe with the black hole information, time in Myr, and jetpower column in erg/s )
    """
    black_hole_info = pd.read_csv(output_directory+'blackholes.txt', delimiter=r'\s+', names=['time', 'num', 'mass', 'mdot', 'mdot_msun_y', 'mass_tot_real', 'mdot_edd'])
    black_hole_info['time'] *= unit_time_in_megayr
    black_hole_info['jetpower'] = jetpower_from_accretion(black_hole_info['mdot_msun_y'] / cgs2solarperyear_accretion(1))    
    return black_hole_info

def blackhole_details(output_directory, number):
    """
    Read the black hole details from the blackhole_details.txt file in the output directory
    Input: output_directory (directory where snapshots are stored)
    Output: black_hole_details (pandas dataframe with the black hole details)
    """
    detailes_directory = output_directory + "/blackhole_details/"
    black_hole_details = pd.read_csv(detailes_directory+f'blackhole_details_{number}.txt', delimiter=r'\s+', names=['id', 'time', 'mass', 'mdot', 'mdot_msun_y', 'mass_tot_real', 'epsilon'])
    return black_hole_details