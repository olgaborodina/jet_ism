import glob
import os
from natsort import natsorted
import soxs
from matplotlib import pyplot as plt

import yt
from yt.utilities.cosmology import Cosmology
from yt.units import mp
import pyxsim

import .soxso as o
import h5py
import astropy.io.fits as fits
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import matplotlib.tri as tri
import matplotlib as mpl

from astropy.coordinates import Angle
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

def _emission_measure(field, data):
    n_HdV = data["gas", "mass"] * X_H / mp
    n_e = data["gas", "density"] * data["PartType0", "ElectronAbundance"] * X_H/mp 
    return n_e*n_HdV

def make_fits(exp_time = (200.0, "ks"),  # exposure time
area = (1000.0, "cm**2"),  # collecting area
redshift = 0.0165, #1e-4
coordinates = ((3 + 1/60 + 42.33/3600) * 15, 35 + 12/60 + 20.03 / 3600),
dist = 70 * 1e,6 #pc
emin = 0.5,
emax = 7,
sigma=2,
levels = np.linspace(0.1, 1, 6)):

