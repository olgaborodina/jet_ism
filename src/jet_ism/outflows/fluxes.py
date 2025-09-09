
import sys    # needed for exit codes\
import numpy as np    # scientific computing package
import pandas as pd
import scipy as scp
import h5py    # hdf5 format
from pathlib import Path
from .. import (unit_velocity, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity)

from .general import (get_temp)

def mass_flux(snapshot, projection='radial', weights='full'):
    """
    Calculate mass flux for each cell for given snapshot and different projection types.
    Input: snapshot   (snapshot)
           projection (projection type, can be 'radial', 'transverse', or 'total')
           weights    ('full' if all particles are considered and 'jet-tracer' if only jet's contribution is needed)
    Output: mass flux in simulation units
    """
    snapshot = h5py.File(directory + filename, "r")
    x, y, z = snapshot['PartType0/Coordinates'][:].T
    vx, vy, vz = snapshot['PartType0/Velocities'][:].T
    center = 0.5 * snapshot.header.BoxSize
    v_r = (vx * (x - center) + vy * (y - center) + vz * (z - center)) / np.sqrt((x - center) ** 2 +(y - center) ** 2 + (z - center) ** 2)
    if projection == 'radial':
        j = snapshot['PartType0/Density'][:] * v_r #* unit_density * unit_velocity
    elif projection == 'transverse':
        v_t = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2 - v_r ** 2)
        j = snapshot['PartType0/Density'][:] * v_t #* unit_density * unit_velocity
    elif projection == 'total':
        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        j = snapshot['PartType0/Density'][:] * v #* unit_density * unit_velocity
    else:
        raise ValueError("Wrong projection type. Expected 'radial', 'transverse', or 'total'. ")
        
    if weights == 'jet-tracer':
        j *= snap_data['PartType0/Jet_Tracer']
    return j

def mass_flux_shell(snapshot, projection, radius, dr=20):
    """
    Calculate mass flux for each cell within a certain radius for given snapshot and different projection types.
    Input: snapshot   (snapshot)
           projection (projection type, can be 'radial', 'transverse', or 'total')
           radius     (at what radius the flux is calculated)
           dr         (the shell width around radius)
    Output: mass flux in simulation units for each pixel on the shell
    """
    x_all, y_all, z_all = snapshot['PartType0/Coordinates'][:].T
    j_all = mass_flux(snapshot, projection=projection)
    
    radius_range = np.linspace(40, 500, 31)
    NSIDE = 31
    NPIX = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(nside=NSIDE, ipix=np.arange(NPIX)) # return colatitude and longtitude in radian
    vec = hp.ang2vec(theta, phi)  # return unit 3D position vector

    vec_scaled = vec * radius
    mask = ((np.sqrt(x_all ** 2 + y_all ** 2 + z_all ** 2) < radius + dr) & 
            (np.sqrt(x_all ** 2 + y_all ** 2 + z_all ** 2) > radius - dr))
    j, x, y, z = j_all[mask], x_all[mask], y_all[mask], z_all[mask]
    j_shell = []

    for vector in vec_scaled:
        distance = (vector[0] - x) ** 2 + (vector[1] - y) ** 2 + (vector[2] - z) ** 2
        j_shell.append(j[distance.argmin()])
    return j_shell