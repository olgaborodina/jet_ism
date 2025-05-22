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

import matplotlib.pyplot as plt    ## plot stuff
from matplotlib import animation
import matplotlib as mpl

from matplotlib.colors import ListedColormap,colorConverter
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera

""" customize color maps """

class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)
    
def make_colormap(cmap_name):
    y = np.linspace(0,1,100)
    COL = MplColorHelper(cmap_name, 0, 1)
    x = COL.get_rgb(y)
    return color.Colormap(x[:,0:3])

def weighted_std(values, weights=None):
    """
    Return the weighted standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


jetmap = make_colormap('cubehelix')
shockmap = make_colormap('autumn')
sfrmap = make_colormap('rainbow')
velmap = make_colormap('coolwarm')


FloatType = np.float64
IntType = np.int32



""" snippets """

def overlap(img1,img2,thr=0):
    """
    overlap img2 onto img1, img2 is transparent in the region of black
    """
    assert (img1.shape == img2.shape)
    
    c2 = np.sum(img2[:,:,0:3],axis=-1)
    img1_ = img1.copy()
    img2_ = img2.copy()
    img1_[c2>thr] = np.array([0,0,0,0])
    img2_[c2<=thr] = np.array([0,0,0,0])
    
    return (img1_ + img2_)

def blend_channel(colorRGBA1, colorRGBA2):
    """
    blend RGB channels by weight of alpha
    """
    assert (colorRGBA1.shape == colorRGBA2.shape)
    
    result = np.zeros_like(colorRGBA1)  
    result[:,:,3] = 255 - ((255 - colorRGBA1[:,:,3]) * (255 - colorRGBA2[:,:,3]) / 255) # alpha
#     print ("alpha:",np.average(result[:,:,3]))
    
    w1 = (255 - colorRGBA2[:,:,3])/255
    w2 = (colorRGBA2[:,:,3])/255
    result[:,:,0] = np.multiply(colorRGBA1[:,:,0],w1)  + np.multiply(colorRGBA2[:,:,0],w2)
    result[:,:,1] = np.multiply(colorRGBA1[:,:,1],w1)  + np.multiply(colorRGBA2[:,:,1],w2)
    result[:,:,2] = np.multiply(colorRGBA1[:,:,2],w1)  + np.multiply(colorRGBA2[:,:,2],w2)
    
    return result

def overlay(img1,img2,x):
    """
    x is the weight of the img2 pixel, will squash x by sigmoid, as the alpha of img2
    """
    c = np.uint(255*(1/(1+np.exp(-1*x)) ** 0.5))
    img1_ = img1.copy()
    img2_ = img2.copy()

    img2_[:,:,3] = 0
    img2_[:,:,3] += np.uint8(c)

    img3 = blend_channel(img1_,img2_)
    
    return img3

def cut(part,center,lbox,slab_width,orientation):
    """
    take the entire snapshot and cut a slab from specified projection
    
    Parameters
    ----------
    part: h5py.File(file,'r')['PartType0']
    center: numpy array, center of the chunk
    lbox: box length
    slab_width: projection depth
    orientation: projection of 'xy','yz','xz'
    
    Returns
    -------
    mask and relative position to the center
    """
    ppos = np.float32(part['Coordinates'][:]) # non-periodic
    ppos -= center
    
    if orientation == 'xy':
        d1,d2,d3 = 0,1,2
    if orientation == 'yz':
        d1,d2,d3 = 1,2,0
    if orientation == 'xz':
        d1,d2,d3 = 0,2,1
    
    mask = np.abs(ppos[:,d1]) < 0.5*lbox
    mask &= np.abs(ppos[:,d2]) < 0.5*lbox
    mask &= np.abs(ppos[:,d3]) < 0.5*slab_width

    return mask,ppos[mask]

def pos2device(ppos,lbox,slab_width,imsize,orientation):
    """
    transfer cartesian position to device coordinates
    
    Parameters
    ----------
    ppos: (N,3), particle position centered at [0,0,0]
    lbox: box length
    slab_width: projection depth
    orientation: projection of 'xy','yz','xz'
    imsize: Npixel of the image
    
    Returns
    particle position in device coordinates
    """    
    lgt = 0.5*lbox
    mpers = camera.ortho(-slab_width,slab_width,(-lgt,lgt,-lgt,lgt))  # (near,far,(left,right,top,bottom)),return 4*4 projectionmatrix
    
    if orientation == 'xy':
        mmv = camera.lookat((0,0,-slab_width),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
    if orientation == 'yz':
        mmv = camera.lookat((slab_width,0,0),(0,0,0),(0,1,0)) # (position of camera, focal point, up direction)
    if orientation == 'xz':
        mmv = camera.lookat((0,slab_width,0),(0,0,0),(1,0,0)) # (position of camera, focal point, up direction)
        
    pos2d = camera.apply(camera.matrix(mpers,mmv),ppos) # apply a camera matrix to data coordinates, return position in clip coordinate
    posdev = camera.todevice(pos2d, extent=(imsize, imsize)) # Convert clipping coordinate to device coordinate
    
    return posdev


