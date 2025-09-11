from numbers import Number

import numpy as np
from astropy import wcs
from astropy.io import fits
from pathlib import Path, PurePath
from tqdm.auto import tqdm

from soxs.instrument_registry import instrument_registry
from soxs.utils import create_region, get_rot_mat, mylog, parse_value

coord_types = {"sky": ("X", "Y", 2, 3), "det": ("DETX", "DETY", 6, 7)}

def make_image(
    evt_file,
    coord_type="sky",
    emin=None,
    emax=None,
    tmin=None,
    tmax=None,
    bands=None,
    expmap_file=None,
    reblock=1,
    zoomin=1
):
    r"""
    Generate an image by binning X-ray counts.

    Parameters
    ----------
    evt_file : string
        The name of the input event file to read.
    coord_type : string, optional
        The type of coordinate to bin into an image.
        Can be "sky" or "det". Default: "sky"
    emin : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The minimum energy of the photons to put in the image, in keV.
    emax : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The maximum energy of the photons to put in the image, in keV.
    tmin : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The minimum energy of the events to be included, in seconds.
        Default is the earliest time available.
    tmax : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The maximum energy of the events to be included, in seconds.
        Default is the latest time available.
    bands : list of tuples, optional
        A list of energy bands to restrict the counts used to make the
        image, in the form of [(emin1, emax1), (emin2, emax2), ...].
        Used as an alternative to emin and emax. Default: None
    expmap_file : string, optional
        Supply an exposure map file to divide this image by
        to get a flux map. Default: None
    reblock : integer, optional
        Change this value to reblock the image to larger
        or small pixel sizes. Only supported for
        sky coordinates. Default: 1
    zoomin : float, optional
        Change this value to crop the image around the center with the
        set fraction of the width/height. Default: 1
    """
    if bands is not None:
        bands = [
            (parse_value(b[0], "keV") * 1000.0, parse_value(b[1], "keV") * 1000.0)
            for b in bands
        ]
    else:
        if emin is None:
            emin = 0.0
        else:
            emin = parse_value(emin, "keV")
        emin *= 1000.0
        if emax is None:
            emax = 100.0
        else:
            emax = parse_value(emax, "keV")
        emax *= 1000.0
    if tmin is None:
        tmin = -np.inf
    else:
        tmin = parse_value(tmin, "s")
    if tmax is None:
        tmax = np.inf
    else:
        tmax = parse_value(tmax, "s")
    if coord_type == "det" and reblock != 1:
        raise RuntimeError(
            "Reblocking images is not supported for detector coordinates!"
        )
    with fits.open(evt_file) as f:
        e = f["EVENTS"].data["ENERGY"]
        t = f["EVENTS"].data["TIME"]
        if bands is not None:
            idxs = False
            for band in bands:
                idxs |= np.logical_and(e > band[0], e < band[1])
        else:
            idxs = np.logical_and(e > emin, e < emax)
        idxs &= np.logical_and(t > tmin, t < tmax)
        xcoord, ycoord, xcol, ycol = coord_types[coord_type]
        x = f["EVENTS"].data[xcoord][idxs]
        y = f["EVENTS"].data[ycoord][idxs]
        exp_time = f["EVENTS"].header["EXPOSURE"]

        xmin_evt = f["EVENTS"].header[f"TLMIN{xcol}"]
        ymin_evt = f["EVENTS"].header[f"TLMIN{ycol}"]
        xmax_evt = f["EVENTS"].header[f"TLMAX{xcol}"]
        ymax_evt = f["EVENTS"].header[f"TLMAX{ycol}"]

        centerx = xmin_evt + (xmax_evt - xmin_evt) / 2
        centery = xmin_evt + (ymax_evt - ymin_evt) / 2

        xmin = xmin_evt + int((centerx - xmin_evt) * (1 - zoomin))
        ymin = ymin_evt + int((centery - ymin_evt) * (1 - zoomin))
        xmax = centerx + int((xmax_evt - centerx) * zoomin)
        ymax = centery + int((ymax_evt - centery) * zoomin)
        
        if coord_type == "sky":
            xctr = f["EVENTS"].header[f"TCRVL{xcol}"]
            yctr = f["EVENTS"].header[f"TCRVL{ycol}"]
            xdel = f["EVENTS"].header[f"TCDLT{xcol}"] * reblock
            ydel = f["EVENTS"].header[f"TCDLT{ycol}"] * reblock

    nx = int(int(xmax - xmin) // reblock)
    ny = int(int(ymax - ymin) // reblock)
    print(xmin, centerx, xmax, xmin_evt, xmax_evt)

    xbins = np.linspace(xmin, xmax, nx + 1, endpoint=True)
    ybins = np.linspace(ymin, ymax, ny + 1, endpoint=True)

    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])

    if expmap_file is not None:
        if coord_type == "det":
            raise RuntimeError(
                "Cannot divide by an exposure map for images "
                "binned in detector coordinates!"
            )
        with fits.open(expmap_file) as f:
            if f["EXPMAP"].shape != (nx, ny):
                raise RuntimeError(
                    "Exposure map and image do not have the same shape!!"
                )
            with np.errstate(invalid="ignore", divide="ignore"):
                H /= f["EXPMAP"].data.T
            H[np.isinf(H)] = 0.0
            H = np.nan_to_num(H)
            H[H < 0.0] = 0.0

    hdu = fits.PrimaryHDU(H.T)

    if coord_type == "sky":
        hdu.header["MTYPE1"] = "EQPOS"
        hdu.header["MFORM1"] = "RA,DEC"
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRVAL1"] = xctr
        hdu.header["CRVAL2"] = yctr
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CUNIT2"] = "deg"
        hdu.header["CDELT1"] = xdel
        hdu.header["CDELT2"] = ydel
        hdu.header["CRPIX1"] = 0.5 * (nx + 1)
        hdu.header["CRPIX2"] = 0.5 * (ny + 1)
    else:
        hdu.header["CUNIT1"] = "pixel"
        hdu.header["CUNIT2"] = "pixel"

    hdu.header["EXPOSURE"] = exp_time
    hdu.name = "IMAGE"
    return hdu

def write_image(
    evt_file,
    out_file,
    coord_type="sky",
    emin=None,
    emax=None,
    tmin=None,
    tmax=None,
    bands=None,
    overwrite=False,
    expmap_file=None,
    reblock=1,    
    zoomin=1
):
    r"""
    Generate a image by binning X-ray counts and write
    it to a FITS file.

    Parameters
    ----------
    evt_file : string
        The name of the input event file to read.
    out_file : string
        The name of the image file to write.
    coord_type : string, optional
        The type of coordinate to bin into an image.
        Can be "sky" or "det". Default: "sky"
    emin : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The minimum energy of the photons to put in the image, in keV.
    emax : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The maximum energy of the photons to put in the image, in keV.
    tmin : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The minimum energy of the events to be included, in seconds.
        Default is the earliest time available.
    tmax : float, (value, unit) tuple, or :class:`~astropy.units.Quantity`, optional
        The maximum energy of the events to be included, in seconds.
        Default is the latest time available.
    bands : list of tuples, optional
        A list of energy bands to restrict the counts used to make the
        image, in the form of [(emin1, emax1), (emin2, emax2), ...].
        Used as an alternative to emin and emax. Default: None
    overwrite : boolean, optional
        Whether to overwrite an existing file with
        the same name. Default: False
    expmap_file : string, optional
        Supply an exposure map file to divide this image by
        to get a flux map. Default: None
    reblock : integer, optional
        Change this value to reblock the image to larger
        or small pixel sizes. Only supported for
        sky coordinates. Default: 1
    zoomin : float, optional
        Change this value to crop the image around the center with the
        set fraction of the width/height. Default: 1
    """
    hdu = make_image(
        evt_file,
        coord_type=coord_type,
        emin=emin,
        emax=emax,
        tmin=tmin,
        tmax=tmax,
        bands=bands,
        expmap_file=expmap_file,
        reblock=reblock, 
        zoomin=zoomin
    )
    hdu.writeto(out_file, overwrite=overwrite)


def plot_image(
    img_file,
    hdu="IMAGE",
    stretch="linear",
    vmin=None,
    vmax=None,
    facecolor="black",
    center=None,
    width=None,
    figsize=(10, 10),
    cmap=None,
    cbar_value='photon-counts',
    background='white'
):
    """
    Plot a FITS image created by SOXS using Matplotlib.

    Parameters
    ----------
    img_file : str
        The on-disk FITS image to plot.
    hdu : str or int, optional
        The image extension to plot. Default is "IMAGE"
    stretch : str, optional
        The stretch to apply to the colorbar scale. Options are "linear",
        "log", and "sqrt". Default: "linear"
    vmin : float, optional
        The minimum value of the colorbar. If not set, it will be the minimum
        value in the image.
    vmax : float, optional
        The maximum value of the colorbar. If not set, it will be the maximum
        value in the image.
    facecolor : str, optional
        The color of zero-valued pixels. Default: "black"
    center : array-like
        A 2-element object giving an (RA, Dec) coordinate for the center
        in degrees. If not set, the reference pixel of the image (usually
        the center) is used.
    width : float, optional
        The width of the image in degrees. If not set, the width of the
        entire image will be used.
    figsize : tuple, optional
        A 2-tuple giving the size of the image in inches, e.g. (12, 15).
        Default: (10,10)
    cmap : str, optional
        The colormap to be used. If not set, the default Matplotlib
        colormap will be used.
    cbar_value: str optional
        The value for a colorbar. Either 'photon-counts' or 'surface-brightness'

    Returns
    -------
    A tuple of the :class:`~matplotlib.figure.Figure` and the
    :class:`~matplotlib.axes.Axes` objects.
    """
    import matplotlib.pyplot as plt
    from astropy.visualization.wcsaxes import WCSAxes
    from astropy.wcs.utils import proj_plane_pixel_scales
    from matplotlib.colors import LogNorm, Normalize, PowerNorm

    if stretch == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif stretch == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif stretch == "sqrt":
        norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
    else:
        raise RuntimeError(f"'{stretch}' is not a valid stretch!")

    if background=='dark':
        params = {"ytick.color" : "w",
              "xtick.color" : "w",
              "axes.labelcolor" : "w",
              "axes.edgecolor" : "w",
              "text.color" : "w",
        #       "figure.facecolor" : '2F366E'}
              "figure.facecolor" : 'none'}
    elif background=='white':
        params = {"ytick.color" : "black",
              "xtick.color" : "black",
              "axes.labelcolor" : "black",
              "axes.edgecolor" : "black",
              "text.color" : "black",
        #       "figure.facecolor" : '2F366E'}
              "figure.facecolor" : 'none'}
    
    plt.rcParams.update(params)
        
    with fits.open(img_file) as f:
        hdu = f[hdu]
        w = wcs.WCS(hdu.header)
        pix_scale = proj_plane_pixel_scales(w)
        if center is None:
            center = w.wcs.crpix
        else:
            center = w.wcs_world2pix(center[0], center[1], 0)
        if width is None:
            dx_pix = 0.5 * hdu.shape[0]
            dy_pix = 0.5 * hdu.shape[1]
        else:
            dx_pix = width / pix_scale[0]
            dy_pix = width / pix_scale[1]
        fig = plt.figure(figsize=figsize)
        ax = WCSAxes(fig, [0.15, 0.1, 0.8, 0.8], wcs=w)
        fig.add_axes(ax)
        #print(pix_scale)
        if cbar_value == 'photon-counts':
            im = ax.imshow(hdu.data, norm=norm, cmap=cmap, extent=(center[0] - 0.5 * dx_pix, center[0] + 0.5 * dx_pix, center[1] - 0.5 * dy_pix, center[1] + 0.5 * dy_pix))
        elif cbar_value == 'surface-brightness':
            im = ax.imshow(hdu.data / pix_scale[0] / pix_scale[1] / 3600**2, norm=norm, cmap=cmap, )        
        ax.set_xlim(center[0] - 0.5 * dx_pix, center[0] + 0.5 * dx_pix)
        ax.set_ylim(center[1] - 0.5 * dy_pix, center[1] + 0.5 * dy_pix)
        ax.set_facecolor(facecolor)
        plt.colorbar(im)
    return fig, ax