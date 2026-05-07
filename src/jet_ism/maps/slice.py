import gadget
from gadget.simulation import Simulation
import h5py
from jet_ism import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from gaepsi2 import color
from .base import _check_showbar, _nice_scalebar_size, add_scalebar, shockmap, velmap, overlay
from ..gas.general import get_temp, calculate_thermal_pressure
from scipy.spatial import cKDTree



def plot_density(ax, fn, fac=1.0, t0=0.0, velocity=True, inj_region=True, vmin=1e-4, vmax=1e3):
    """
    Plot density distribution in a slice with velocity vectors
    Input: ax (matplotlib axis where you want to plot), 
           fn (filename), 
           fac (factor for a box size; default is 1), 
           t0 (absolute time of the first snapshot to calculate relative time of the current snapshot)
    Output: none
    
    """
    sn = Simulation(fn)

    part = h5py.File(fn, 'r')
    numdensity = part['PartType0/Density'][:] * rho_to_numdensity
    fontprop = fm.FontProperties(size=12)

    box = np.array([fac * sn.header.BoxSize,fac * sn.header.BoxSize] )
    center = np.array( [0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize, 0.5 * sn.header.BoxSize] )

    sn.plot_Aslice(numdensity, log=True,  axes=ax, box=box, center=center, vmin=vmin, vmax=vmax, 
        cblabel=r'density [cm$^{-3}$]', cmap='coolwarm_r')
    
    ax.set_title(get_time_title(part, t0))

    #---------
    xx = np.linspace(center[0]-0.5*box[0], center[0]+0.5*box[0], 25)
    yy = np.linspace(center[1]-0.5*box[1], center[1]+0.5*box[1], 25)

    meshgridx, meshgridy = np.meshgrid(xx, yy)

    if inj_region == True:
        circle1 = plt.Circle((1000, 1000), 90, color='whitesmoke', fill=False, linewidth=2)
        circle2 = plt.Circle((1000, 1000), 30, color='whitesmoke', fill=False, linewidth=2, ls='--')
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
    
    if velocity ==True:
        slicex = sn.get_Aslice(sn.part0.Velocities[:,0], res=25, box = box, center = center)
        slicey = sn.get_Aslice(sn.part0.Velocities[:,1], res=25, box = box, center = center)
        ax.quiver(meshgridx, meshgridy, slicex["grid"].T, slicey["grid"].T, zorder=1, color='w', scale_units='xy', scale=10)


        scalebar = AnchoredSizeBar(ax.transData,
                   200, '200 pc', 'lower center', 
                   pad=0.2,
                   color='white',
                   frameon=False,
                   size_vertical=1.5,
                   fontproperties=fontprop,
                   sep=3)
    ax.add_artist(scalebar)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.close()


class slice_snapshot:
    """
    Slice snapshot class. Produces nearest-neighbor slice maps via
    inspector_gadget's get_Aslice (calcGrid.calcASlice C extension), rendered
    through jet_ism's gaepsi2 colormap stack so they compose with overlap_*
    projection maps from onepanel.
    Input: fn (filename, including directory)
    Example: snap = slice_snapshot(output_directory + 'snap_020.hdf5')
    """
    def __init__(self, fn):
        part = h5py.File(fn, 'r')
        self.fn = fn
        try:
            self.UnitLength = part['Header'].attrs['UnitLength_in_cm']
            self.UnitVelocity = part['Header'].attrs['UnitVelocity_in_cm_per_s']
            self.UnitMass = part['Header'].attrs['UnitMass_in_g']
        except KeyError:
            self.UnitLength = unit_length
            self.UnitVelocity = unit_velocity
            self.UnitMass = unit_mass

        self.UnitTime = self.UnitLength / self.UnitVelocity
        self.UnitDensity = self.UnitMass / (self.UnitLength ** 3)
        self.rho_to_numdensity = 1. * self.UnitDensity / (mu * PROTONMASS)
        self.redshift = float(part['Header'].attrs['Redshift'])
        if abs(self.redshift) < 1e-6:
            self.time = float(part['Header'].attrs['Time']) * self.UnitTime / megayear
        else:
            self.time = 0.0

        BoxSize = part['Header'].attrs['BoxSize']
        self.BoxSize = BoxSize
        self.center = np.array([0.5 * BoxSize, 0.5 * BoxSize, 0.5 * BoxSize])

    def _time_title(self, t0=0):
        if abs(self.redshift) < 1e-6:
            return r'$t=$%.2f Myr' % (self.time - t0)
        else:
            return r'$z=$%.4f' % self.redshift

    def _slice_field(self, field_values, lbox, imsize=2000, orientation='xy',
                     center=None, z_offset=0.0):
        """
        Compute a 2D nearest-neighbor slice of a per-PartType0 field via
        inspector_gadget Simulation.get_Aslice. Returns a (imsize, imsize) ndarray.
        """
        sn = Simulation(self.fn)
        if center is None:
            slice_center = self.center.copy()
        else:
            slice_center = np.asarray(center, dtype=float).copy()
        axis_map = {'xy': [0, 1], 'xz': [0, 2], 'yz': [1, 2]}
        axes = axis_map[orientation]
        perp = 3 - axes[0] - axes[1]
        slice_center[perp] += z_offset
        box = np.array([lbox, lbox])
        result = sn.get_Aslice(field_values, res=imsize, box=box,
                               center=slice_center, axis=axes)
        return result['grid']

    def slice_temp(self, lbox, z_offset=0, imsize=2000, orientation='xy',
                   showbar='bottom', show=True, savefig_file=None, t0=0,
                   tmin=2.5, tmax=7.0, vmin=-5, vmax=0, center=None,
                   scalebar_size=None, scalebar_label=None):
        """
        plot temperature slice (true nearest-neighbor) on density+temp CoolWarm map

        Parameters
        ----------
        lbox: box length
        z_offset: offset along the perpendicular axis from center, default 0
        imsize: Npixel of the image
        orientation: projection of 'xy','yz','xz'
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        tmin, tmax: color range for the temperature in log10 K, default (2.5, 7.0)
        vmin, vmax: color range for the numdensity in log10 cm^-3, default (-5, 0)
        center: center of the box, default None (box center)
        """
        _check_showbar(showbar)
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        part = h5py.File(self.fn, 'r')
        gast = get_temp(part, GAMMA)
        gasn = part['PartType0/Density'][:] * self.rho_to_numdensity

        temp_grid = self._slice_field(gast, lbox, imsize=imsize, orientation=orientation,
                                      center=center, z_offset=z_offset)
        n_grid = self._slice_field(gasn, lbox, imsize=imsize, orientation=orientation,
                                   center=center, z_offset=z_offset)

        img1 = color.CoolWarm(color.NL(temp_grid, range=(tmin, tmax)),
                              color.NL(n_grid, range=(vmin, vmax)))

        density_array = 10 ** np.linspace(vmin, vmax, 100)
        temperature_array = 10 ** np.linspace(tmin, tmax, 100)
        Density_cmap, Temperature_cmap = np.meshgrid(density_array, temperature_array, indexing='ij')
        cmap_density_temperature = color.CoolWarm(color.NL(Temperature_cmap, range=(tmin, tmax)),
                                                  color.NL(Density_cmap, range=(vmin, vmax)))

        if showbar == 'bottom':
            f, axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cmap_density_temperature, extent=[tmin, tmax, vmax, vmin], aspect=0.32)
            axes[1].set_xlabel(r'log $T$ [K]', labelpad=2)
            axes[1].set_ylabel(r'log $n$ [cm$^{-3}$]', labelpad=2)
        elif showbar == 'right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cmap_density_temperature, axes=(1, 0, 2)),
                           extent=[vmin, vmax, tmax, tmin], aspect=4.1)
            axes[1].set_ylabel('log T [K]', labelpad=4)
            axes[1].set_xlabel(r'log n [cm$^{-3}$]', labelpad=2)
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar == 'none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar == 'blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        return f

    def slice_pressure(self, lbox, z_offset=0, imsize=2000, orientation='xy',
                       showbar='bottom', show=True, savefig_file=None, t0=0,
                       pmin=-15, pmax=-9, vmin=-5, vmax=0, center=None,
                       scalebar_size=None, scalebar_label=None):
        """
        plot thermal pressure slice (log scale, cgs dyn/cm^2) on shockmap

        Parameters
        ----------
        lbox: box length
        z_offset: offset along the perpendicular axis from center, default 0
        imsize: Npixel of the image
        orientation: projection of 'xy','yz','xz'
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        pmin, pmax: color range for pressure in log10 dyn/cm^2, default (-15, -9)
        vmin, vmax: log range for the colorbar density axis (cosmetic), default (-5, 0)
        center: center of the box, default None (box center)
        """
        _check_showbar(showbar)
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        part = h5py.File(self.fn, 'r')
        gasp = calculate_thermal_pressure(part)

        p_grid = self._slice_field(gasp, lbox, imsize=imsize, orientation=orientation,
                                   center=center, z_offset=z_offset)

        cn0 = np.ones((imsize, imsize)) * 1e-6
        p_low, p_up = 10 ** pmin, 10 ** pmax

        img2 = shockmap(color.NL(p_grid, range=(p_low, p_up)),
                        color.NL(cn0, range=(-7, -6)))

        pressure_array = 10 ** np.linspace(pmin, pmax, 100)
        density_array = 10 ** np.linspace(vmin, vmax, 100)
        Density_cmap, P_cmap = np.meshgrid(density_array, pressure_array, indexing='ij')
        cbar_p = shockmap(color.NL(P_cmap, range=(p_low, p_up)),
                          color.NL(np.ones_like(Density_cmap) * 1e-6, range=(-7, -6)))

        if showbar == 'bottom':
            f, axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cbar_p, extent=[pmin, pmax, pmin, pmax], aspect=0.055)
            axes[1].set_xlabel(r'log $P$ [dyn cm$^{-2}$]', labelpad=2)
            axes[1].set_yticks([])
        elif showbar == 'right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cbar_p, axes=(1, 0, 2)),
                           extent=[pmin, pmax, pmax, pmin], aspect='auto')
            axes[1].set_ylabel(r'log $P$ [dyn cm$^{-2}$]', labelpad=4)
            axes[1].set_xticks([])
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar == 'none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar == 'blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        return f

    def slice_sigma_velocity(self, lbox, z_offset=0, imsize=500, orientation='xy',
                             showbar='bottom', show=True, savefig_file=None, t0=0,
                             range=(0, 300), vmin=-5, vmax=0, k=32,
                             weight='density_square', center=None,
                             scalebar_size=None, scalebar_label=None):
        """
        plot line-of-sight velocity dispersion in a slice plane (km/s, linear).

        Unlike overlap_sigma_velocity (column second-moment), this queries the
        k nearest particles in 3D to each pixel center on the slice plane and
        computes the weighted std of v_los over those neighbors. This is a
        true volumetric statistic at the slice plane rather than a column
        statistic, so it differs from the projection version semantically.

        Note: imsize defaults to 500 here (vs 2000 for slice_temp / slice_pressure)
        because the per-pixel KDTree query is the bottleneck; raise it if you
        have time and need finer resolution.

        Parameters
        ----------
        lbox: box length
        z_offset: offset along the perpendicular axis from center, default 0
        imsize: Npixel of the image, default 500
        orientation: projection of 'xy','yz','xz'
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        range: linear color range for sigma_v in km/s, default (0, 300)
        vmin, vmax: log range for the colorbar density axis (cosmetic), default (-5, 0)
        k: number of nearest neighbors used per pixel, default 32
        weight: 'mass' or 'density_square', default 'density_square'
        center: center of the box, default None (box center)
        """
        _check_showbar(showbar)
        if center is None:
            center = self.center
        else:
            center = np.asarray(center, dtype=float)
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        part = h5py.File(self.fn, 'r')
        pos = part['PartType0/Coordinates'][:]
        vel = part['PartType0/Velocities'][:]
        dens = part['PartType0/Density'][:]

        axis_map = {'xy': [0, 1], 'xz': [0, 2], 'yz': [1, 2]}
        axes_idx = axis_map[orientation]
        perp = 3 - axes_idx[0] - axes_idx[1]

        v_los = vel[:, perp] * unit_velocity / 1e5

        if weight == 'mass':
            w = part['PartType0/Masses'][:]
        elif weight == 'density_square':
            w = dens * dens
        else:
            raise NotImplementedError(f"weight {weight!r} not supported")

        cx = center[axes_idx[0]]
        cy = center[axes_idx[1]]
        cz = center[perp] + z_offset

        pix_x = np.linspace(cx - 0.5 * lbox, cx + 0.5 * lbox, imsize)
        pix_y = np.linspace(cy - 0.5 * lbox, cy + 0.5 * lbox, imsize)
        XX, YY = np.meshgrid(pix_x, pix_y, indexing='xy')
        qpts = np.empty((imsize * imsize, 3))
        qpts[:, axes_idx[0]] = XX.ravel()
        qpts[:, axes_idx[1]] = YY.ravel()
        qpts[:, perp] = cz

        tree = cKDTree(pos)
        _, idx = tree.query(qpts, k=k)

        v_neigh = v_los[idx]
        w_neigh = w[idx]
        sumw = w_neigh.sum(axis=1)
        mean_v = (w_neigh * v_neigh).sum(axis=1) / sumw
        var_v = (w_neigh * (v_neigh - mean_v[:, None]) ** 2).sum(axis=1) / sumw
        sigma_v = np.sqrt(np.maximum(var_v, 0))

        sigma_grid = sigma_v.reshape(imsize, imsize)

        sv_low, sv_up = range[0], range[1]
        cn0 = np.ones((imsize, imsize)) * 1e-6
        img2 = velmap(color.N(sigma_grid, range=(sv_low, sv_up)),
                      color.NL(cn0, range=(-7, -6)))

        density_array = 10 ** np.linspace(vmin, vmax, 100)
        sv_array = np.linspace(sv_low, sv_up, 100)
        Density_cmap, SV_cmap = np.meshgrid(density_array, sv_array, indexing='ij')
        cbar_sv = velmap(color.N(SV_cmap, range=(sv_low, sv_up)),
                         color.NL(np.ones_like(Density_cmap) * 1e-6, range=(-7, -6)))

        if showbar == 'bottom':
            f, axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cbar_sv, extent=[sv_low, sv_up, sv_low, sv_up], aspect=0.055)
            axes[1].set_xlabel(r'$\sigma_v$ [km s$^{-1}$]', labelpad=2)
            axes[1].set_yticks([])
        elif showbar == 'right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cbar_sv, axes=(1, 0, 2)),
                           extent=[sv_low, sv_up, sv_up, sv_low], aspect='auto')
            axes[1].set_ylabel(r'$\sigma_v$ [km s$^{-1}$]', labelpad=4)
            axes[1].set_xticks([])
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar == 'none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar == 'blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        return f