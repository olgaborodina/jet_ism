from .. import (unit_velocity, unit_length, unit_mass, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, get_time_title, rho_to_numdensity, megayear)
from .base import *
from .base import _nice_scalebar_size, _check_showbar, SHOWBAR_OPTIONS
from ..gas.general import *
from .get_channel import *

import matplotlib.font_manager as fm
from matplotlib.patches import Circle


class snapshot:
    """
    Snapshot class to read in the snapshot and plot maps
    Input: fn (filename, including directory)
    Example: snap = snapshot(output_directory + 'snap_020.hdf5')
    """
    def __init__(self,fn):
        part = h5py.File(fn,'r')
        self.fn = fn
        try:
            self.UnitLength = part['Header'].attrs['UnitLength_in_cm']
            self.UnitVelocity = part['Header'].attrs['UnitVelocity_in_cm_per_s']
            self.UnitMass = part['Header'].attrs['UnitMass_in_g']
        except:
            self.UnitLength = unit_length # kpc
            self.UnitVelocity = unit_velocity # km/s
            self.UnitMass = unit_mass # 1e10Msun

        self.UnitTime = self.UnitLength/self.UnitVelocity
        self.UnitDensity = self.UnitMass / (self.UnitLength**3)
        self.UnitEnergy = self.UnitMass * (self.UnitLength**2) / (self.UnitTime**2)
        # self.temp_to_u = (BOLTZMANN/PROTONMASS)/mu/(GAMMA-1)/self.UnitVelocity/self.UnitVelocity
        self.rho_to_numdensity = 1.*self.UnitDensity/(mu*PROTONMASS)
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

    def overlap_jet(self, lbox, slab_width=None, imsize=2000, orientation='xy',
                    show=True, range=(-5,0), vmin=None, vmax=None, center=None,
                    t0=0, scalebar_size=None, scalebar_label=None, savefig_file=None):
        """
        plot jet map overlaid on top of gas density + temp map

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        range: color range for the jet map in log scale, default (-5,0)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize,imsize)) * 1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize)) * 1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
        
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0] > 0]
        vmin0, vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))
        
        # jet tracer
        channels = get_channel_jet_tracer(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                          imsize=imsize, jetcolumn=-1, orientation=orientation)
        img2 = jetmap(color.NL(channels[1], range=range), color.NL(cn0, range=(-7, -6)))
        
        #---------
        f,ax = plt.subplots(1,1,figsize=(5, 5))
        scale = color.NL(channels[1], range=range)
        img = overlay(img1, img2, 2 * np.nan_to_num(scale))
        
        ax.imshow(img, extent=[0, lbox, 0, lbox],origin='lower')
        ax.set_xlabel('pc', fontsize=15)
        ax.set_title(self._time_title(t0), fontsize=18)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f

    def overlap_shock(self, lbox, slab_width=None, imsize=2000, orientation='xy',
                        show=True, range=(-3.5, -1.8), vmin=None, vmax=None, center=None,
                        t0=0, scalebar_size=None, scalebar_label=None, savefig_file=None):
        """
        plot shock map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        range: color range for the shock map in log scale, default (-3.5, -1.8)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize, imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize, imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation)
        x = channels[0][channels[0] > 0]
        vmin0, vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0

        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))

        # shock
        channels = get_channel_shock(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                          imsize=imsize, shockcolumn=-1, orientation=orientation)
        img2 = shockmap(color.NL(channels[0], range=range), color.NL(cn0, range=(-7, -6)))

        print(channels[0].min(), channels[0].max())
        #---------
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        scale = color.NL(channels[0], range=range)
        img = overlay(img1, img2, 2 * np.nan_to_num(scale))

        ax.imshow(img,extent=[0, lbox, 0, lbox], origin='lower')
        ax.set_xlabel('pc', fontsize=15)
        ax.set_title(self._time_title(t0), fontsize=18)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f


    def overlap_soundspeed(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                            showbar='bottom', range=(4, 5.2), vmin=-2, vmax=2, center=None, savefig_file=None, t0=0,
                            scalebar_size=None, scalebar_label=None):
        """
        plot sound speed map overlaid on top of gas density+temp map

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        range: color range for the sound speed map in log scale, default (4, 5.2)
        vmin, vmax: color range for the density map in log scale, default (-2, 2)
        center: center of the box, default None (box center)
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        """
        _check_showbar(showbar)
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'
        fontprop = fm.FontProperties(size=12)

        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
        ss_low, ss_up = 10 ** range[0], 10 ** range[1] # cm/s
        
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation)
        x = channels[0][channels[0] > 0]
        vmin0, vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0

        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))
        
        # sound speed
        channels = get_channel_soundspeed(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                          imsize=imsize, sscolumn=-1, orientation=orientation)
        img2 = shockmap(color.N(channels[0], range=(ss_low, ss_up)), color.NL(cn0, range=(-7, -6)))

        #---------

        scale = color.N(channels[0], range=(ss_low, ss_up))
        img = overlay(img1, img2, 2 * np.nan_to_num(scale))

        # create cmaps
        density_array = 10 ** np.linspace(vmin, vmax, 100)
        temperature_array = 10 ** np.linspace(t_low, t_up, 100)
        ss_array = np.linspace(ss_low, ss_up, 100)
        Density_cmap, Temperature_cmap = np.meshgrid(density_array, temperature_array, indexing='ij')
        Density_cmap, ss_cmap = np.meshgrid(density_array, ss_array, indexing='ij')
        
        cmap_density_temperature = color.CoolWarm(color.NL(Temperature_cmap, range=(t_low, t_up)), color.NL(Density_cmap, range=(vmin, vmax)))

        cmap_density = color.CoolWarm(color.NL(np.ones_like(Temperature_cmap) * 10 ** np.mean([t_low, t_up]),range=(t_low, t_up)),color.NL(Density_cmap.T, range=(vmin,vmax)))
        cbar_ss = shockmap(color.N(ss_cmap, range=(ss_low, ss_up)), color.NL(np.ones_like(Density_cmap) * 1e-6, range=(-7, -6)))
    
        if showbar=='bottom':
            f, axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cbar_ss, extent=[ss_low, ss_up, ss_low, ss_up], aspect=0.055)
            axes[1].set_xlabel(r'sound speed cm/s', labelpad=2)
            axes[1].set_yticks([])
            axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        elif showbar=='right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cbar_ss, axes=(1, 0, 2)),
                           extent=[ss_low, ss_up, ss_up, ss_low], aspect='auto')
            axes[1].set_ylabel(r'sound speed cm/s', labelpad=4)
            axes[1].set_xticks([])
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
            axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        elif showbar=='none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar=='blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f


    def overlap_pressure(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                         showbar='bottom', pmin=-15, pmax=-9, vmin=-2, vmax=2, center=None,
                         savefig_file=None, t0=0, scalebar_size=None, scalebar_label=None):
        """
        plot mass-weighted thermal pressure map (log scale, cgs dyn/cm^2)

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        pmin, pmax: color range for the pressure map in log10 dyn/cm^2, default (-15, -9)
        vmin, vmax: color range for the density map in log scale, default (-2, 2)
        center: center of the box, default None (box center)
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        """
        _check_showbar(showbar)
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize, imsize)) * 1e-6  # dens placeholder
        cn1 = np.ones((imsize, imsize)) * 1e4   # temp placeholder
        t_low, t_up = 2.0, 7.0
        p_low, p_up = 10 ** pmin, 10 ** pmax  # dyn/cm^2

        # mass-weighted pressure
        channels = get_channel_pressure(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                        imsize=imsize, orientation=orientation)
        img2 = shockmap(color.NL(channels[1], range=(p_low, p_up)), color.NL(cn0, range=(-7, -6)))

        # cmap for colorbar
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
        if show == True:
            plt.show()
        else:
            plt.close()
        return f


    def overlap_temp(self, lbox, slab_width=None, imsize=2000, orientation='xy', showbar='bottom',
                    show=True, savefig_file=None, t0=0, tmin=2.5, tmax=7.0, vmin=-2, vmax=2, center=None,
                    scalebar_size=None, scalebar_label=None):
        """
        plot jet map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        imsize: Npixel of the image
        orientation: projection of 'xy','yz','xz'
        showbar: whether to show the colorbar, bottom, right, none, or blank
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        tmin, tmax: color range for the temperature map in log scale, default (2.5, 7.0)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        """
        _check_showbar(showbar)
        BoxSize = self.BoxSize
        if center==None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize,imsize)) * 1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize)) * 1e4 # temp placeholder
        t_low, t_up = tmin, tmax

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0,vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(channels[1], range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))

        #-------- create cmaps:
        density_array = 10 ** np.linspace(vmin, vmax, 100)
        temperature_array = 10 ** np.linspace(t_low, t_up, 100)
        Density_cmap, Temperature_cmap = np.meshgrid(density_array, temperature_array, indexing='ij')
        
        cmap_density_temperature = color.CoolWarm(color.NL(Temperature_cmap,range=(t_low, t_up)), color.NL(Density_cmap, range=(vmin, vmax)))

        #---------
        if showbar=='bottom':
            f,axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cmap_density_temperature, extent=[t_low, t_up, vmax, vmin], aspect=0.32)
            axes[1].set_xlabel(r'log $T$ [K]', labelpad=2)
            axes[1].set_ylabel(r'$\log{\rho}$ ' + '\n' + r' [$M_\odot$ pc$^{-2}$])', labelpad=2)
        elif showbar=='right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cmap_density_temperature, axes=(1, 0, 2)), extent=[vmin, vmax, t_up, t_low], aspect=4.1)
            axes[1].set_ylabel('log (temperature [K])', labelpad=4)
            axes[1].set_xlabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar=='none':
            f,ax = plt.subplots(1, 1, figsize=(5,5))
            ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar=='blank':
            f,ax = plt.subplots(1, 1, figsize=(5,5))
            ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f

    def overlap_sigma_velocity(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                                center=None, t0=0, scalebar_size=None, scalebar_label=None,
                                savefig_file=None):
        """
        TODO: NOT TESTED
        plot velocity-dispersion map overlaid on top of gas density+temp map

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        center: center of the box, default None (box center)
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0,vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        img1 = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(channels[0], range=(vmin0,vmax0)))

        # velocity dispersion
        channels = get_channel_sigma_velocity(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                          imsize=imsize,orientation=orientation)
        img2 = velmap(color.N(channels[1],range=(50, 300)),color.NL(cn0, range=(-7,-6)))

        print(channels[1].min(), channels[1].max())
        #---------
        f,ax = plt.subplots(1,1,figsize=(5,5))
        scale = color.N(channels[1],range=(50, 300))
        img = overlay(img1,img2,2*np.nan_to_num(scale))

        ax.imshow(img,extent=[0,lbox,0,lbox],origin='lower')
        ax.set_xlabel('pc', fontsize=15)
        ax.set_title(self._time_title(t0), fontsize=18)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f

    
    def overlap_sfr(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                    range=(-5, 0), vmin=None, vmax=None, weight='sfr', center=None,
                    t0=0, scalebar_size=None, scalebar_label=None, savefig_file=None):
        """
        plot SFR map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        range: color range for the SFR map in log scale, default (-5,0)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        weight: weighting for the SFR map, 'mass' or 'sfr', default 'sfr'
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize,imsize)) * 1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize)) * 1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0, vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))

        # SFR
        channels = get_channel_sfr(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                          imsize=imsize, sfrcolumn=-1, orientation=orientation, weight=weight)
        img2 = sfrmap(color.NL(channels[1], range=range), color.NL(cn0, range=(-7, -6)))

        #---------
        f,ax = plt.subplots(1, 1, figsize=(5,5))
        scale = color.NL(channels[1], range=range)
        img = overlay(img1, img2, 2 * np.nan_to_num(scale))

        ax.imshow(img, extent=[0, lbox, 0, lbox],origin='lower')
        ax.set_xlabel('pc', fontsize=15)
        ax.set_title(self._time_title(t0), fontsize=18)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f


    def overlap_random(self, mask_random, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                        range=(-5,0), vmin=-2, vmax=2, center=None,
                        t0=0, scalebar_size=None, scalebar_label=None, savefig_file=None):
        """
        plot map of a random mask overlaid on top of gas density+temp map
        
        Parameters
        ----------
        mask_random: boolean array of the random mask, shape (number of gas particles in snapshot file)
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        range: color range for the random map in log scale, default (-5,0)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize, imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize, imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation)
        x = channels[0][channels[0] > 0]
        vmin0, vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))

        # random field
        channels = get_channel_random(self.fn, mask_random, center=center, lbox=lbox, slab_width=slab_width,
                                          imsize=imsize, orientation=orientation)
        img2 = sfrmap(color.NL(channels[1], range=range), color.NL(cn0, range=(-7, -6)))

        #---------
        f,ax = plt.subplots(1, 1, figsize=(5, 5))
        scale = color.NL(channels[1], range=range)
        img = overlay(img1, img2, 2 * np.nan_to_num(scale))

        ax.imshow(img,extent=[0, lbox, 0, lbox],origin='lower')
        ax.set_xlabel('pc', fontsize=15)
        ax.set_title(self._time_title(t0), fontsize=18)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f



    def overlap_temp_stars(self, lbox, slab_width=None, imsize=2000, orientation='xy', showbar='bottom', show=True, savefig_file=None, t0=0,
                            tmin=2.5, tmax=7.0, vmin=-2, vmax=2, center=None, age_cut=False,
                            scalebar_size=None, scalebar_label=None):
        """
        plot density+temperature map with star particles overlaid

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        tmin, tmax: color range for the temperature map in log scale, default (2.5, 7.0)
        vmin, vmax: color range for the density map in log scale, default (-2, 2)
        center: center of the box, default None (box center)
        age_cut: whether to apply age cut (1 Myr) for the star particles, boolean, default False
        """
        _check_showbar(showbar)
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize, imsize)) * 1e-6 # dens placeholder
        cn1 = np.ones((imsize, imsize)) * 1e4 # temp placeholder
        t_low, t_up = tmin, tmax

        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation)
        x = channels[0][channels[0] > 0]
        vmin0,vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(channels[1], range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))

        stars_coords, stars_mass = get_channel_stars(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize, orientation=orientation, age_cut=age_cut)
        coord_shift = np.array(center) - 0.5 * lbox
        
        #-------- create cmaps:
        density_array = 10 ** np.linspace(vmin, vmax, 100)
        temperature_array = 10 ** np.linspace(t_low, t_up, 100)
        Density_cmap, Temperature_cmap = np.meshgrid(density_array, temperature_array, indexing='ij')
        
        cmap_density_temperature = color.CoolWarm(color.NL(Temperature_cmap, range=(t_low, t_up)), color.NL(Density_cmap, range=(vmin, vmax)))

        #---------
        fontprop = fm.FontProperties(size=12)

        def _plot_stars(target_ax):
            target_ax.imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            if len(stars_mass) > 0:
                target_ax.scatter(stars_coords[0] - coord_shift[0], stars_coords[1] - coord_shift[1],
                                  c='yellow', s=25 * (stars_mass / stars_mass.max()) ** 2, marker='*')

        if showbar=='bottom':
            f, axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            _plot_stars(axes[0])
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cmap_density_temperature, extent=[t_low, t_up, vmax, vmin], aspect=0.32)
            axes[1].set_xlabel(r'log $T$ [K]', labelpad=2)
            axes[1].set_ylabel(r'$\log{\rho}$ ' + '\n' + r' [$M_\odot$ pc$^{-2}$])', labelpad=2)
        elif showbar=='right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            _plot_stars(axes[0])
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cmap_density_temperature, axes=(1, 0, 2)),
                           extent=[vmin, vmax, t_up, t_low], aspect=4.1)
            axes[1].set_ylabel('log (temperature [K])', labelpad=4)
            axes[1].set_xlabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar=='none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            _plot_stars(ax)
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar=='blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            _plot_stars(ax)
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f

    def overlap_coolingtime(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True, showbar='bottom',
                    range=(-1, 5), vmin=None, vmax=None, weight='none', t0=0, postprocessing=False,
                    center=None, savefig_file=None,
                    scalebar_size=None, scalebar_label=None):
        """
        plot cooling-time map overlaid on top of gas density+temp map

        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        showbar: colorbar placement, one of 'bottom', 'right', 'none', 'blank'
        range: color range for the cooling time in log scale, default (-1, 5)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        weight: weighting for the cooling time map, default 'none'
        t0: time offset for the title in Myr, default 0
        center: center of the box, default None (box center)
        savefig_file: filename to save the figure, default None (not saving)
        """
        _check_showbar(showbar)
        BoxSize = self.BoxSize
        if center is None:
            center = self.center
        if scalebar_size is None:
            scalebar_size = _nice_scalebar_size(lbox)
        if scalebar_label is None:
            scalebar_label = f'{scalebar_size:g}'

        cn0 = np.ones((imsize,imsize)) * 1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize)) * 1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K

        # cooling time
        channels = get_channel_coolingtime(self.fn, center=center, lbox=lbox, slab_width=slab_width,
                                   imsize=imsize, orientation=orientation, postprocessing=postprocessing,  weight=weight,)
        x = channels[0][channels[0]>0]
        vmin0, vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
        img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))


        img2 = jetmap(color.NL(channels[1], range=range), color.NL(cn0, range=(-7, -6)))

        cooltime_array = 10 ** np.linspace(range[0], range[1], 100)
        density_array = 10 ** np.linspace(vmin,vmax, 100)
        Density_cmap, Cooltime_cmap = np.meshgrid(density_array, cooltime_array, indexing='ij')
        cmap_cooltime = jetmap(color.NL(Cooltime_cmap, range=range),color.NL(np.ones_like(Density_cmap) * 1e-6, range=(-7,-6)))

        if showbar=='bottom':
            f,axes = plt.subplots(2, 1, height_ratios=[0.93, 0.1], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.1)
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(cmap_cooltime, extent=[range[0], range[1], range[0], range[1]], aspect=0.055)
            axes[1].set_xlabel(r'$\log{t_\text{cool}}$', labelpad=2, fontsize=13)
            axes[1].set_yticklabels([])
            axes[1].set_yticks([])
        elif showbar=='right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(self._time_title(t0), fontsize=18)
            add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
            axes[0].set_xticks([]); axes[0].set_yticks([])

            axes[1].imshow(np.transpose(cmap_cooltime, axes=(1, 0, 2)),
                           extent=[range[0], range[1], range[1], range[0]], aspect='auto')
            axes[1].set_ylabel(r'$\log{t_\text{cool}}$', labelpad=4, fontsize=13)
            axes[1].set_xticks([])
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        elif showbar=='none':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(self._time_title(t0), fontsize=18)
            add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
            ax.set_xticks([]); ax.set_yticks([])
        elif showbar=='blank':
            f, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([]); ax.set_yticks([])

        if savefig_file is not None:
            plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close()
        return f


def overlap_jet_contour(snapshot_turb, snapshot_jet, lbox, slab_width=None, imsize=2000, orientation='xy', showbar='bottom', show_BH=False,
                    show=True, vmin=None, vmax=None, center=None, savefig_file=None, t0=15, range_jet=(1e-4,2e-4), levels=1,
                        contour_color='lightpink', scalebar_size=None, scalebar_label=None):
    _check_showbar(showbar)
    assert abs(snapshot_jet.time - snapshot_turb.time) < 0.03
    if snapshot_turb.BoxSize == snapshot_jet.BoxSize:
        BoxSize = snapshot_turb.BoxSize
    else:
        raise('box sizes are not the same')
    
    if center == None:
        if snapshot_turb.center.all() == snapshot_jet.center.all():
            center = snapshot_turb.center
        else:
            raise('center is not the same for the snapshots, specify it as an argument')

    if scalebar_size is None:
        scalebar_size = _nice_scalebar_size(lbox)
    if scalebar_label is None:
        scalebar_label = f'{scalebar_size:g}'

    cn0 = np.ones((imsize,imsize)) * 1e-6 # dens placeholder
    cn1 = np.ones((imsize,imsize)) * 1e4 # temp placeholder
    t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
    
    # gas rho temp
    channels = get_channel_gas_rhotemp(snapshot_turb.fn,center=center,lbox=lbox,slab_width=slab_width,
                               imsize=imsize,orientation=orientation)
    x = channels[0][channels[0] > 0]
    vmin0, vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
    if vmin == None:
        vmin = vmin0
    if vmax == None:
        vmax = vmax0
    img1 = color.CoolWarm(color.NL(cn1, range=(t_low, t_up)), color.NL(channels[0], range=(vmin, vmax)))
    
    # jet tracer
    channels_jet = get_channel_jet_tracer(snapshot_jet.fn, center=center, lbox=lbox, slab_width=slab_width,
                                      imsize=imsize, jetcolumn=-1, orientation=orientation)
    channels_jet[channels_jet < range_jet[0]] = 1e-24
    channels_jet[channels_jet > range_jet[1]] = 1e-24

    density_array = 10 ** np.linspace(vmin, vmax, 100)
    jet_array = 10 ** np.linspace(range_jet[0], range_jet[1], 100)
    Density_cmap, Jet_cmap = np.meshgrid(density_array, jet_array, indexing='ij')
    cmap_density = color.CoolWarm(color.NL(np.ones_like(Jet_cmap) * 10 ** np.mean(range_jet),range=range_jet),color.NL(Density_cmap.T, range=(vmin,vmax)))

    #---------
    if showbar=='bottom':
        f,axes = plt.subplots(2, 1, height_ratios=[0.93, 0.1], figsize=(5.2, 5 /0.7))
        axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
        axes[0].set_title(snapshot_jet._time_title(t0), fontsize=18)
        axes[0].contour(channels_jet[1], extent=[0, lbox, 0, lbox], levels=levels, colors=contour_color, linewidths=1.5)

        if show_BH:
            inner = Circle((lbox / 2, lbox / 2), 30, ec='black', fc="none", lw=2, ls= '--')
            outer = Circle((lbox / 2, lbox / 2), 90, ec='black', fc="none", lw=2, ls= '-')
            axes[0].add_patch(inner)
            axes[0].add_patch(outer)

        add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(cmap_density, extent=[vmin, vmax, 0, 1], aspect=0.4)
        axes[1].set_yticks([])
        axes[1].set_xlabel(r'log $\rho$ [$M_\odot$ pc$^{-2}$]', labelpad=2, fontsize=13)
        plt.tight_layout()

    elif showbar=='right':
        f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.05], gridspec_kw={'wspace': 0.02})
        axes[0].imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
        axes[0].set_title(snapshot_jet._time_title(t0), fontsize=18)
        axes[0].contour(channels_jet[1], extent=[0, lbox, 0, lbox],levels=levels, colors=contour_color, linewidths=1.5)

        if show_BH:
            inner = Circle((lbox / 2, lbox / 2), 30, ec='black', fc="none", lw=2, ls= '--')
            outer = Circle((lbox / 2, lbox / 2), 90, ec='black', fc="none", lw=2, ls= '-')
            axes[0].add_patch(inner)
            axes[0].add_patch(outer)

        add_scalebar(axes[0], lbox, size=scalebar_size, label=scalebar_label)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].imshow(np.transpose(cmap_density, axes=(1, 0, 2)), extent=[t_up, t_low, vmax, vmin], aspect=10)
        axes[1].set_xticks([])
        axes[1].set_ylabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position('right')

    elif showbar=='none':
        f,ax = plt.subplots(1, 1, figsize=(5,5))
        ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
        ax.set_title(snapshot_jet._time_title(t0), fontsize=18)
        ax.contour(channels_jet[1], extent=[0, lbox, 0, lbox],levels=levels, colors=contour_color, linewidths=1.5)
        add_scalebar(ax, lbox, size=scalebar_size, label=scalebar_label)
        ax.set_xticks([]); ax.set_yticks([])

    elif showbar=='blank':
        f,ax = plt.subplots(1, 1, figsize=(5,5))
        ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
        ax.contour(channels_jet[1], extent=[0, lbox, 0, lbox], levels=levels, colors=contour_color, linewidths=1.5)
        ax.set_xticks([]); ax.set_yticks([])

    if savefig_file is not None:
        plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()
    return f
