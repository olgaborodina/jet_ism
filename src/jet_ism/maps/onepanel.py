from .. import (unit_velocity, unit_length, unit_mass, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity)
from .base import *
from ..gas.general import *
from .get_channel import *

import matplotlib.font_manager as fm

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
        self.time = part['Header'].attrs['Time'] * unit_time_in_megayr
        
        BoxSize = part['Header'].attrs['BoxSize']
        self.BoxSize = BoxSize
        self.center = np.array([0.5 * BoxSize, 0.5 * BoxSize, 0.5 * BoxSize])

                
    def overlap_jet(self, lbox, slab_width=None, imsize=2000, orientation='xy', 
                    show=True, range=(-5,0), vmin=None, vmax=None, center=None):
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
        """
        BoxSize = self.BoxSize
        if center == None:
            center = self.center

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
        ax.set_title('t=%.2f Myr'%(self.time), fontsize=18)

        if show == True:
            plt.show()
        else: 
            plt.close()
        return f

    def overlap_shock(self, lbox, slab_width=None, imsize=2000, orientation='xy', 
                        show=True, range=(-3.5, -1.8), vmin=None, vmax=None, center=None):
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
        """
        BoxSize = self.BoxSiz
        if center == None:
            center = self.center

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
        ax.set_title('t=%.2f Myr'%(self.time), fontsize=18)
        
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f


    def overlap_soundspeed(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True, 
                            showbar=False, range=(4, 5.2), vmin=-2, vmax=2, center=None, savefig_file=None, t0=0):
        """
        plot shock map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        show: whether to show the plot, boolean, default True
        showbar: whether to show the colorbar, boolean, default False
        range: color range for the sound speed map in log scale, default (4, 5.2)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        """
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        # fontprop = fm.FontProperties(size=12)

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
    
        if showbar==True:
            f,axes = plt.subplots(2,1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img2,extent=[0, lbox, 0, lbox],origin='lower')
            axes[0].set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)

            scalebar = AnchoredSizeBar(axes[0].transData,
                           200, '200 pc', 'lower center', 
                           pad=0.2,
                           color='white',
                           frameon=False,
                           size_vertical=1.5,
                           fontproperties=fontprop,
                           sep=3)
            axes[0].add_artist(scalebar)
            axes[0].set_xticks([])
            axes[0].set_yticks([])            
            
            axes[1].imshow(cbar_ss, extent=[ss_low, ss_up,ss_low, ss_up,], aspect=0.055)
            axes[1].set_xlabel(r'sound speed cm/s', labelpad=2)
            axes[1].set_yticks([])
            axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

            try:
                plt.savefig(savefig_file)
            except: print("Can't save file")
            if show == True:
                plt.show()
            else: 
                plt.close()
        else:
            f,ax = plt.subplots(1, 1, figsize=(5, 5))            
            ax.imshow(img2, extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)
            
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
            try:
                plt.savefig(savefig_file)
            except: print("Can't save file")        
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f


    def overlap_temp(self, lbox, slab_width=None, imsize=2000, orientation='xy', showbar='bottom', 
                    show=True, savefig_file=None, t0=0, tmin=2.5, tmax=7.0, vmin=-2, vmax=2, center=None):
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
        BoxSize = self.BoxSize
        if center==None:
            center = self.center

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
        fontprop = fm.FontProperties(size=12)
        if showbar=='bottom':
            f,axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)

            scalebar = AnchoredSizeBar(axes[0].transData,
                           200, '200 pc', 'lower center', 
                           pad=0.2,
                           color='white',
                           frameon=False,
                           size_vertical=1.5,
                           fontproperties=fontprop,
                           sep=3)
            axes[0].add_artist(scalebar)
            axes[0].set_xticks([])
            axes[0].set_yticks([])            
            
            axes[1].imshow(cmap_density_temperature, extent=[t_low, t_up, vmax, vmin], aspect=0.32)
            axes[1].set_xlabel(r'log $T$ [K]', labelpad=2)
            axes[1].set_ylabel(r'$\log{\rho}$ ' + '\n' + r' [$M_\odot$ pc$^{-2}$])', labelpad=2)


            try:
                plt.savefig(savefig_file)
            except: print("Can't save file")
            if show == True:
                plt.show()
            else: 
                plt.close()
        elif showbar=='right':
            f, axes = plt.subplots(1, 2, figsize=(7, 5), width_ratios=[0.4, 0.1])
            axes[0].imshow(img1, extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].set_title('t=%.2f Myr'%(self.time), fontsize=18)

            scalebar = AnchoredSizeBar(axes[0].transData,
                           200, '200 pc', 'lower center', 
                           pad=0.2,
                           color='white',
                           frameon=False,
                           size_vertical=1.5,
                           fontproperties=fontprop,
                           sep=3)
            axes[0].add_artist(scalebar)
            axes[0].set_xticks([])
            axes[0].set_yticks([]) 
    
            axes[1].imshow(np.transpose(cmap_density_temperature, axes=(1, 0, 2)), extent=[vmin, vmax, t_up, 3], aspect=4.1)
            axes[1].set_ylabel('log (temperature [K])', labelpad=4)
            axes[1].set_xlabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
            axes[1].yaxis.tick_right()
            axes[1].yaxis.set_label_position('right')
        
            if savefig_file != None:
                plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
            if show == True:
                plt.show()
            else: 
                plt.close()
        elif showbar=='none':
            f,ax = plt.subplots(1, 1, figsize=(5,5))            
            ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)
            
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
            if savefig_file != None:
                plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
            if show == True:
                plt.show()
            else: 
                plt.close()
        elif showbar=='blank':
            f,ax = plt.subplots(1, 1, figsize=(5,5))            
            ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if savefig_file != None:
                plt.savefig(savefig_file, bbox_inches='tight', dpi=300)
            if show == True:
                plt.show()
            else: 
                plt.close()
        return f

    def overlap_sigma_velocity(self,lbox,slab_width=None,imsize=2000,orientation='xy', show=True):
        """
        TODO: NOT TESTED
        plot shock map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        """
        BoxSize = self.BoxSize
        center = self.center

        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize))*1e4 # temp placeholder
        t_low, t_up = 2.0, 7.0 # 1e2K, 1e7K
        
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        x = channels[0][channels[0]>0]
        vmin0,vmax0 = np.log10(np.percentile(x,1)),np.log10(np.percentile(x,99))
        img1 = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(channels[0], range=(vmin0,vmax0)))
        #img1 = color.CoolWarm(color.NL(channels[0], range=(vmin0,vmax0)))
        
        # radial velocity
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
        ax.set_title('t=%.2f Myr'%(self.time), fontsize=18)
        
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f

    
    def overlap_sfr(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True, 
                    range=(-5, 0), vmin=None, vmax=None, weight='sfr'):
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
        """
        BoxSize = self.BoxSize
        center = self.center

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
        ax.set_title('t=%.2f Myr'%(self.time), fontsize=18)

        if show == True:
            plt.show()
        else: 
            plt.close()
        return f


    def overlap_random(self, mask_random, lbox, slab_width=None, imsize=2000, orientation='xy', show=True,
                        range=(-5,0), vmin=-2, vmax=2):
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
        """
        BoxSize = self.BoxSize
        center = self.center

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
        ax.set_title('t=%.2f Myr'%(self.time), fontsize=18)

        if show == True:
            plt.show()
        else: 
            plt.close()
        return f



    def overlap_temp_stars(self, lbox, slab_width=None, imsize=2000, orientation='xy', showbar=False, show=True, savefig_file=None, t0=0, 
                            tmin=2.5, tmax=7.0, vmin=-2, vmax=2, center=None, age_cut=False):
        """
        plot jet map overlaid on top of gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        orientation: projection of 'xy','yz','xz'
        imsize: Npixel of the image
        showbar: whether to show the colorbar, boolean, default False
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        tmin, tmax: color range for the temperature map in log scale, default (
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        age_cut: whether to apply age cut (1 Myr) for the star particles, boolean, default False
        """
        BoxSize = self.BoxSize
        if center == None:
            center = self.center

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
        if showbar==True:
            f,axes = plt.subplots(2, 1, height_ratios=[0.93, 0.2], figsize=(5.2, 5 / 0.7))
            f.subplots_adjust(hspace=0.3)
            axes[0].imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            axes[0].scatter(stars_coords[0] - coord_shift[0], stars_coords[1] - coord_shift[1], 
                            c='yellow', s=25 * (stars_mass / stars_mass.max()) ** 2, marker='*')
            axes[0].set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)

            scalebar = AnchoredSizeBar(axes[0].transData,
                           200, '200 pc', 'lower center', 
                           pad=0.2,
                           color='white',
                           frameon=False,
                           size_vertical=1.5,
                           fontproperties=fontprop,
                           sep=3)
            axes[0].add_artist(scalebar)
            axes[0].set_xticks([])
            axes[0].set_yticks([])            
            
            axes[1].imshow(cmap_density_temperature, extent=[t_low, t_up, vmax, vmin], aspect=0.32)
            axes[1].set_xlabel(r'log $T$ [K]', labelpad=2)
            axes[1].set_ylabel(r'$\log{\rho}$ ' + '\n' + r' [$M_\odot$ pc$^{-2}$])', labelpad=2)


            try:
                plt.savefig(savefig_file)
            except: print("Can't save file")
            if show == True:
                plt.show()
            else: 
                plt.close()
        else:
            f,ax = plt.subplots(1, 1, figsize=(5,5))            
            ax.imshow(img1,extent=[0, lbox, 0, lbox], origin='lower')
            ax.scatter(stars_coords[0]- coord_shift[0], stars_coords[1]- coord_shift[1], 
                            c='yellow', s=25 * (stars_mass / stars_mass.max()) ** 2, marker='*')
            ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=18)
            
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
            try:
                plt.savefig(savefig_file)
            except: print("Can't save file")        
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f
