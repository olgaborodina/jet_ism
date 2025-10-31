from .. import (unit_velocity, unit_length, unit_mass, unit_time_in_megayr, PROTONMASS, BOLTZMANN, mu, GAMMA, get_time_from_snap, rho_to_numdensity, _make_gif)
from .base import *
from ..gas.general import *
from .get_channel import *

import matplotlib.font_manager as fm

class snapshot2:
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


    def overlap2_jet_temp(self, lbox, slab_width=None, imsize=2000, orientation='xy', show=True, center=None, cmap_jet='cubehelix', showbar=True, title='', vmin=-2, vmax=2, jmin=-5, jmax=-0, tmin=2.5, tmax=7, savefig_file=None, t0=0):
        """
        plot jet map together with gas density+temp map
        
        Parameters
        ----------
        lbox: box length
        slab_width: projection depth
        imsize: Npixel of the image
        orientation: projection of 'xy','yz','xz'
        showbar: whether to show the colorbar, under, side, none, or blankor none
        show: whether to show the plot, boolean, default True
        savefig_file: filename to save the figure, default None (not saving)
        t0: time offset for the title in Myr, default 0
        tmin, tmax: color range for the temperature map in log scale, default (2.5, 7.0)
        jmin, jmax: color range for the jet map in log scale, default (-5, 0)
        vmin, vmax: color range for the density map in log scale, default None (1 and 99 percentiles)
        center: center of the box, default None (box center)
        """
        
        BoxSize = self.BoxSize
        if center == None:
            center = self.center
        t_low, t_up = tmin, tmax
        
        cn0 = np.ones((imsize,imsize))*1e-6 # dens placeholder
        cn1 = np.ones((imsize,imsize)) * 10 ** np.mean([t_low, t_up]) # temp placeholder
        
        
        #---------

        #f.tight_layout()
        # gas rho temp
        channels = get_channel_gas_rhotemp(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                   imsize=imsize,orientation=orientation)
        
        x = channels[0][channels[0]>0]
        
        vmin0,vmax0 = np.log10(np.percentile(x, 1)), np.log10(np.percentile(x, 99))
        if vmin == None:
            vmin = vmin0
        if vmax == None:
            vmax = vmax0
            
        img_dens = color.CoolWarm(color.NL(cn1,range=(t_low,t_up)),color.NL(channels[0], range=(vmin,vmax)))
        img_temp = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(cn0, range=(-7,-6)))
        img_2field = color.CoolWarm(color.NL(channels[1],range=(t_low,t_up)),color.NL(channels[0], range=(vmin,vmax)))
        
        # jet tracer
        jetmap = make_colormap(cmap_jet)
        channels = get_channel_jet_tracer(self.fn,center=center,lbox=lbox,slab_width=slab_width,
                                          imsize=imsize,jetcolumn=-1,orientation=orientation)
        
        _img_jet = jetmap(color.NL(channels[1],range=(jmin, jmax)),color.NL(cn0, range=(-7,-6)))
        scale = color.NL(channels[1],range=(jmin, jmax))
        img_jet = overlay(img_dens,_img_jet,1 * np.nan_to_num(scale))
        
        #-------- create cmaps:
        density_array = 10 ** np.linspace(vmin,vmax, 100)
        temperature_array = 10 ** np.linspace(t_low,t_up, 100)
        jet_array = 10 ** np.linspace(jmin, jmax, 100)
        Density_cmap, Temperature_cmap = np.meshgrid(density_array, temperature_array, indexing='ij')
        Density_cmap, Jet_cmap = np.meshgrid(density_array, jet_array, indexing='ij')
        
        cmap_density_temperature = color.CoolWarm(color.NL(Temperature_cmap,range=(t_low,t_up)),color.NL(Density_cmap, range=(vmin,vmax)))

        cmap_density = color.CoolWarm(color.NL(np.ones_like(Temperature_cmap) * 10 ** np.mean([t_low, t_up]),range=(t_low,t_up)),color.NL(Density_cmap.T, range=(vmin,vmax)))
        cmap_jet = jetmap(color.NL(Jet_cmap, range=(jmin, jmax)),color.NL(np.ones_like(Density_cmap) * 1e-6, range=(-7,-6)))

        
        # scale_cmap = color.NL(Jet_cmap,range=(-5,0))
        # cmap_density_jet = overlay(cmap_density, cmap_jet, 0.2 * np.nan_to_num(scale_cmap))
        #---------
        fields = [img_jet, img_2field]
        labels = [r'Density + Jet Tracer', r'Density + Temperature']

        fontprop = fm.FontProperties(size=12)
        if showbar == 'under':
            f,axes = plt.subplots(3,2, width_ratios=[0.5, 0.5], height_ratios=[0.92, 0.05, 0.05], figsize=(11,7))
            f.subplots_adjust(hspace=0.5,wspace=0.3)
            for i in range(0,len(fields)):
                ax = axes[0][i]
                ax.imshow(fields[i],extent=[0,lbox,0,lbox],origin='lower')
                ax.text(0.05,0.9,labels[i],color='white',fontsize=15,weight="bold", transform=ax.transAxes)
                # ax.set_xlabel('pc', fontsize=15)
                if i==0:
                    ax.set_title(title, fontsize=15)
                if i==1:
                    ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=15)
                
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
            axes[1][0].imshow(cmap_jet, extent=[jmin, jmax, jmin, jmax], aspect=0.055)
            axes[1][0].set_xlabel(r'$\log{X_\text{jet}}$', labelpad=2, fontsize=13)
            axes[1][0].set_yticklabels([])
            axes[1][0].set_yticks([])
            axes[2][0].imshow(cmap_density, extent=[vmin, vmax, vmin, vmax], aspect=0.055)
            axes[2][0].set_yticklabels([])
            axes[2][0].set_yticks([])
            axes[2][0].set_xlabel(r'log $\rho$ [$M_\odot$ pc$^{-2}$]', labelpad=2, fontsize=13)
    
            gs = axes[1, 1].get_gridspec()
            # remove the underlying axes
            for ax in axes[1:, -1]:
                ax.remove()
            axbig = f.add_subplot(gs[1:, -1])
            axbig.imshow(cmap_density_temperature, extent=[t_low, t_up, vmax, vmin], aspect=0.4)
            axbig.set_xlim(t_low, t_up)
            # axbig.set_xticks(ticks=np.arange(np.round(t_low), np.round(t_up)), labels=np.arange(np.round(t_low), np.round(t_up)))
            axbig.set_xlabel(r'$\log T$ [K]', labelpad=2, fontsize=13)
            axbig.set_ylabel(r'log $\rho$ ' + '\n' + r' [$M_\odot$ pc$^{-2}$]', labelpad=2, fontsize=13)

        elif showbar == 'side':
            f,axes = plt.subplots(1,5, width_ratios=[0.07, 0.07, 0.3, 0.3, 0.11], figsize=(16,5))
            f.subplots_adjust(hspace=0.5,wspace=0.01)
            for i in range(0,len(fields)):
                ax = axes[i+2]
                ax.imshow(fields[i],extent=[0,lbox,0,lbox],origin='lower')
                ax.text(0.05,0.9,labels[i],color='white',fontsize=15,weight="bold", transform=ax.transAxes)
                if i==0:
                    ax.set_title(title, fontsize=15)
                if i==1:
                    ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=15)
                
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
            axes[0].imshow(np.transpose(cmap_jet, axes=(1,0,2)), extent=[jmax, jmin, jmax, jmin], aspect=18)
            axes[0].set_ylabel('log (jet tracer)', labelpad=2)
            axes[0].set_xticklabels([])
            axes[3].set_yticks([])
            axes[1].imshow(np.transpose(cmap_density, axes=(1,0,2)), extent=[vmax, vmin, vmax, vmin], aspect=18)
            axes[1].set_xticklabels([])
            axes[1].set_xticks([])
            axes[1].set_ylabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
            # axes[1][0].set_xlabel(labels[i].split("+")[1])
            # axes[1][1].set_ylabel(labels[i].split("+")[0])
    
            # gs = axes[1, 1].get_gridspec()
            # # # remove the underlying axes
            # for ax in axes[2:4]:
            #     ax.remove()
            # axbig = f.add_subplot(gs[1:, -1])
            axes[4].imshow(np.transpose(cmap_density_temperature, axes=(1,0,2)), extent=[vmin, vmax, t_up, t_low], aspect=4.3)
            axes[4].set_ylabel('log (temperature [K])', labelpad=4)
            axes[4].set_xlabel(r'log ($\rho$ [$M_\odot$ pc$^{-2}$])', labelpad=2)
            axes[4].yaxis.tick_right()
            axes[4].yaxis.set_label_position('right')
            
        elif showbar == 'none':
            f,axes = plt.subplots(1,2, width_ratios=[0.5, 0.5], figsize=(11,7*0.92))
            f.subplots_adjust(wspace=0.3)
            fontprop = fm.FontProperties(size=12)
            for i in range(0,len(fields)):
                ax = axes[i]
                ax.imshow(fields[i],extent=[0,lbox,0,lbox],origin='lower')
                ax.text(0.05,0.9,labels[i],color='white',fontsize=15,weight="bold", transform=ax.transAxes)
                # ax.set_xlabel('pc', fontsize=15)
                if i==0:
                    ax.set_title(title, fontsize=15)
                if i==1:
                    ax.set_title(r'$t=$%.2f Myr'%(self.time - t0), fontsize=15)

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
        else:
            raise ValueError('showbar can be only under,sode or none')
        
        try:
            plt.savefig(savefig_file)
        except: print("Can't save file")
        if show == True:
            plt.show()
        else: 
            plt.close()
        return f


def overlap2_jet_temp_gif(figures_directory, lbox, slab_width, timestep=4):
    """
    Create a gif from the phase diagram images in the given directory
    Input: figures_directory (string, location of the figure files)
           timestep (integer, time between frames in the gif in ms, default is 4)
    Output: None
    """       
    ifilename = figures_directory + f'gaepsi_2jettemp_{lbox}_{slab_width}*.png'
    ofilename = figures_directory + f'/gaepsi_2jettemp_{lbox}_{slab_width}.gif'
    _make_gif(ifilename, ofilename, timestep)