from ..styles import *
import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl
import numpy as np

def jet_distances_paper1(fractions, fractions_uniform, fractions_uniform_dilute, 
                  dist, dist_uniform, dist_uniform_dilute, 
                  time_max, time_max_uniform, time_max_uniform_dilute, title=r'jet power $10^{40}$ erg/s'):
    c1 = 'crimson'#cmap_2(255)
    c2 = 'darkred'#cmap_2(180)
    c3 = 'lightsalmon'#cmap_2(110)

    cmap1 = shading_cmaps(c1)
    cmap2 = shading_cmaps(c2)
    cmap3 = shading_cmaps(c3)

    
    fig, ax = plt.subplots(figsize=(5,3))

    plt.imshow(fractions, cmap=cmap1, extent=(0, time_max-15, 0, 750), vmin=0.5, vmax=1, aspect='auto')
    plt.imshow(fractionsuniform, cmap=cmap2, extent=(0, time_max_uniform-15, 0, 750), vmin=0.5, vmax=1, aspect='auto')
    plt.imshow(fractionsuniform_dilute, cmap=cmap3, extent=(0, time_max_uniform_dilute-15, 0, 750), vmin=0.5, vmax=1, aspect='auto')

    mask = (dist[0] < time_max)
    mask_uniform = (dist_uniform[0] < time_max_uniform)
    mask_uniform_dilute = (dist_uniform_dilute[0] < time_max_uniform_dilute)
    
    ax.plot(dist_uniform_dilute[0][mask_uniform_dilute]-15, dist_uniform_dilute[1][mask_uniform_dilute], c=c3, ls='-', lw=3.5, label='uniform dilute gas')
    ax.plot(dist_uniform_dilute[0][mask_uniform_dilute]-15, dist_uniform_dilute[2][mask_uniform_dilute], c=c3, ls='--', lw=2)

    ax.plot(dist_uniform[0][[mask_uniform]]-15, dist_uniform[1][[mask_uniform]], c=c2, ls='-', lw=3.5, label='uniform gas', zorder=100)
    ax.plot(dist_uniform[0][[mask_uniform]]-15, dist_uniform[2][[mask_uniform]], c=c2, ls='--', lw=2)
      
    ax.plot(dist[0][mask]-15, dist[1][mask], c=c1, ls='-', lw=3.5, label='turbulent gas')
    ax.plot(dist[0][mask]-15, dist[2][mask], c=c1, ls='--', lw=2)
   
    ax.set_title(title)
    ax.set_xlabel('t [Myr]', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 750)
    ax.grid(ls='--', c='gray', alpha=0.4, zorder=0)
    ax.set_ylabel(r'$r$ [pc]', fontsize=10, zorder=100)
    
    ax.legend(fontsize=8, loc='lower right', framealpha=0.6, labelcolor='black')
    
    plt.show()
    
def jet_rampressure(ram_pressure_list_parallel, jet_tracer_fall_x_parallel_left, jet_tracer_fall_x_parallel_right, x_axis, time, time_max=5, savefig=''):
    
    cmap_2, cmap_3 = define_cmaps('paper1')
    real_y = np.flip(x_axis)
    real_x = time
    
    dx = (real_x[1]-real_x[0])/2.
    dy = (real_y[1]-real_y[0])/2.
    extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plt.imshow(np.array(ram_pressure_list_parallel).T, cmap=cmap_3.reversed(), extent=extent, aspect='auto', norm=colors.LogNorm(vmin=1e-11, vmax=1e-7))#
    #plt.colorbar(im, label=r'number density, cm$^{-3}$')
    cb = plt.colorbar(im, label=r'$p_\text{ram}$ [Bar]')
    cb.set_label(label=r'$p_\text{ram}$ [Bar]', fontsize=15)
    # ax.set(xticks=np.arange(0, len(time), 10), xticklabels=np.arange(time.min(), time.max(), 10));
    
    plt.plot(real_x, np.array(jet_tracer_fall_x_parallel_left),lw=2,  c='white', zorder=100)
    plt.plot(real_x, np.array(jet_tracer_fall_x_parallel_right),lw=2,  c='white', zorder=100)
    
    plt.gca().invert_yaxis()
    plt.xlim(0, time_max)
    plt.xticks(ticks=range(int(time_max)), labels=range(int(time_max)))
    plt.xlabel(r'$t$ [Myr]', fontsize=15)
    plt.ylabel(r'$x$ [pc]', fontsize=15)
    
    if len(savefig) > 0:
        plt.savefig(savefig)
    else:
        plt.show()
    plt.close()


def jet_distances(fractions_list, dist_list, 
                  time_max_list, labels_list, 
                  colors_list, title=r'jet power $10^{40}$ erg/s'):
    
    fig, ax = plt.subplots(figsize=(5,3))

    for i in range(len(fractions_list)):
        cmap = shading_cmaps(colors_list[i])
    
    
        plt.imshow(fractions_list[i], cmap=cmap, extent=(0, time_max_list[i]-15, 0, 750), vmin=0.5, vmax=1, aspect='auto')
    
        mask = (dist_list[i][0] < time_max_list[i])
        
        ax.plot(dist_list[i][0][mask]-15, dist_list[i][1][mask], c=colors_list[i], ls='-', lw=3.5, label=labels_list[i])
        ax.plot(dist_list[i][0][mask]-15, dist_list[i][2][mask], c=colors_list[i], ls='--', lw=2)
       
    ax.set_title(title)
    ax.set_xlabel('t [Myr]', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 750)
    ax.grid(ls='--', c='gray', alpha=0.4, zorder=0)
    ax.set_ylabel(r'$r$ [pc]', fontsize=10, zorder=100)
    
    ax.legend(fontsize=8, loc='lower right', framealpha=0.6, labelcolor='black')
    
    plt.show()