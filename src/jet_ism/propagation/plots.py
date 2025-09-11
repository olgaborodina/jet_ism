from ..styles import *
import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl
import numpy as np

def jet_distances_paper1(distance_info_turb, distance_info_uniform, distance_info_uniform_dilute, distance_min=0, distance_max=750, title=r'jet power $10^{40}$ erg/s'):
    """
    Make plot of jet distances for three different environments
    Input: distance_info_turb            (output of the function percentile_distance_3 and percentile_evolution for turbulent medium
           distance_info_uniform         (output of the function percentile_distance_3 and percentile_evolution for uniform medium)
           distance_info_uniform_dilute  (output of the function percentile_distance_3 and percentile_evolution for uniform dilute medium)
           distance_min                  (minimum distance in pc, default is 0)
           distance_max                  (maximum distance in pc, default is 750)
           title                         (title of the plot, default is 'jet power $10^{40}$ erg/s')
    """
    dist, fractions, time_max = distance_info_turb
    dist_uniform, fractions_uniform, time_max_uniform = distance_info_uniform
    dist_uniform_dilute, fractions_uniform_dilute, time_max_uniform_dilute = distance_info_uniform_dilute

    c1, c2, c3 = define_colors('paper1')

    cmap1 = shading_cmaps(c1)
    cmap2 = shading_cmaps(c2)
    cmap3 = shading_cmaps(c3)

    fig, ax = plt.subplots(figsize=(5,3))

    plt.imshow(fractions, cmap=cmap1, extent=(0, time_max-15, distance_min, distance_max), vmin=0.5, vmax=1, aspect='auto')
    plt.imshow(fractionsuniform, cmap=cmap2, extent=(0, time_max_uniform-15, distance_min, distance_max), vmin=0.5, vmax=1, aspect='auto')
    plt.imshow(fractionsuniform_dilute, cmap=cmap3, extent=(0, time_max_uniform_dilute-15,distance_min, distance_max), vmin=0.5, vmax=1, aspect='auto')

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
    
def jet_rampressure(ram_pressure_list_parallel, jet_tracer_fall_x_parallel_left, jet_tracer_fall_x_parallel_right, x_axis, time, tmin=15, tmax=5, savefig=''):
    """Overplot ram pressure with jet tracer extent as a function of time
    Input: ram_pressure_list_parallel       (output of the function calculate_rampressure_jet_axis, list of ram pressure arrays at different times)
           jet_tracer_fall_x_parallel_left  (output of the function calculate_rampressure_jet_axis, list of x positions where jet tracer falls below threshold on the left side)
           jet_tracer_fall_x_parallel_right (output of the function calculate_rampressure_jet_axis, list of x positions where jet tracer falls below threshold on the right side)
           x_axis                           (output of the function calculate_rampressure_jet_axis, array of x positions)
           time                             (output of the function calculate_rampressure_jet_axis, array of times in Myr)
           tmin                             (minimum time in Myr, default is 15)
           tmax                             (maximum time in Myr after tmin, default is 5)
           savefig                          (name of the file to save the figure, if empty string, show the figure instead, default is '')
        """
    cmap_2, cmap_3 = define_cmaps('paper1')
    real_y = np.flip(x_axis)
    real_x = time - tmin
    
    dx = (real_x[1]-real_x[0])/2.
    dy = (real_y[1]-real_y[0])/2.
    extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]
    
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plt.imshow(np.array(ram_pressure_list_parallel).T, cmap=cmap_3.reversed(), extent=extent, aspect='auto', norm=colors.LogNorm(vmin=1e-11, vmax=1e-7))#
    cb = plt.colorbar(im, label=r'$p_\text{ram}$ [Bar]')
    cb.set_label(label=r'$p_\text{ram}$ [Bar]', fontsize=15)

    plt.plot(real_x, np.array(jet_tracer_fall_x_parallel_left),lw=2,  c='white', zorder=100)
    plt.plot(real_x, np.array(jet_tracer_fall_x_parallel_right),lw=2,  c='white', zorder=100)
    
    plt.gca().invert_yaxis()
    plt.xlim(0, tmax)
    plt.xticks(ticks=range(int(tmax)), labels=range(int(tmax)))
    plt.xlabel(r'$t$ [Myr]', fontsize=15)
    plt.ylabel(r'$x$ [pc]', fontsize=15)
    
    if len(savefig) > 0:
        plt.savefig(savefig)
    else:
        plt.show()
    plt.close()


def jet_distances(dist_list, fractions_list,  
                  time_max_list, labels_list, 
                  colors_list, title=r'jet power $10^{40}$ erg/s'):
    """
    Make plot of jet distances with shading for fractions of jet material
    Input: dist_list        (list of outputs of the function percentile_distance_3 for different simulations)
           fractions_list   (list of outputs of the function percentile_evolution for different simulations)
           time_max_list    (list of maximum times in Myr for different simulations, also output of the function percentile_evolution)
           labels_list      (list of labels for different simulations)
           colors_list      (list of colors for different simulations)
           title            (title of the plot, default is 'jet power $10^{40}$ erg/s')
    """   
    
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


def jet_distances_3(dist_list, labels_list, 
                  colors_list, title=r'jet power $10^{40}$ erg/s', loc='best'):
    """
    Make plot of jet distances for for 50th, 80th, and 100th percentiles 
    Input: dist_list        (list of outputs of the function percentile_distance_3 for different simulations)
           labels_list      (list of labels for different simulations)
           colors_list      (list of colors for different simulations)
           title            (title of the plot, default is 'jet power $10^{40}$ erg/s')
           loc              (location of the legend, default is 'best')
    """    
    fig, ax = plt.subplots(figsize=(5,3))

    for i in range(len(dist_list)):
        ax.plot(dist_list[i][0], dist_list[i][2], label=labels_list[i], c=colors_list[i], ls='-')
        ax.fill_between(dist_list[i][0], dist_list[i][1], dist_list[i][3], color=colors_list[i], ls='-', alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel('t [Myr]', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 750)
    ax.grid(ls='--', c='gray', alpha=0.4, zorder=0)
    ax.set_ylabel(r'$r$ [pc]', fontsize=10, zorder=100)
    
    ax.legend(fontsize=8, loc=loc, framealpha=0.6, labelcolor='black')
    
    plt.show()