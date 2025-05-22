import matplotlib.pyplot as plt    ## plot stuff
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib as mpl

def define_colors(style):
    if style == 'paper1':
        color1 = "gold"
        color2 = "darkmagenta"
        color3 = "darkblue"
    if style == 'paper2':
        color1 = "gold"
        color2 = "darkmagenta"
        color3 = "darkblue"

    return color1, color2, color3
    

def define_cmaps(style):
    color1, color2, color3 = define_colors(style)
    cmap_2 = colors.LinearSegmentedColormap.from_list("", [color1, color2])
    cmap_3 = colors.LinearSegmentedColormap.from_list("", [color1, color2, color3])

    return cmap_2, cmap_3

def shading_cmaps(color):
    n = 100
    cmap_list = np.array([colors.to_rgba(color)] * n, dtype=np.float64)
    cmap_list[:,3] = np.linspace(0, 1, n) ** 3
    cmap = colors.ListedColormap(cmap_list)
    return cmap

def make_scientific():
    import scienceplots
    plt.style.use(['science','no-latex'])
    plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=15) 
    plt.rcParams['axes.linewidth'] = 1.8 #set the value globally
    plt.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.unicode_minus"] = False
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['xtick.minor.size'] = 3


def plot_size(size=300):
    mpl.rcParams['figure.dpi']= size


def background(color, full=False):
    if color == 'cfa':
        color = '2F366E'
    if (color == '2F366E') | (color == 'black'):
        params = {"figure.facecolor" : color,
            "ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w",
          "text.color" : "w",
                 # 'boxplot.whiskerprops.color': 'white', ###
                 # 'boxplot.boxprops.color': 'white',
                 # 'boxplot.capprops.color': 'white',
                 #  'boxplot.flierprops.markeredgecolor': 'white',
                 #  'patch.edgecolor': 'white'
                 }
        plt.rcParams.update(params)
    if full == True:
        params = {"axes.facecolor" : color}    
        plt.rcParams.update(params)