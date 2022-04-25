##important for dockerizing later on: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#installing-using-docker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
matplotlib.use("macOSX")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from deap import benchmarks
from sklearn.ensemble import RandomForestRegressor

from SALib.sample import saltelli,finite_diff, fast_sampler, latin
from SALib.analyze import morris,sobol, dgsm, fast, delta, rbd_fast
from SALib.util import read_param_file
from SALib.sample.morris import sample
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


from tqdm import tqdm

# Import seaborn
import seaborn as sns

import warnings; warnings.filterwarnings('ignore')
import copy
from graph_tool.all import *
from graph_tool import Graph, draw, inference
from bokeh.plotting import show, output_notebook
import os.path as op
import os
import savvy.data_processing as dp
import savvy.interactive_plots as ip
from savvy.plotting import make_plot, make_second_order_heatmap
import savvy.network_tools as nt

#output_notebook()

def improved_covariance_plot(ax, Si, opts=None, unit=""):
    '''Plots mu* against sigma or the 95% confidence interval

    '''
    if opts is None:
        opts = {}

    if Si['sigma'] is not None:
        # sigma is not present if using morris groups
        y = Si['sigma']
        out = ax.scatter(Si['mu_star'], y, c=u'b', marker=u'o',
                         **opts)
        ax.set_ylabel(r'$\sigma$')

        ax.set_xlim(0,)
        ax.set_ylim(0,)

        x_axis_bounds = np.array(ax.get_xlim())

        line1, = ax.plot(x_axis_bounds, x_axis_bounds, 'k-')
        line2, = ax.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'y--')
        line3, = ax.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'r-.')

        ax.legend((line1, line2, line3), (r'$\sigma / \mu^{\star} = 1.0$',
                                          r'$\sigma / \mu^{\star} = 0.5$',
                                          r'$\sigma / \mu^{\star} = 0.1$'),
                  loc='best')

    else:
        y = Si['mu_star_conf']
        out = ax.scatter(Si['mu_star'], y, c=u'k', marker=u'o',
                         **opts)
        ax.set_ylabel(r'$95\% CI$')

    ax.set_xlabel(r'$\mu^\star$ ' + unit)
    ax.set_ylim(0-(0.01 * np.array(ax.get_ylim()[1])), )

    return out

def generate_report(problem, sample_size, fun, top=50, seed=42):
    #create a sobol analysis
    morris_plt(problem, sample_size, fun, top=50, num_levels=4)
    #sobol_plt(problem, sample_size, fun, top, seed=)

def morris_plt(problem, sample_size, fun, top=50, num_levels=4):
    X = sample(problem, N=sample_size, num_levels=num_levels)
    y =  np.asarray(list(map(fun, X)))
    sns.set_theme()
    fig, (ax1) = plt.subplots(1, 1, figsize=[8,problem['num_vars']])
    Si = morris.analyze(problem, X, y, conf_level=0.95,
                    print_to_console=False, num_levels=num_levels)
    horizontal_bar_plot(ax1, Si,{}, sortby='mu_star')
    plt.tight_layout()
    plt.savefig("morris.png")
    plt.clf()

    sns.set_theme()
    fig, (ax1) = plt.subplots(1, 1, figsize=[8,8])
    improved_covariance_plot(ax1, Si)
    plt.tight_layout()
    plt.savefig("morris2.png")

def sobol_plt(problem, sample_size, fun, top=50, seed=42):
    # create sobol analysis
    X = saltelli.sample(problem, N=sample_size, calc_second_order=True)
    y =  np.asarray(list(map(fun, X)))

    sa = sobol.analyze(problem, y, print_to_console=False, seed=seed, calc_second_order=True)
    
    sa_dict = dp.format_salib_output(sa, "problem", pretty_names=None)
    g = nt.build_graph(sa_dict['problem'], sens='ST', top=top, min_sens=0.01,
                       edge_cutoff=0.005)
    inline=True
    scale=200
    for i in range(g.num_vertices()):
        g.vp['sensitivity'][i] = (scale * g.vp['sensitivity'][i] )

    #epsens = g.edge_properties['second_sens']
    #for i in g.edges():
    #    g.ep['second_sens'][i] =  (g.ep['second_sens'][i])

    filename = "sensivity_network.png"
    state = graph_tool.inference.minimize_nested_blockmodel_dl(g)
    draw.draw_hierarchy(state,
                        vertex_text=g.vp['param'],
                        vertex_text_position="centered",
                        layout = "radial",
                        hide = 2,
                        # vertex_text_color='black',
                        vertex_font_size=12,
                        vertex_size=g.vp['sensitivity'],
                        #vertex_color='#006600',
                        #vertex_fill_color='#008800',
                        vertex_halo=True,
                        vertex_halo_color='#b3c6ff',
                        vertex_halo_size=g.vp['confidence'],
                        edge_pen_width=g.ep['second_sens'],
                        # subsample_edges=100,
                        output_size=(1600, 1600),
                        inline=inline,
                        output=filename
                        )

from benchmark import bbobbenchmarks as bn
dim = 50
problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[-5.0, 5.0]] * dim
    }
fun, opt = bn.instantiate(22, iinstance=1)
generate_report(problem, 500, fun, seed=42)