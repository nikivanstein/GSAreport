##important for dockerizing later on: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#installing-using-docker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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


def saReport(problem, sample_size, fun, seed=42):
    # create sobol analysis
    X_sobol = saltelli.sample(problem, N=sample_size, calc_second_order=True)
    z_sobol =  np.asarray(list(map(fun, X_sobol)))

    import contextlib
 
    path = 'temp/analysis_temp.txt'
    with open(path, 'w') as f:
        with contextlib.redirect_stdout(f):
            sa = sobol.analyze(problem, z_sobol, print_to_console=True, seed=seed, calc_second_order=True)
    
    sa_dict = dp.format_salib_output(sa, "problem", pretty_names=None)
    print(sa_dict)

    # Apply the default theme
    #sns.set_theme()

    #import graph_tool.inference as community

    #sa_dict_net = copy.deepcopy(sa_dict)
    g = nt.build_graph(sa_dict['problem'], sens='ST', top=150, min_sens=0.01,
                       edge_cutoff=0.005)
    #nt.plot_network_circle(g, inline=True, scale=200)

    inline=True
    scale=200
    
    print("Network plot")

    for i in range(g.num_vertices()):
        g.vp['sensitivity'][i] = (scale * g.vp['sensitivity'][i] )

    epsens = g.edge_properties['second_sens']
    for i in g.edges():
        g.ep['second_sens'][i] =  (g.ep['second_sens'][i]) ** 2

    filename = "sensivity_network.pdf"
    state = graph_tool.inference.minimize_nested_blockmodel_dl(g, deg_corr=True)
    draw.draw_hierarchy(state,
                        vertex_text=g.vp['param'],
                        vertex_text_position="centered",
                        layout = "radial",
                        # vertex_text_color='black',
                        vertex_font_size=8,
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
problem = {
    'num_vars': 5,
    'names': ['X'+str(x) for x in range(5)],
    'bounds': [[-5.0, 5.0]] * 5
    }
fun, opt = bn.instantiate(5, iinstance=1)
saReport(problem, 500, fun, seed=42)