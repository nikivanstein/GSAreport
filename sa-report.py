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
import pandas as pd
from SALib.sample import saltelli,finite_diff, fast_sampler, latin
from SALib.analyze import morris,sobol, dgsm, fast, delta, rbd_fast, pawn
from SALib.util import read_param_file
from SALib.sample.morris import sample
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from SALib.plotting.bar import plot as barplot
from SALib.plotting.hdmr import plot as hdmrplot
from tqdm import tqdm
from bokeh.embed import components

# Import seaborn
import seaborn as sns
import pandas_bokeh
import warnings; warnings.filterwarnings('ignore')
import copy
from graph_tool.all import *
from graph_tool import Graph, draw, inference
from bokeh.plotting import output_file, save

import os.path as op
import os
import savvy.data_processing as dp
import plotting.interactive_plots as ip
from plotting.plotting import make_plot, make_second_order_heatmap
import savvy.network_tools as nt
from bokeh.plotting import figure
#output_notebook()


def generate_report(problem, sample_size, fun, top=50, seed=42):
    if top > problem['num_vars']:
        top = problem['num_vars']

    lhs_scripts, lhs_divs = lhs_methods(problem, sample_size, fun, top, seed=seed)
    morris_scripts, morris_divs = morris_plt(problem, sample_size, fun, top=top, num_levels=4)
    sobol_scripts, sobol_divs = sobol_plt(problem, sample_size, fun, top, seed=seed)
    file = open('template/index.html', mode='r')
    html_template = file.read()
    html_template = html_template.replace("#SOBOL1#", sobol_divs[0])
    html_template = html_template.replace("#SOBOL2#", sobol_divs[1])
    html_template = html_template.replace("#SOBOL_SCRIPT1#", sobol_scripts[0])
    html_template = html_template.replace("#SOBOL_SCRIPT2#", sobol_scripts[1])

    html_template = html_template.replace("#MORRIS1#", morris_divs[0])
    html_template = html_template.replace("#MORRIS2#", morris_divs[1])
    html_template = html_template.replace("#MORRIS_SCRIPT1#", morris_scripts[0])
    html_template = html_template.replace("#MORRIS_SCRIPT2#", morris_scripts[1])

    html_template = html_template.replace("#FAST#", lhs_divs[0])
    html_template = html_template.replace("#DELTA1#", lhs_divs[1])
    html_template = html_template.replace("#DELTA2#", lhs_divs[2])
    html_template = html_template.replace("#PAWN#", lhs_divs[3])
    html_template = html_template.replace("#FAST_SCRIPT#", lhs_scripts[0])
    html_template = html_template.replace("#DELTA_SCRIPT1#", lhs_scripts[1])
    html_template = html_template.replace("#DELTA_SCRIPT2#", lhs_scripts[2])
    html_template = html_template.replace("#PAWN_SCRIPT#", lhs_scripts[3])
    with open('template/report.html', 'w') as f:
        f.write(html_template)
    

def lhs_methods(problem, sample_size, fun, top, seed):
    plottools = "hover, wheel_zoom, save, reset," # , tap"
    X = latin.sample(problem, sample_size, seed=seed)
    y =  np.asarray(list(map(fun, X)))
    Si = rbd_fast.analyze(problem, X, y, print_to_console=False)

    df = Si.to_df()
    df.reset_index(inplace=True)
    df = df.sort_values(by=['S1'], ascending=False)
    dftop = df.iloc[:top]
    p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="RDB Fast", tools=plottools)
    p = ip.plot_errorbar(dftop, p, base_col="S1", error_col="S1_conf")
    p.sizing_mode = "scale_width"
    script1, div1 = components(p)
    
    Si = delta.analyze(problem, X, y, print_to_console=False)
    df = Si.to_df()
    df.reset_index(inplace=True)
    df = df.sort_values(by=['S1'], ascending=False)
    dftop = df.iloc[:top]
    p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="S1", tools=plottools)
    p = ip.plot_errorbar(dftop, p, base_col="S1", error_col="S1_conf")
    p.sizing_mode = "scale_width"
    script2, div2 = components(p)

    df = df.sort_values(by=['delta'], ascending=False)
    dftop = df.iloc[:top]
    p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, title="Delta",
               toolbar_location="right",
               tools=plottools)
    p = ip.plot_errorbar(dftop, p, base_col="delta", error_col="delta_conf")
    p.sizing_mode = "scale_width"
    script3, div3 = components(p)

    Si = pawn.analyze(problem, X, y, S=10, print_to_console=False, seed=seed)
    df = Si.to_df()
    df = df.sort_values(by=['mean'], ascending=False)
    df.reset_index(inplace=True)
    dftop = df.iloc[:top]
    p = figure(x_range=dftop['index'], plot_height=300, plot_width=80*top, toolbar_location="right", title="Pawn", tools=plottools)
    p = ip.plot_pawn(dftop, p)
    p.sizing_mode = "scale_width"
    script4, div4 = components(p)

    return ([script1,script2,script3,script4], [div1,div2,div3,div4])
    

def morris_plt(problem, sample_size, fun, top=50, num_levels=4):
    X = sample(problem, N=sample_size, num_levels=num_levels)
    y =  np.asarray(list(map(fun, X)))
    Si = morris.analyze(problem, X, y, conf_level=0.95,
                    print_to_console=False, num_levels=num_levels)
    df = Si.to_df()
    df.reset_index(inplace=True)
    df = df.sort_values(by=['mu_star'], ascending=False)
    dftop = df.iloc[:top]

    #output_file(filename="template/morris1.html", title="Interactive plot of Morris")
    p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="Morris mu_star")
    p = ip.plot_errorbar(dftop, p, base_col="mu_star", error_col="mu_star_conf")
    p.sizing_mode = "scale_width"
    script1, div1 = components(p)
    #save(p)

    #output_file(filename="template/morris2.html", title="Covariance plot of Morris")
    p = ip.interactive_covariance_plot(df, top=top)
    p.sizing_mode = "scale_width"
    script2, div2 = components(p)
    #save(p)
    return ([script1,script2],[div1,div2])



def sobol_plt(problem, sample_size, fun, top=50, seed=42):
    # create sobol analysis
    X = saltelli.sample(problem, N=sample_size, calc_second_order=True)
    y =  np.asarray(list(map(fun, X)))

    sa = sobol.analyze(problem, y, print_to_console=False, seed=seed, calc_second_order=True)
    sa_dict = dp.format_salib_output(sa, "problem", pretty_names=None)

    #try interactive plot
    #output_file(filename="template/interactive1.html", title="Interactive plot of Sobol")
    #ip.interact_with_plot_all_outputs(sa_dict)
    #p = plot_all_outputs_mine(sa_dict, top=top, log_axis=False)
    p = ip.plot_dict(sa_dict['problem'], min_val=0, top=top, log_axis=True)
    p.sizing_mode = "scale_width"
    p.title="First order and total sensitivity"
    script1, div1 = components(p)
    #save(p)

    #output_file(filename="template/interactive2.html", title="Interactive plot of Sobol")
    #ip.interact_with_plot_all_outputs(sa_dict)
    p = ip.plot_second_order(sa_dict['problem'], top=top)
    p.sizing_mode = "scale_width"
    script2, div2 = components(p)
    #save(p)

    g = nt.build_graph(sa_dict['problem'], sens='ST', top=top, min_sens=0.01,
                       edge_cutoff=0.005)
    inline=True
    scale=200
    for i in range(g.num_vertices()):
        g.vp['sensitivity'][i] = (scale * g.vp['sensitivity'][i] )

    #epsens = g.edge_properties['second_sens']
    #for i in g.edges():
    #    g.ep['second_sens'][i] =  (g.ep['second_sens'][i])

    filename = "template/images/sobol.png"
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
                        output_size=(600, 600),
                        inline=inline,
                        output=filename
                        )
    filename = "template/images/sobol_full.png"
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
    return ([script1,script2],[div1,div2])

from benchmark import bbobbenchmarks as bn
dim = 50
problem = {
    'num_vars': dim,
    'names': ['X'+str(x) for x in range(dim)],
    'bounds': [[-5.0, 5.0]] * dim
    }
fun, opt = bn.instantiate(22, iinstance=1)
generate_report(problem, 500, fun, 10, seed=42)