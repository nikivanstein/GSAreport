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
import os
import os.path
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
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')
from graph_tool.all import *
from graph_tool import draw
from bokeh.plotting import output_file, save
import savvy.data_processing as dp
import plotting.interactive_plots as ip
from plotting.plotting import make_plot, make_second_order_heatmap, TS_CODE
import savvy.network_tools as nt
from bokeh.plotting import figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import webbrowser

class SAReport():

    def __init__(self, problem, top=20, name="SA experiment", seed=42):
        """Initialises the SAReport object.
        Parameters
        ----------
        problem : dict
            The problem definition
        top: integer
            The number of important features we are interested in
        name: string
            A name for the analysis experiment (used in the report)
        seed : integer
            The random seed.
        Examples
        --------
            >>> problem = {
                'num_vars': 3,
                'names': ['x1', 'x2', 'x3'],
                'bounds': [[-np.pi, np.pi]]*3
                }
            >>> report = SAReport(problem, "Test problem")
        """
        if top > problem['num_vars']:
            top = problem['num_vars']
        self.problem = problem
        self.top = top
        self.name = name
        self.seed = seed
        self.lhs = False
        self.sobol = False
        self.morris = False
    
    def generateSamples(self, sample_size=10000):
        """Generate samples for the different SA techniques.
        Parameters
        ----------
        sample_size : integer
            The number of samples you want to perform.
        Examples
        --------
            >>> report = SAReport(problem, "Test problem")
            >>> lhs,morris,sobol = report.generateSamples(500)
        """
        self.x_lhs = latin.sample(self.problem, sample_size, seed=self.seed)
        self.x_morris = sample(self.problem, sample_size, seed=self.seed)
        self.x_sobol = saltelli.sample(self.problem, sample_size)
        return (self.x_lhs, self.x_morris, self.x_sobol)

    def storeSamples(self, outputdir="data"):
        """Stores the generated samples
        Parameters
        ----------
        outputdir: string
            The location to store the csv files.
        Examples
        --------
            >>> report = SAReport(problem, "Test problem")
            >>> report.generateSamples(500)
            >>> report.storeSamples("data")
        """
        np.savetxt(f"{outputdir}/x_lhs.csv", self.x_lhs)
        np.savetxt(f"{outputdir}/x_morris.csv", self.x_morris)
        np.savetxt(f"{outputdir}/x_sobol.csv", self.x_sobol)

    def trainModel(self, X, y):
        """Tran a Random Forest regressor on real world data to generate different samples.
        Parameters
        ----------
        X: np.array
            Array with shape (instances, features)
        y: np.array
            Array with target values
        sample_size: integer
            Number of samples to take from the trained regressor.
        Examples
        --------
            >>> report = SAReport(problem, "Test problem")
            >>> r2 = report.trainModel(X,y)
            >>> print(r2)
        """
        self.regr = RandomForestRegressor(max_depth=2, random_state=self.seed, n_estimators=100)
        self.model_score = cross_val_score(self.regr, X, y, cv=3)
        self.regr.fit(X, y)
        self.model = True
        return self.model_score

    def sampleUsingModel(self, sample_size=1000):
        '''use the trained model to generate the samples for each SA method
        '''
        if self.model:
            self.x_lhs,self.x_morris,self.x_sobol = self.generateSamples(sample_size)
            self.y_lhs = self.regr.predict(self.x_lhs)
            self.y_morris = self.regr.predict(self.x_morris)
            self.y_sobol = self.regr.predict(self.x_sobol)
            self.morris = self.sobol = self.lhs = True


    def loadData(self, dir="data"):
        """Loads the X and y data generated by the sampling and by an external evaluation.

        Parameters
        ----------
        dir: string
            The location to load the csv files from.
            It expects the following combinations of csv files: 
                x.csv y.csv
                x_lhs.csv y_lhs.csv
                x_morris.csv y_morris.csv
                x_sobol.csv y_sobol.csv
            At least one of these file pairs should be present.
        Examples
        --------
            >>> report.loadData("data")
        """
        if (os.path.exists(f"{dir}/y.csv") and os.path.exists(f"{dir}/x.csv")):
            X = np.loadtxt(f"{dir}/x.csv")
            y = np.loadtxt(f"{dir}/y.csv")
            self.trainModel(X,y)
            self.sampleUsingModel()
        else:
            if (os.path.exists(f"{dir}/y_lhs.csv") and os.path.exists(f"{dir}/x_lhs.csv")):
                self.x_lhs = np.loadtxt(f"{dir}/x_lhs.csv")
                self.y_lhs = np.loadtxt(f"{dir}/y_lhs.csv")
                self.lhs = True
            if (os.path.exists(f"{dir}/y_sobol.csv") and os.path.exists(f"{dir}/x_sobol.csv")):
                self.x_sobol = np.loadtxt(f"{dir}/x_sobol.csv")
                self.y_sobol = np.loadtxt(f"{dir}/y_sobol.csv")
                self.sobol = True
            if (os.path.exists(f"{dir}/y_morris.csv") and os.path.exists(f"{dir}/x_morris.csv")):
                self.x_morris = np.loadtxt(f"{dir}/x_morris.csv")
                self.y_morris = np.loadtxt(f"{dir}/y_morris.csv")
                self.morris = True
            if (self.lhs):
                self.trainModel(self.x_lhs, self.y_lhs)
            elif (self.sobol):
                self.trainModel(self.x_sobol, self.y_sobol)
            elif (self.morris):
                self.trainModel(self.x_morris, self.y_morris)
            else:
                raise Exception("Pleaase provide at least one x and y csv file")
        

    def analyse(self):
        '''Perform the SA analysis steps and generate the report.
        '''

        if (self.lhs):
            lhs_scripts, lhs_divs = self._lhs_methods()
        else:
            lhs_scripts = ["", "","", ""]
            lhs_divs = ["", "","", ""]
        if (self.morris):
            morris_scripts, morris_divs, df = self._morris_plt(num_levels=4)
        else:
            morris_scripts = ["", ""]
            morris_divs = ["", ""]
        if (self.sobol):
            sobol_scripts, sobol_divs = self._sobol_plt()
        else:
            sobol_scripts = ["", ""]
            sobol_divs = ["", ""]
        if (self.model):
            surface_script, surface_div = self._surface_plot(df.index[0], df.index[1])
        else:
            surface_div = ""
            surface_script = ""

        file = open('template/index.html', mode='r')
        html_template = file.read()
        html_template = html_template.replace("#SURFACE#", surface_div)
        html_template = html_template.replace("#SURFACE_SCRIPT#", surface_script)

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
        webbrowser.open('file://' + os.path.realpath('template/report.html'))
        

    def _surface_plot(self, i=0, j=0):
        '''Generate a surface plot using the ith and jth parameter.
        '''
        p = ip.surface3dplot(problem, self.regr.predict, i, j)
        p.sizing_mode = "scale_width"
        return components(p)

    def _lhs_methods(self):
        plottools = "hover, wheel_zoom, save, reset," # , tap"
        X = self.x_lhs
        y =  self.y_lhs
        Si = rbd_fast.analyze(self.problem, X, y, print_to_console=False)
        top = self.top
        df = Si.to_df()
        df.reset_index(inplace=True)
        df = df.sort_values(by=['S1'], ascending=False)
        dftop = df.iloc[:top]
        p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="RDB Fast", tools=plottools)
        p = ip.plot_errorbar(dftop, p, base_col="S1", error_col="S1_conf")
        p.sizing_mode = "scale_width"
        script1, div1 = components(p)
        
        Si = delta.analyze(self.problem, X, y, print_to_console=False)
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

        Si = pawn.analyze(problem, X, y, S=10, print_to_console=False, seed=self.seed)
        df = Si.to_df()
        df = df.sort_values(by=['mean'], ascending=False)
        df.reset_index(inplace=True)
        dftop = df.iloc[:top]
        p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="Pawn", tools=plottools)
        p = ip.plot_pawn(dftop, p)
        p.sizing_mode = "scale_width"
        script4, div4 = components(p)

        return ([script1,script2,script3,script4], [div1,div2,div3,div4])
        

    def _morris_plt(self, num_levels=4):
        X = self.x_morris
        y =  self.y_morris
        top = self.top
        Si = morris.analyze(problem, X, y, conf_level=0.95,
                        print_to_console=False, num_levels=num_levels)
        df = Si.to_df()
        df.reset_index(inplace=True)
        df = df.sort_values(by=['mu_star'], ascending=False)
        dftop = df.iloc[:top]

        p = figure(x_range=dftop['index'], plot_height=300, plot_width=20*top, toolbar_location="right", title="Morris mu_star")
        p = ip.plot_errorbar(dftop, p, base_col="mu_star", error_col="mu_star_conf")
        p.sizing_mode = "scale_width"
        script1, div1 = components(p)

        p = ip.interactive_covariance_plot(df, top=top)
        p.sizing_mode = "scale_width"
        script2, div2 = components(p)
        return ([script1,script2],[div1,div2], dftop)



    def _sobol_plt(self):
        # create sobol analysis
        X = self.x_sobol
        y = self.y_sobol
        top = self.top

        sa = sobol.analyze(problem, y, print_to_console=False, seed=self.seed, calc_second_order=True)
        sa_dict = dp.format_salib_output(sa, "problem", pretty_names=None)

        p = ip.plot_dict(sa_dict['problem'], min_val=0, top=top, log_axis=True)
        p.sizing_mode = "scale_width"
        p.title="First order and total sensitivity"
        script1, div1 = components(p)

        p = ip.plot_second_order(sa_dict['problem'], top=top)
        p.sizing_mode = "scale_width"
        script2, div2 = components(p)

        g = nt.build_graph(sa_dict['problem'], sens='ST', top=top, min_sens=0.01,
                        edge_cutoff=0.005)
        inline=True
        scale=200
        for i in range(g.num_vertices()):
            g.vp['sensitivity'][i] = (scale * g.vp['sensitivity'][i] )

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

if __name__ == "__main__":
    from benchmark import bbobbenchmarks as bn
    dim = 50
    problem = {
        'num_vars': dim,
        'names': ['X'+str(x) for x in range(dim)],
        'bounds': [[-5.0, 5.0]] * dim
        }
    fun, opt = bn.instantiate(12, iinstance=1)
    report = SAReport(problem, top=10, name="F12")
    X, _, _ = report.generateSamples(10000)
    y =  np.asarray(list(map(fun, X)))

    report.trainModel(X, y)
    report.sampleUsingModel(5000)
    report.analyse()