# -*- coding: utf-8 -*-
"""GSA report.

This software can be used to automatically apply different global sensitivity analysis methods
to an existing data set, model or real-world process. 
GSAreport is an application to easily generate reports that describe the global sensitivities of your input parameters as best as possible. You can use the reporting application to inspect which features are important for a given real world function / simulator or model. Using the dockerized application you can generate a report with just one line of code and no additional dependencies (except for Docker of course).

Global Sensitivity Analysis is one of the tools to better understand your machine learning models or get an understanding in real-world processes.

Example:
    Define a problem definition, initialize the SAreport class, load the data and perform the analysis::

        $ problem = {
                'num_vars': 3,
                'names': ['x1', 'x2', 'x3'],
                'bounds': [[-np.pi, np.pi]]*3
                }
        $ report = SAReport(problem, "Test problem")
        $ report.loadData()
        $ report.analyse()
"""
import os
import os.path
import warnings
from cProfile import label

import numpy as np
from bokeh.embed import components
from SALib.analyze import delta, dgsm, fast, morris, pawn, rbd_fast, sobol
from SALib.plotting.bar import plot as barplot
from SALib.plotting.hdmr import plot as hdmrplot
from SALib.plotting.morris import (
    covariance_plot,
    horizontal_bar_plot,
    sample_histograms,
)
from SALib.sample import latin, saltelli
from SALib.sample.morris import sample
from SALib.util import read_param_file
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
use_graph_tool = False
try:
    from graph_tool import draw
    from graph_tool.all import *

    import plotting.network_tools as nt

    use_graph_tool = True
except ModuleNotFoundError:
    # Error handling (ignore and do not use)
    pass
except ImportError:
    pass

import json
import shutil
import textwrap
from datetime import datetime

import matplotlib.pyplot as pl
import pandas as pd
import shap
from bokeh.plotting import figure, output_file, save
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from plotting import data_processing as dp
from plotting import interactive_plots as ip
from plotting.plotting import TS_CODE, make_plot, make_second_order_heatmap


class SAReport:
    """SAReport object to generate samples, load samples and generate the sensitivity analysis report.
    Depending on the number of samples and dimensions of the problem the methods used might differ.
    The following methods are included in the report:
        * Sobol
        * Morris
        * Delta
        * PAWN
        * Random Forest
        * TreeSHAP
    If the number of dimensions exceeds 64, Sobol, Delta and Pawn are excluded because they do not perform well in such high dimensional spaces.
    If the number of samples per dimension is less than 50, Delta and Pawn are excluded due to their low performancee with small sample sizes.

    Args:
        problem (dict): The problem definition.
        top (int): The number of important features we are interested in.
        name (str): The name of the experiment.
        output_dir (str): The location where the report will be written to.
        data_dir (str): The directory path where the data can be loaded from.
            It expects the following combinations of csv files:
            either x.csv and y.csv or,
            x_lhs.csvm y_lhs.csv and
            x_morris.csv, y_morris.csv and
            x_sobol.csv, y_sobol.csv.
            At least one of these file pairs should be present.
        model_samples (int): The number of samples (per dim) generated using the Random Forest model.
        num_levels (int): The number of levels for the Morris method (default to 4).
        seed (int): random seed.
    """

    def __init__(
        self,
        problem,
        top=20,
        name="SA experiment",
        output_dir="output/",
        data_dir="data/",
        model_samples=1000,
        num_levels=4,
        seed=42,
    ):
        if top > problem["num_vars"]:
            top = problem["num_vars"]
        self.problem = problem
        self.top = top
        self.name = name
        self.seed = seed
        self.morris = False
        self.sobol = False
        self.delta = False
        self.pawn = False
        self.rbd_fast = False
        self.num_levels = num_levels
        self.output_dir = output_dir
        self.data_dir = data_dir
        now = datetime.now()
        self.start_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.tag = now.strftime("%Y-%m-%dT%H-%M")
        self.model_samples = model_samples
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(os.path.join(output_dir, "includes")):
            os.makedirs(os.path.join(output_dir, "includes"))
        if not os.path.exists(os.path.join(output_dir, "images")):
            os.makedirs(os.path.join(output_dir, "images"))

    def generateSamples(self, sample_size=10000):
        """Generate samples for the different SA techniques.

        Args:
            sample_size (int): The number of base samples (per dimension) to generate for each design of experiments.
                Note that for Morris sample_size * (dim+1) and for Sobol sample_size * (dim+2) are generated.

        Returns:
            list: A list containing the 3 design of experiments (3 times sample_size samples).

        Example:
            Generate 500*d samples per DOE.

                $ report = SAReport(problem, "Test problem")
                $ lhs,morris,sobol = report.generateSamples(500)
        """
        if sample_size > 50 or self.problem["num_vars"] < 64:
            self.x_lhs = latin.sample(
                self.problem, sample_size * self.problem["num_vars"], seed=self.seed
            )
        else:
            self.x_lhs = None
        self.x_morris = sample(self.problem, sample_size, seed=self.seed)
        if self.problem["num_vars"] < 64:
            self.x_sobol = saltelli.sample(self.problem, sample_size)
        else:
            self.x_sobol = None
        return (self.x_lhs, self.x_morris, self.x_sobol)

    def storeSamples(self):
        """Store the generated samples to csv files in the data dir.

        Example:
            Generate 500*d samples per DOE and store them in the data directory.

                $ report = SAReport(problem, "Test problem")
                $ report.generateSamples(500)
                $ report.storeSamples("data")
        """
        if self.x_lhs is not None:
            np.savetxt(f"{self.data_dir}/x_lhs.csv", self.x_lhs)
        np.savetxt(f"{self.data_dir}/x_morris.csv", self.x_morris)
        if self.x_sobol is not None:
            np.savetxt(f"{self.data_dir}/x_sobol.csv", self.x_sobol)

    def trainModel(self, X, y):
        """Train a Random Forest regressor on provided data to generate different samples.

        Args:
            X (array): Two dimensional array of instances and features.
            y (list): List with target values for X.

        Returns:
            float: The cross validation score of the RF model.

        Example:
            Train a model for a given input and output.

                $ report = SAReport(problem, "Test problem")
                $ r2 = report.trainModel(X,y)
        """
        self.regr = RandomForestRegressor(
            max_depth=5, random_state=self.seed, n_estimators=100
        )
        self.model_score = cross_val_score(self.regr, X, y, cv=3)
        self.regr.fit(X, y)
        self.X = X
        self.model = True
        return self.model_score

    def sampleUsingModel(self):
        """Use the trained model to generate the samples for each SA method."""
        if self.model:
            self.x_lhs, self.x_morris, self.x_sobol = self.generateSamples(
                self.model_samples
            )
            self.y_lhs = self.regr.predict(self.x_lhs)
            self.y_morris = self.regr.predict(self.x_morris)
            self.y_sobol = self.regr.predict(self.x_sobol)

            self.morris = self.sobol = self.delta = self.pawn = self.rbd_fast = True
            if self.problem["num_vars"] > 64:
                self.sobol = False
                self.delta = False
                self.pawn = False
            if len(self.X) / self.problem["num_vars"] < 50:
                self.delta = False
                self.pawn = False

    def tree_shap(self):
        """Generate the shap values and SHAP summary plot for the tree based model."""
        if self.model:
            explainer = shap.Explainer(self.regr, feature_names=self.problem["names"])
            shap_values = explainer(self.X)
            shap.plots.beeswarm(shap_values, max_display=self.top, show=False)
            loc = f"{self.output_dir}/images/{self.tag}shap.png"
            pl.gcf().axes[-1].set_aspect("auto")
            pl.gcf().axes[-1].set_box_aspect(50)
            pl.tight_layout()
            pl.savefig(loc)

    def loadData(self):
        """Loads the X and y data csv files generated by the sampling and by an external evaluation function.

        Example:
            Load the csv files from the data folder.

                $ report.loadData()
        """
        dir = self.data_dir
        if os.path.exists(f"{dir}/y.csv") and os.path.exists(f"{dir}/x.csv"):
            X = np.loadtxt(f"{dir}/x.csv")
            y = np.loadtxt(f"{dir}/y.csv")
            self.trainModel(X, y)
            self.sampleUsingModel()
        else:
            if os.path.exists(f"{dir}/y_lhs.csv") and os.path.exists(
                f"{dir}/x_lhs.csv"
            ):
                self.x_lhs = np.loadtxt(f"{dir}/x_lhs.csv")
                self.y_lhs = np.loadtxt(f"{dir}/y_lhs.csv")
                self.pawn = True
                self.delta = True
                self.rbd_fast = True
            if os.path.exists(f"{dir}/y_sobol.csv") and os.path.exists(
                f"{dir}/x_sobol.csv"
            ):
                self.x_sobol = np.loadtxt(f"{dir}/x_sobol.csv")
                self.y_sobol = np.loadtxt(f"{dir}/y_sobol.csv")
                self.sobol = True
            if os.path.exists(f"{dir}/y_morris.csv") and os.path.exists(
                f"{dir}/x_morris.csv"
            ):
                self.x_morris = np.loadtxt(f"{dir}/x_morris.csv")
                self.y_morris = np.loadtxt(f"{dir}/y_morris.csv")
                self.morris = True

            if self.sobol:
                self.trainModel(self.x_sobol, self.y_sobol)
            elif self.pawn or self.delta or self.rbd_fast:
                self.trainModel(self.x_lhs, self.y_lhs)
            elif self.morris:
                self.trainModel(self.x_morris, self.y_morris)
            else:
                raise Exception("Pleaase provide at least one x and y csv file")

    def analyse(self):
        """Perform the SA analysis steps and generate the report."""
        if self.model:
            importances = self.regr.feature_importances_
            feature_rank = np.argsort(importances)[::-1]
            surface_script, surface_div = self._surface_plot(
                feature_rank[0], feature_rank[1]
            )
            rf_script, rf_div = self._model_plot()
            self.tree_shap()
        else:
            surface_div = ""
            surface_script = ""
            rf_script, rf_div = "", ""
        if self.pawn or self.delta or self.rbd_fast:
            lhs_scripts, lhs_divs = self._lhs_methods()
        else:
            lhs_scripts = ["", "", "", ""]
            lhs_divs = ["", "", "", ""]
        if self.morris:
            morris_scripts, morris_divs, df = self._morris_plt()
        else:
            morris_scripts = ["", ""]
            morris_divs = ["", ""]
        if self.sobol:
            sobol_scripts, sobol_divs = self._sobol_plt()
        else:
            sobol_scripts = ["", ""]
            sobol_divs = ["", ""]

        # copy js/css/images template files
        src = os.path.join("template", "includes")
        files = os.listdir(src)
        dest = os.path.join(self.output_dir, "includes")
        for fname in files:
            shutil.copy2(os.path.join(src, fname), dest)

        file = open("template/index.html", mode="r")

        name1 = self.problem["names"][df.index[0]]
        name2 = self.problem["names"][df.index[1]]
        report_div = f"""
        Sensitivity analysis is the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be divided and allocated to different sources of uncertainty in its inputs.
        This experiments shows the first, second and total order sensitivities of the provided features (input) and visualises them in an interactive way. The different sensitivity algorithms can provide additional insight and information in the underlying process.<br/>
        <hr>
        Experiment: <strong>{self.name}</strong><br/>
        Started: <strong>{self.start_time}</strong><br/><br/>
        <hr>
        Number of parameters: {self.problem['num_vars']} <br/>
        Showing the top {self.top} parameters  <br/>
        Random Forest mean R<sup>2</sup> score over 3 folds: {np.mean(self.model_score)} <br/>
        """
        surface_text = f"Interactive surface plot of a slice (using a Random Forest model) with {name1} on the X axis and {name2} on the Y axis. All other parameters are set to the center of their range."
        html_template = file.read()
        # enable and disable sections

        if not self.sobol:
            html_template = html_template.replace(
                "SOBOL_SHOW", "\" style='display:none;'"
            )
        if not self.morris:
            html_template = html_template.replace(
                "MORRIS_SHOW", "\" style='display:none;'"
            )
        if not self.delta:
            html_template = html_template.replace(
                "DELTA_SHOW", "\" style='display:none;'"
            )
        if not self.pawn:
            html_template = html_template.replace(
                "PAWN_SHOW", "\" style='display:none;'"
            )
        if not self.rbd_fast:
            html_template = html_template.replace(
                "RBD_FAST_SHOW", "\" style='display:none;'"
            )

        html_template = html_template.replace("#EXPERIMENT_REPORT#", report_div)
        html_template = html_template.replace("#NAME#", self.name)

        html_template = html_template.replace("#SURFACE#", surface_div)
        html_template = html_template.replace("#SURFACETEXT#", surface_text)

        html_template = html_template.replace("#RF_DIV#", rf_div)
        html_template = html_template.replace("#RF_SCRIPT#", rf_script)

        html_template = html_template.replace("#SURFACE_SCRIPT#", surface_script)

        html_template = html_template.replace("#SOBOL1#", sobol_divs[0])
        html_template = html_template.replace("#SOBOL2#", sobol_divs[1])
        html_template = html_template.replace("#SOBOL_SCRIPT1#", sobol_scripts[0])
        html_template = html_template.replace("#SOBOL_SCRIPT2#", sobol_scripts[1])
        if use_graph_tool:
            html_template = html_template.replace("#SOBOL_NETWORK_SUPPORT#", "")
        else:
            html_template = html_template.replace(
                "#SOBOL_NETWORK_SUPPORT#", 'style="display:none;"'
            )

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
        html_template = html_template.replace("#TAG#", self.tag)
        with open(f"{self.output_dir}/{self.tag}-{self.name}-report.html", "w") as f:
            f.write(html_template)
        # webbrowser.open('file://' + os.path.realpath('template/report.html'))

    def _surface_plot(self, i=0, j=0):
        """Generate a surface plot using the ith and jth parameter."""
        p = ip.surface3dplot(problem, self.regr.predict, i, j)
        p.sizing_mode = "scale_width"
        return components(p)

    def _lhs_methods(self):
        plottools = "wheel_zoom, save, reset, tap,"  # , tap"
        X = self.x_lhs
        y = self.y_lhs
        top = self.top
        plot_width = max(200, 20 * top)
        plot_height = 200
        if top < 10:
            plot_height = 100
        if self.rbd_fast:
            Si = rbd_fast.analyze(self.problem, X, y, print_to_console=False)
            df = Si.to_df()
            df.reset_index(inplace=True)
            df = df.sort_values(by=["S1"], ascending=False)
            dftop = df.iloc[:top]
            p = figure(
                x_range=dftop["index"],
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location="right",
                title="RDB Fast",
                tools=plottools,
            )
            p = ip.plot_errorbar(
                dftop,
                p,
                base_col="S1",
                error_col="S1_conf",
                label_x="S1",
                label_y="S1 conf.",
            )
            p.sizing_mode = "scale_width"
            script1, div1 = components(p)
        else:
            script1 = div1 = ""

        if self.delta:
            Si = delta.analyze(self.problem, X, y, print_to_console=False)
            df = Si.to_df()
            df.reset_index(inplace=True)
            df = df.sort_values(by=["S1"], ascending=False)
            dftop = df.iloc[:top]
            p = figure(
                x_range=dftop["index"],
                plot_height=plot_height,
                plot_width=plot_width,
                toolbar_location="right",
                title="S1",
                tools=plottools,
            )
            p = ip.plot_errorbar(
                dftop,
                p,
                base_col="S1",
                error_col="S1_conf",
                label_x="S1",
                label_y="S1 conf.",
            )
            p.sizing_mode = "scale_width"
            script2, div2 = components(p)

            df = df.sort_values(by=["delta"], ascending=False)
            dftop = df.iloc[:top]
            p = figure(
                x_range=dftop["index"],
                plot_height=plot_height,
                plot_width=plot_width,
                title="Delta",
                toolbar_location="right",
                tools=plottools,
            )
            p = ip.plot_errorbar(
                dftop,
                p,
                base_col="delta",
                error_col="delta_conf",
                label_x="Delta",
                label_y="Delta conf.",
            )
            p.sizing_mode = "scale_width"
            script3, div3 = components(p)
        else:
            script2 = div2 = ""
            script3 = div3 = ""

        if self.pawn:
            Si = pawn.analyze(
                self.problem, X, y, S=10, print_to_console=False, seed=self.seed
            )
            df = Si.to_df()
            df = df.sort_values(by=["median"], ascending=False)
            df.reset_index(inplace=True)
            dftop = df.iloc[:top]
            # p = figure(x_range=dftop['index'], plot_height=200, plot_width=20*top, toolbar_location="right", title="Pawn", tools=plottools)
            p = ip.plot_pawn(dftop, p)
            p.sizing_mode = "scale_width"
            script4, div4 = components(p)
        else:
            script4 = div4 = ""

        return ([script1, script2, script3, script4], [div1, div2, div3, div4])

    def _model_plot(self):
        # Generate the random forest feature importance plot

        importances = self.regr.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in self.regr.estimators_], axis=0
        )

        ranks = np.argsort(importances)[::-1]
        top_importances = importances[ranks[: self.top]]
        top_std = std[ranks[: self.top]]
        feature_names = np.array(self.problem["names"])[ranks[: self.top]]
        dftop = pd.DataFrame(
            data={"S1": top_importances, "std": top_std, "index": feature_names}
        )
        plot_width = max(400, 20 * self.top)
        plot_height = 150
        if self.top < 10:
            plot_height = 100
        p = figure(
            x_range=dftop["index"],
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location="right",
            title="RF feature importances",
        )
        p = ip.plot_errorbar(
            dftop,
            p,
            base_col="S1",
            error_col="std",
            label_x="Importance",
            label_y="Std.",
        )
        p.sizing_mode = "scale_width"
        return components(p)

    def _morris_plt(self):
        # FIX sigma mu on plot -- add labels
        X = self.x_morris
        y = self.y_morris
        top = self.top
        Si = morris.analyze(
            self.problem,
            X,
            y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=self.num_levels,
        )
        df = Si.to_df()
        df.reset_index(inplace=True)
        df = df.sort_values(by=["mu_star"], ascending=False)
        dftop = df.iloc[:top]

        plot_width = max(400, 20 * len(df))
        plot_height = 150
        if len(df) < 10:
            plot_height = 100
        p = figure(
            x_range=df["index"],
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location="right",
            title="Morris mu_star",
        )
        p = ip.plot_errorbar_morris(
            df, p, base_col="mu_star", error_col="mu_star_conf", top=top
        )
        p.sizing_mode = "scale_width"
        script1, div1 = components(p)

        p = ip.interactive_covariance_plot(df, top=top)
        p.sizing_mode = "scale_width"
        script2, div2 = components(p)
        return ([script1, script2], [div1, div2], dftop)

    def _sobol_plt(self):
        global use_graph_tool
        # create sobol analysis
        X = self.x_sobol
        y = self.y_sobol
        top = self.top

        sa = sobol.analyze(
            self.problem,
            y,
            print_to_console=False,
            seed=self.seed,
            calc_second_order=True,
        )
        sa_dict = dp.format_salib_output(sa, "problem", pretty_names=None)

        p = ip.plot_dict(sa_dict["problem"], min_val=0, top=top, log_axis=True)
        p.sizing_mode = "scale_width"
        p.title = "First order and total sensitivity"
        script1, div1 = components(p)

        p = ip.plot_second_order(sa_dict["problem"], top=top)
        p.sizing_mode = "scale_width"
        script2, div2 = components(p)
        if use_graph_tool:
            g = nt.build_graph(sa_dict["problem"], sens="ST", top=top)
            inline = True
            scale = 200
            for i in range(g.num_vertices()):
                g.vp["sensitivity"][i] = scale * g.vp["sensitivity"][i]

            filename = f"{self.output_dir}/images/{self.tag}sobol.png"
            state = graph_tool.inference.minimize_nested_blockmodel_dl(g)
            draw.draw_hierarchy(
                state,
                vertex_text=g.vp["param"],
                vertex_text_position="centered",
                layout="radial",
                hide=0,
                # vertex_text_color='black',
                vertex_font_size=12,
                vertex_size=g.vp["sensitivity"],
                # vertex_color='#006600',
                # vertex_fill_color='#008800',
                vertex_halo=True,
                vertex_halo_color="#b3c6ff",
                vertex_halo_size=g.vp["confidence"],
                edge_pen_width=g.ep["second_sens"],
                # subsample_edges=100,
                output_size=(600, 600),
                inline=inline,
                output=filename,
            )
            filename = f"{self.output_dir}/images/{self.tag}sobol_full.png"
            draw.draw_hierarchy(
                state,
                vertex_text=g.vp["param"],
                vertex_text_position="centered",
                layout="radial",
                hide=0,
                # vertex_text_color='black',
                vertex_font_size=12,
                vertex_size=g.vp["sensitivity"],
                # vertex_color='#006600',
                # vertex_fill_color='#008800',
                vertex_halo=True,
                vertex_halo_color="#b3c6ff",
                vertex_halo_size=g.vp["confidence"],
                edge_pen_width=g.ep["second_sens"],
                # subsample_edges=100,
                output_size=(1600, 1600),
                inline=inline,
                output=filename,
            )
        return ([script1, script2], [div1, div2])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="GSAreport",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Generate a global sensitivity analysis report for a given data set or function.
            Common uses cases:
            --------------------------------
            Generate samples for evaluation by a real world function / simulator
                > python GSAreport.py -p problem.json -d data_dir --sample --samplesize 1000
            Analyse the samples with their output stored in the data folder
                > python GSAreport.py -p problem.json -d data_dir -o output_dir
            Analyse a real-world data set and use a Random Forest model to interpolate (data_dir contains x.csv and y.csv)
                > python GSAreport.py -p problem.json -d data_dir -o output_dir --samplesize 10000
            """
        ),
    )
    parser.add_argument(
        "--problem",
        "-p",
        default="problem.json",
        type=str,
        help="File path to the problem definition in json format.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="/data/",
        help="Directory where the intermediate data is stored. (defaults to /data)",
    )  # will be accesible under args.data
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="/output/",
        help="Directory where the output report is stored. (defaults to /output/)",
    )  # will be accesible under args.output
    parser.add_argument(
        "--name",
        type=str,
        default="SA",
        help="Name of the experiment, will be used in the report output.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="The number of important features to focus on, default is 10.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="When you use this flag, only the samples are generated to be used in the analyse phase.",
    )
    parser.add_argument(
        "--samplesize",
        "-n",
        type=int,
        help="Number of samples to generate.",
        default=1000,
    )
    parser.add_argument(
        "--modelsize", type=int, help="Number of samples for the model.", default=1000
    )
    parser.add_argument(
        "--demo",
        help="Demo mode, uses a BBOB function as test function.",
        action="store_true",
    )
    args = parser.parse_args()

    output_dir = args.output
    data_dir = args.data
    top = args.top

    problem = {}
    if not args.demo:
        with open(args.problem) as json_file:
            problem = json.load(json_file)
            print("loaded problem definition")
        with tqdm(total=100) as pbar:
            report = SAReport(
                problem,
                top=top,
                name=args.name,
                output_dir=output_dir,
                data_dir=data_dir,
                model_samples=args.modelsize,
            )
            pbar.update(10)
            if args.sample:
                # generate only samples
                report.generateSamples(args.samplesize)
                pbar.update(40)
                report.storeSamples()
                pbar.update(40)
                pbar.close()
            else:
                report.loadData()
                pbar.update(50)
                report.analyse()
                pbar.update(40)
                pbar.close()
    else:
        from benchmark import bbobbenchmarks as bn

        dim = 20
        problem = {
            "num_vars": dim,
            "names": ["X" + str(x) for x in range(dim)],
            "bounds": [[-5.0, 5.0]] * dim,
        }
        with open("problem.json", "w") as outfile:
            json.dump(problem, outfile)
        fun, opt = bn.instantiate(5, iinstance=1)
        report = SAReport(
            problem,
            top=10,
            name="F5",
            output_dir=output_dir,
            data_dir=data_dir,
            model_samples=5000,
        )
        X_lhs, X_morris, X_sobol = report.generateSamples(200)

        if not os.path.exists(f"{data_dir}/y_lhs.csv"):
            report.storeSamples()
            if X_lhs is not None:
                y_lhs = np.asarray(list(map(fun, X_lhs)))
            y_morris = np.asarray(list(map(fun, X_morris)))
            if X_sobol is not None:
                y_sobol = np.asarray(list(map(fun, X_sobol)))
            if X_lhs is not None:
                np.savetxt(f"{data_dir}/y_lhs.csv", y_lhs)
            np.savetxt(f"{data_dir}/y_morris.csv", y_morris)
            if X_sobol is not None:
                np.savetxt(f"{data_dir}/y_sobol.csv", y_sobol)

        report.loadData()
        report.analyse()
