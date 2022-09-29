"""
This modules adds interactivity to plots in plotting.py through Bokeh tabs and
ipython widgets.

Dependencies:
plotting.py
data_processing.py
matplotlib
numpy
pandas
os
bokeh
ipywidgets
collections

"""
from __future__ import absolute_import, division, print_function

import warnings

from bokeh.io import curdoc
from bokeh.models import LabelSet, Whisker
from bokeh.models.widgets import Panel, Tabs
from bokeh.palettes import GnBu3, OrRd3
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.transform import factor_cmap
from ipywidgets import BoundedFloatText, Checkbox, IntText, SelectMultiple

from .plotting import make_plot, make_second_order_heatmap

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from bokeh.models import FactorRange, HoverTool, VBar


def plot_errorbar(
    df, p, base_col="mu_star", error_col="mu_star_conf", label_x="ST", label_y="ST conf"
):
    # plot an errorbar using the figure
    upper = df[base_col] + df[error_col]
    lower = df[base_col] - df[error_col]

    source = ColumnDataSource(
        data=dict(
            groups=df["index"],
            counts=df[base_col],
            muconf=df[error_col],
            upper=upper,
            lower=lower,
        )
    )
    p.vbar(x="groups", top="counts", width=0.9, source=source, line_color="white")

    p.add_layout(
        Whisker(
            source=source, base="groups", upper="upper", lower="lower", level="overlay"
        )
    )
    p.add_tools(
        HoverTool(
            tooltips=[
                ("Parameter", "@groups"),
                (label_x, "@counts"),
                (label_y, "@counts"),
            ]
        )
    )

    # p.xaxis.ticker = df.index
    p.legend.visible = False
    p.toolbar.autohide = True
    return p


def plot_errorbar_morris(df, p, base_col="mu_star", error_col="mu_star_conf", top=10):
    # plot an errorbar using the figure
    upper = df[base_col] + df[error_col]
    lower = df[base_col] - df[error_col]

    color_list = []
    for i in range(len(df)):
        if i < top:
            color_list.append("#2171b5")
        else:
            color_list.append("#c6dbef")

    source = ColumnDataSource(
        data=dict(
            groups=df["index"],
            color=color_list,
            counts=df[base_col],
            muconf=df[error_col],
            sigma=df["sigma"],
            upper=upper,
            lower=lower,
        )
    )
    p.vbar(
        x="groups",
        top="counts",
        width=0.9,
        source=source,
        line_color="white",
        color="color",
    )

    p.add_layout(
        Whisker(
            source=source, base="groups", upper="upper", lower="lower", level="overlay"
        )
    )
    p.add_tools(
        HoverTool(
            tooltips=[
                ("", "@groups"),
                ("μ*", "@counts"),
                ("μ* conf", "@muconf"),
                ("σ", "@sigma"),
            ]
        )
    )

    # p.xaxis.ticker = df.index
    p.legend.visible = False
    p.toolbar.autohide = True
    return p


def plot_pawn(df, p):
    # plot the pawn analysis

    # colors = [ "#4292c6", "#2171b5", "#08306b"]
    # p.vbar_stack(['minimum', 'median', 'maximum'], x="index", source = df, line_color='white', color=colors ,width = 0.5)

    parameters = df["index"].values
    groups = ["min", "med", "max"]
    x = [(param, group) for param in parameters for group in groups]
    counts = sum(zip(df["minimum"], df["median"], df["maximum"]), ())  # like an hstack
    source = ColumnDataSource(data=dict(x=x, counts=counts))

    p = figure(
        x_range=FactorRange(*x),
        height=200,
        title="PAWN Analysis",
        toolbar_location="right",
        tools="save,reset",
    )

    p.vbar(
        x="x",
        top="counts",
        width=0.9,
        source=source,
        line_color="white",
        # use the palette to colormap based on the the x[1:2] values
        fill_color=factor_cmap("x", palette=GnBu3, factors=groups, start=1, end=2),
    )

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    p.add_tools(HoverTool(tooltips="@x: @counts"))
    # p.xaxis.ticker = df.index
    # p.legend.visible = False
    p.toolbar.autohide = True
    return p


from .plotting import TS_CODE, Surface3d


def surface3dplot(problem, fun, x_i, y_i):
    # Plots an interactive 3d plot using fun.
    x_bound = problem["bounds"][x_i]
    y_bound = problem["bounds"][y_i]
    x_name = problem["names"][x_i]
    y_name = problem["names"][y_i]
    x = np.linspace(x_bound[0], x_bound[1], 50)
    y = np.linspace(y_bound[0], y_bound[1], 50)
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()

    X = None
    # create correct shape
    for i in range(len(problem["bounds"])):
        mid = (problem["bounds"][i][0] + problem["bounds"][i][1]) / 2
        if X is None:
            if i == x_i:
                X = xx
            elif i == y_i:
                X = yy
            else:
                X = np.ones(xx.shape) * mid
        else:
            if i == x_i:
                X = np.column_stack((X, xx))
            elif i == y_i:
                X = np.column_stack((X, yy))
            else:
                X = np.column_stack((X, np.ones(xx.shape) * mid))
    z = fun(X)
    data = dict(x=xx, y=yy, z=z)
    source = ColumnDataSource(data=data)

    p = Surface3d(x="x", y="y", z="z", data_source=source, xLabel=x_name, yLabel=y_name)
    return p


def interactive_covariance_plot(df, top=10):
    """Plots mu* against sigma

    Parameters
    -----------
    df                   : dataframe
                             a dataframe with one sensitibity analysis result.
    top                   : integer, optional
                             highlight the top highest mu_star parameters
    """

    hover = HoverTool(
        tooltips=[
            ("", "@desc"),
            ("μ*", "@x"),
            ("σ", "@y"),
        ]
    )
    p = figure(
        plot_height=500,
        plot_width=500,
        toolbar_location="right",
        title="Morris Covariance plot",
        tools=[hover, "save", "pan"],
        x_axis_label="μ*",
        y_axis_label="σ",
    )

    source = ColumnDataSource(
        data=dict(x=df["mu_star"].values, y=df["sigma"].values, desc=df["index"].values)
    )
    p.circle("x", "y", size=6, color="#c6dbef", source=source)

    # highlight the top x
    dftop = df.iloc[:top]
    sourceTop = ColumnDataSource(
        data=dict(
            x=dftop["mu_star"].values,
            y=dftop["sigma"].values,
            desc=dftop["index"].values,
        )
    )
    p.circle("x", "y", size=8, color="#2171b5", source=sourceTop)

    # labels = LabelSet(x='x', y='y', text='desc',
    #          x_offset=0, y_offset=0, source=sourceTop, render_mode='canvas')

    # p.add_layout(labels)
    x_axis_bounds = np.array([0, max(dftop["mu_star"].values) + 0.002])
    p.line(
        x_axis_bounds,
        x_axis_bounds,
        legend_label="σ / μ* = 1.0",
        line_width=2,
        color="black",
    )
    p.line(
        x_axis_bounds,
        0.5 * x_axis_bounds,
        legend_label="σ / μ* = 0.5",
        line_width=1,
        color="orange",
    )
    p.line(
        x_axis_bounds,
        0.1 * x_axis_bounds,
        legend_label="σ / μ* = 0.1",
        line_width=1,
        color="red",
    )
    p.legend.location = "top_left"
    p.toolbar.autohide = True
    return p


def plot_dict(
    sa_df,
    min_val=0,
    top=100,
    stacked=True,
    error_bars=True,
    log_axis=True,
    highlighted_parameters=[],
):
    """
    This function calls plotting.make_plot() for one of the sensitivity
    analysis output files and does not use tabs.

    Parameters
    -----------
    sa_df                   : dataframe
                             a dataframe with one sensitibity analysis result.
    demo                   : bool, optional
                             plot only two outcomes instead of all outcomes
                             for demo purpose.
    min_val                : float, optional
                             a float indicating the minimum sensitivity value
                             to be shown.
    top                    : int, optional
                             integer indicating the number of parameters to
                             display (highest sensitivity values).
    stacked                : bool, optional
                             Boolean indicating in bars should be stacked for
                             each parameter.
    error_bars             : bool, optional
                             Boolean indicating if error bars are shown (True)
                             or are omitted (False).
    log_axis               : bool, optional
                             Boolean indicating if log axis should be used
                             (True) or if a linear axis should be used (False).
    highlighted_parameters : list, optional
                             List of strings indicating which parameter wedges
                             will be highlighted.

    Returns
    --------
    p : bokeh plot
        a Bokeh plot generated with plotting.make_plot() that includes tabs
        for all the possible outputs.
    """

    p = make_plot(
        sa_df[0],
        top=top,
        minvalues=min_val,
        stacked=stacked,
        errorbar=error_bars,
        lgaxis=log_axis,
        highlight=highlighted_parameters,
    )
    return p


def plot_all_outputs(
    sa_dict,
    demo=False,
    min_val=0.01,
    top=100,
    stacked=True,
    error_bars=True,
    log_axis=True,
    highlighted_parameters=[],
):
    """
    This function calls plotting.make_plot() for all the sensitivity
    analysis output files and lets you choose which output to view
    using tabs.

    Parameters
    -----------
    sa_dict                : dict
                             a dictionary with all the sensitivity analysis
                             results.
    min_val                : float, optional
                             a float indicating the minimum sensitivity value
                             to be shown.
    top                    : int, optional
                             integer indicating the number of parameters to
                             display (highest sensitivity values).
    stacked                : bool, optional
                             Boolean indicating in bars should be stacked for
                             each parameter.
    error_bars             : bool, optional
                             Boolean indicating if error bars are shown (True)
                             or are omitted (False).
    log_axis               : bool, optional
                             Boolean indicating if log axis should be used
                             (True) or if a linear axis should be used (False).
    highlighted_parameters : list, optional
                             List of strings indicating which parameter wedges
                             will be highlighted.

    Returns
    --------
    p : bokeh plot
        a Bokeh plot generated with plotting.make_plot()
    """

    tabs_dictionary = {}
    outcomes_array = []
    if demo:
        for files in sa_dict.keys()[0:2]:
            outcomes_array.append(sa_dict[files][0])
    else:
        for files in sa_dict.keys():
            outcomes_array.append(sa_dict[files][0])

    for i in range(len(outcomes_array)):
        p = make_plot(
            outcomes_array[i],
            top=top,
            minvalues=min_val,
            stacked=stacked,
            errorbar=error_bars,
            lgaxis=log_axis,
            highlight=highlighted_parameters,
        )
        tabs_dictionary[i] = Panel(child=p, title=list(sa_dict.keys())[i])

    tabs = Tabs(tabs=list(tabs_dictionary.values()))
    p = show(tabs)

    return p


def plot_all_second_order(sa_dict, top=5, mirror=True, include=[]):
    """
    This function calls plotting.make_second_order_heatmap() for all the
    sensitivity analysis output files and lets you choose which output to view
    using tabs

    Parameters
    -----------
    sa_dict : dict
              a dictionary with all the sensitivity analysis results.
    top     : int, optional
              the number of parameters to display
              (highest sensitivity values).
    include : list, optional
              a list of parameters you would like to include even if they
              are not in the top `top` values.

    Returns
    --------
    p : bokeh plot
        a Bokeh plot that includes tabs for all the possible outputs.
    """

    tabs_dictionary = {}
    outcomes_array = []

    for files in sa_dict.keys():
        outcomes_array.append(sa_dict[files][1])

    for i in range(len(sa_dict)):
        p = make_second_order_heatmap(
            outcomes_array[i], top=top, mirror=mirror, include=include
        )
        tabs_dictionary[i] = Panel(child=p, title=list(sa_dict.keys())[i])

    tabs = Tabs(tabs=list(tabs_dictionary.values()))
    p = show(tabs)

    return p


def plot_second_order(sa_df, top=5, mirror=True, include=[]):
    """
    This function calls plotting.make_second_order_heatmap() for one
    sensitivity analysis output file.

    Parameters
    -----------
    sa_df :   dataframe
              a dictionary with one of the sensitivity analysis results.
    top     : int, optional
              the number of parameters to display
              (highest sensitivity values).
    include : list, optional
              a list of parameters you would like to include even if they
              are not in the top `top` values.

    Returns
    --------
    p : bokeh plot
    """

    p = make_second_order_heatmap(sa_df[1], top=top, mirror=mirror, include=include)

    return p


import warnings

warnings.filterwarnings("ignore")
