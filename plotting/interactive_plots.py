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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import show
from bokeh.io import curdoc
from ipywidgets import BoundedFloatText, IntText, Checkbox, SelectMultiple
from IPython.html.widgets import interact, fixed
from bokeh.models import Whisker
from bokeh.plotting import figure, ColumnDataSource
from .plotting import make_plot, make_second_order_heatmap
from bokeh.transform import factor_cmap
import warnings; warnings.filterwarnings('ignore')
import numpy as np

def plot_errorbar(df, p, base_col="mu_star", error_col="mu_star_conf"):
    #plot an errorbar using the figure
    upper = df[base_col] + df[error_col]
    lower = df[base_col] - df[error_col]

    source = ColumnDataSource(data=dict(groups=df['index'], counts=df[base_col], upper=upper, lower=lower))
    p.vbar(x='groups', top='counts', width=0.9, source=source, line_color='white')

    p.add_layout(
        Whisker(source=source, base="groups", upper="upper", lower="lower", level="overlay")
    )
    #p.xaxis.ticker = df.index
    p.legend.visible = False
    p.toolbar.autohide = True
    return p


def interactive_covariance_plot(ax, Si, opts=None, unit=""):
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

def plot_dict(sa_df, min_val=0, top=100, stacked=True,
                     error_bars=True, log_axis=True,
                     highlighted_parameters=[]):
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


    p = make_plot(sa_df[0],
                top=top,
                minvalues=min_val,
                stacked=stacked,
                errorbar=error_bars,
                lgaxis=log_axis,
                highlight=highlighted_parameters
                )
    return p

def plot_all_outputs(sa_dict, demo=False, min_val=0.01, top=100, stacked=True,
                     error_bars=True, log_axis=True,
                     highlighted_parameters=[]):
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
        p = make_plot(outcomes_array[i],
                      top=top,
                      minvalues=min_val,
                      stacked=stacked,
                      errorbar=error_bars,
                      lgaxis=log_axis,
                      highlight=highlighted_parameters
                      )
        tabs_dictionary[i] = Panel(child=p, title=list(sa_dict.keys())[i])

    tabs = Tabs(tabs=list(tabs_dictionary.values()))
    p = show(tabs)

    return p


def interact_with_plot_all_outputs(sa_dict, demo=False, manual=True):
    """
    This function adds the ability to interactively adjust all of the
    plotting.make_plot() arguments.

    Parameters
    ----------
    sa_dict : dict
              a dictionary with all the sensitivity analysis results.
    demo    : bool, optional
              plot only few outcomes for demo purpose.

    Returns
    -------
    Interactive widgets to control plot
    """
    min_val_box = BoundedFloatText(value=0.01, min=0, max=1,
                                   description='Min value:')
    top_box = IntText(value=20, description='Show top:')
    stacks = Checkbox(description='Show stacked plots:', value=True,)
    error_bars = Checkbox(description='Show error bars:', value=True)
    log_axis = Checkbox(description='Use log axis:', value=True)

    # get a list of all the parameter options
    key = list(sa_dict.keys())[0]

    # get a list of the options (supports old and new salib format)
    try:
        param_options = list(sa_dict[key][0].Parameter.values)
    except AttributeError:
        param_options = list(sa_dict[key][0].index.values)

    highlighted = SelectMultiple(description="Choose parameters to highlight",
                                 options=param_options, value=[])

    return interact(plot_all_outputs,
                    sa_dict=fixed(sa_dict),
                    demo = fixed(demo),
                    min_val=min_val_box,
                    top=top_box,
                    stacked=stacks,
                    error_bars=error_bars,
                    log_axis=log_axis,
                    highlighted_parameters=highlighted,
                    __manual=manual
                    )


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
        p = make_second_order_heatmap(outcomes_array[i],
                                      top=top,
                                      mirror=mirror,
                                      include=include)
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



    p = make_second_order_heatmap(sa_df[1],
                                top=top,
                                mirror=mirror,
                                include=include)

    return p

import warnings; warnings.filterwarnings('ignore')
