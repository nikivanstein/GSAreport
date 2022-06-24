"""
This module creates plots for visualizing sensitivity analysis dataframes.

`make_plot()` creates a radial plot of the first and total order indices.

`make_second_order_heatmap()` creates a square heat map showing the second
order interactions between model parameters.

"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import pandas_bokeh
from bokeh.models import HoverTool, VBar
from bokeh.plotting import ColumnDataSource, figure


def make_plot(
    dataframe=pd.DataFrame(),
    highlight=[],
    top=100,
    minvalues=0.01,
    stacked=True,
    lgaxis=True,
    errorbar=True,
    showS1=True,
    showST=True,
):
    """
    Basic method to plot first and total order sensitivity indices.

    This is the method to generate a Bokeh plot similar to the burtin example
    template at the Bokeh website. For clarification, parameters refer to an
    input being measured (Tmax, C, k2, etc.) and stats refer to the 1st or
    total order sensitivity index.

    Parameters
    -----------
    dataframe  : pandas dataframe
                 Dataframe containing sensitivity analysis results to be
                 plotted.
    highlight  : lst, optional
                 List of strings indicating which parameter wedges will be
                 highlighted.
    top        : int, optional
                 Integer indicating the number of parameters to display
                 (highest sensitivity values) (after minimum cutoff is
                 applied).
    minvalues  : float, optional
                 Cutoff minimum for which parameters should be plotted.
                 Applies to total order only.
    stacked    : bool, optional
                 Boolean indicating in bars should be stacked for each
                 parameter (True) or unstacked (False).
    lgaxis     : bool, optional
                 Boolean indicating if log axis should be used (True) or if a
                 linear axis should be used (False).
    errorbar   : bool, optional
                 Boolean indicating if error bars are shown (True) or are
                 omitted (False).
    showS1     : bool, optional
                 Boolean indicating whether 1st order sensitivity indices
                 will be plotted (True) or omitted (False).
    showST     : bool, optional
                 Boolean indicating whether total order sensitivity indices
                 will be plotted (True) or omitted (False).

                 **Note if showS1 and showST are both false, the plot will
                 default to showing ST data only instead of a blank plot**

    Returns
    --------
    p : bokeh figure
        A Bokeh figure of the data to be plotted
    """

    df = dataframe
    top = int(top)
    # Initialize boolean checks and check dataframe structure
    if (
        ("S1" not in df)
        or ("ST" not in df)
        or ("Parameter" not in df)
        or ("ST_conf" not in df)
        or ("S1_conf" not in df)
    ):
        raise Exception("Dataframe not formatted correctly")

    # Remove rows which have values less than cutoff values
    df = df[df["ST"] > minvalues]
    df = df.dropna()

    # Only keep top values indicated by variable top
    df = df.sort_values("ST", ascending=False)
    df = df.head(top)
    df = df.reset_index(drop=True)

    # Create arrays of colors and order labels for plotting
    colors = ["#c6dbef", "#4292c6", "#2171b5", "#08306b"]

    # colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
    #          "#4292c6", "#2171b5", "#08519c", "#08306b"]

    s1color = np.array(["#c6dbef"] * df.S1.size)
    sTcolor = np.array(["#4292c6"] * df.ST.size)
    errs1color = np.array(["#2171b5"] * df.S1.size)
    errsTcolor = np.array(["#08306b"] * df.ST.size)
    firstorder = np.array(["1st (S1)"] * df.S1.size)
    totalorder = np.array(["Total (ST)"] * df.S1.size)

    # Add column indicating which parameters should be highlighted
    tohighlight = df.Parameter.isin(highlight)
    df["highlighted"] = tohighlight

    back_color = {
        True: "#aeaeb8",
        False: "#ffffff",
    }

    # Switch to bar chart if dataframe shrinks below 5 parameters
    if len(df) < 5:
        # tools enabled for bokeh figure
        p = df.plot_bokeh.bar(
            x="Parameter", stacked=stacked, alpha=0.6, show_figure=False
        )

        return p

    # Create Dictionary of colors
    stat_color = OrderedDict()
    error_color = OrderedDict()
    for i in range(0, 2):
        stat_color[i] = colors[i]
    # Reset index of dataframe.
    for i in range(2, 4):
        error_color[i] = colors[i]

    # Sizing parameters
    width = 800
    height = 800
    inner_radius = 90
    outer_radius = 300 - 10

    # Determine wedge size based off number of parameters
    big_angle = 2.0 * np.pi / (len(df) + 1)
    # Determine division of wedges for plotting bars based on # stats plotted
    # for stacked or unstacked bars
    if stacked is False:
        small_angle = big_angle / 5
    else:
        small_angle = big_angle / 3
    # tools enabled for bokeh figure
    plottools = "hover, wheel_zoom, save, reset,"  # , tap"
    # Initialize figure with tools, coloring, etc.
    p = figure(
        plot_width=width,
        plot_height=height,
        title="",
        x_axis_type=None,
        y_axis_type=None,
        x_range=(-350, 350),
        y_range=(-350, 350),
        min_border=0,
        outline_line_color="#ffffff",
        background_fill_color="#ffffff",
        border_fill_color="#ffffff",
        toolbar_location="right",
        tools=plottools,
    )
    p.toolbar.autohide = True
    # Specify labels for hover tool
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Order", "@Order"),
        ("Parameter", "@Param"),
        ("Sensitivity", "@Sens"),
        ("Confidence", "@Conf"),
    ]
    hover.point_policy = "follow_mouse"

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # annular wedges divided into smaller sections for bars
    # Angles for axial line placement
    num_lines = np.arange(0, len(df) + 1, 1)
    line_angles = np.pi / 2 - big_angle / 2 - num_lines * big_angle

    # Angles for data placement
    angles = np.pi / 2 - big_angle / 2 - df.index.to_series() * big_angle

    # circular axes and labels
    minlabel = min(
        round(np.log10(max(min(df.ST), 0.0000001))),
        round(np.log10(max(0.00000001, min(df.S1)))),
    )
    labels = np.power(10.0, np.arange(0, minlabel - 1, -1))

    # Set max radial line to correspond to 1.1 * maximum value + error
    maxvalST = max(df.ST + df.ST_conf)
    maxvalS1 = max(df.S1 + df.S1_conf)
    maxval = max(maxvalST, maxvalS1)
    labels = np.append(labels, 0.0)
    labels[0] = round(1.1 * maxval, 1)

    # Determine if radial axis are log or linearly scaled
    if lgaxis is True:
        radii = ((np.log10(labels / labels[0])) + labels.size) * (
            outer_radius - inner_radius
        ) / labels.size + inner_radius
        radii[-1] = inner_radius
    else:
        labels = np.delete(labels, -2)
        radii = (outer_radius - inner_radius) * labels / labels[0] + inner_radius

    # Convert sensitivity values to the plotted values
    # Same conversion as for the labels above
    # Also calculate the angle to which the bars are placed
    # Add values to the dataframe for future reference
    cols = np.array(["S1", "ST"])
    for statistic in range(0, 2):
        if lgaxis is True:
            radius_of_stat = (
                (np.log10(df[cols[statistic]] / labels[0])) + labels.size
            ) * (outer_radius - inner_radius) / labels.size + inner_radius
            lower_of_stat = (
                (
                    np.log10(
                        (df[cols[statistic]] - df[cols[statistic] + "_conf"])
                        / labels[0]
                    )
                )
                + labels.size
            ) * (outer_radius - inner_radius) / labels.size + inner_radius
            higher_of_stat = (
                (
                    np.log10(
                        (df[cols[statistic]] + df[cols[statistic] + "_conf"])
                        / labels[0]
                    )
                )
                + labels.size
            ) * (outer_radius - inner_radius) / labels.size + inner_radius
        else:

            radius_of_stat = (outer_radius - inner_radius) * df[
                cols[statistic]
            ] / labels[0] + inner_radius
            lower_of_stat = (outer_radius - inner_radius) * (
                df[cols[statistic]] - df[cols[statistic] + "_conf"]
            ) / labels[0] + inner_radius
            higher_of_stat = (outer_radius - inner_radius) * (
                (df[cols[statistic]] + df[cols[statistic] + "_conf"]) / labels[0]
            ) + inner_radius

        if stacked is False:
            startA = -big_angle + angles + (2 * statistic + 1) * small_angle
            stopA = -big_angle + angles + (2 * statistic + 2) * small_angle
            df[cols[statistic] + "_err_angle"] = pd.Series(
                (startA + stopA) / 2, index=df.index
            )
        else:
            startA = -big_angle + angles + (1) * small_angle
            stopA = -big_angle + angles + (2) * small_angle
            if statistic == 0:
                df[cols[statistic] + "_err_angle"] = pd.Series(
                    (startA * 2 + stopA) / 3, index=df.index
                )
            if statistic == 1:
                df[cols[statistic] + "_err_angle"] = pd.Series(
                    (startA + stopA * 2) / 3, index=df.index
                )
        df[cols[statistic] + "radial"] = pd.Series(radius_of_stat, index=df.index)
        df[cols[statistic] + "upper"] = pd.Series(higher_of_stat, index=df.index)
        df[cols[statistic] + "lower"] = pd.Series(lower_of_stat, index=df.index)
        df[cols[statistic] + "_start_angle"] = pd.Series(startA, index=df.index)
        df[cols[statistic] + "_stop_angle"] = pd.Series(stopA, index=df.index)

        # df[cols[statistic]+'_err_angle'] = pd.Series((startA+stopA)/2,
        #                                              index=df.index)
        inner_rad = np.ones_like(angles) * inner_radius
        df[cols[statistic] + "lower"] = df[cols[statistic] + "lower"].fillna(90)
    # Store plotted values into dictionary to be add glyphs
    pdata = pd.DataFrame(
        {
            "x": np.append(np.zeros_like(inner_rad), np.zeros_like(inner_rad)),
            "y": np.append(np.zeros_like(inner_rad), np.zeros_like(inner_rad)),
            "ymin": np.append(inner_rad, inner_rad),
            "ymax": pd.Series.append(
                df[cols[1] + "radial"], df[cols[0] + "radial"]
            ).reset_index(drop=True),
            "starts": pd.Series.append(
                df[cols[1] + "_start_angle"], df[cols[0] + "_start_angle"]
            ).reset_index(drop=True),
            "stops": pd.Series.append(
                df[cols[1] + "_stop_angle"], df[cols[0] + "_stop_angle"]
            ).reset_index(drop=True),
            "Param": pd.Series.append(df.Parameter, df.Parameter).reset_index(
                drop=True
            ),
            "Colors": np.append(sTcolor, s1color),
            "Error Colors": np.append(errsTcolor, errs1color),
            "Conf": pd.Series.append(df.ST_conf, df.S1_conf).reset_index(drop=True),
            "Order": np.append(totalorder, firstorder),
            "Sens": pd.Series.append(df.ST, df.S1).reset_index(drop=True),
            "Lower": pd.Series.append(df.STlower, df.S1lower).reset_index(drop=True),
            "Upper": pd.Series.append(
                df.STupper,
                df.S1upper,
            ).reset_index(drop=True),
            "Err_Angle": pd.Series.append(
                df.ST_err_angle,
                df.S1_err_angle,
            ).reset_index(drop=True),
        }
    )
    # removed S1 or ST values if indicated by input
    if showS1 is False:
        pdata = pdata.head(len(df))
    if showST is False:
        pdata = pdata.tail(len(df))
    # convert dataframe to ColumnDataSource for glyphs
    pdata_s = ColumnDataSource(pdata)

    colors = [back_color[highl] for highl in df.highlighted]
    p.annular_wedge(
        0,
        0,
        inner_radius,
        outer_radius,
        -big_angle + angles,
        angles,
        color=colors,
    )
    # Adding axis lines and labels
    p.circle(0, 0, radius=radii, fill_color=None, line_color="#e6e6e6")
    p.text(
        0,
        radii[:],
        [str(r) for r in labels[:]],
        text_font_size="8pt",
        text_align="center",
        text_baseline="middle",
    )

    # Specify that the plotted bars are the only thing to activate hovertool
    hoverable = p.annular_wedge(
        x="x",
        y="y",
        inner_radius="ymin",
        outer_radius="ymax",
        start_angle="starts",
        end_angle="stops",
        color="Colors",
        source=pdata_s,
    )
    hover.renderers = [hoverable]

    # Add error bars
    if errorbar is True:
        p.annular_wedge(
            0,
            0,
            pdata["Lower"],
            pdata["Upper"],
            pdata["Err_Angle"],
            pdata["Err_Angle"],
            color=pdata["Error Colors"],
            line_width=1.0,
        )

        p.annular_wedge(
            0,
            0,
            pdata["Lower"],
            pdata["Lower"],
            pdata["starts"],
            pdata["stops"],
            color=pdata["Error Colors"],
            line_width=2.0,
        )

        p.annular_wedge(
            0,
            0,
            pdata["Upper"],
            pdata["Upper"],
            pdata["starts"],
            pdata["stops"],
            color=pdata["Error Colors"],
            line_width=2.0,
        )
    # Placement of parameter labels
    xr = (radii[0] * 1.1) * np.cos(np.array(-big_angle / 2 + angles))
    yr = (radii[0] * 1.1) * np.sin(np.array(-big_angle / 2 + angles))

    label_angle = np.array(-big_angle / 2 + angles)
    label_angle[label_angle < -np.pi / 2] += np.pi

    # Placing Labels and Legend
    legend_text = ["ST", "ST Conf", "S1", "S1 Conf"]
    p.text(
        xr,
        yr,
        df.Parameter,
        angle=label_angle,
        text_font_size="9pt",
        text_align="center",
        text_baseline="middle",
    )

    p.rect([-40, -40], [30, -10], width=30, height=13, color=list(stat_color.values()))
    p.rect([-40, -40], [10, -30], width=30, height=1, color=list(error_color.values()))
    p.text(
        [-15, -15, -15, -15],
        [30, 10, -10, -30],
        text=legend_text,
        text_font_size="9pt",
        text_align="left",
        text_baseline="middle",
    )
    p.annular_wedge(
        0,
        0,
        inner_radius - 10,
        outer_radius + 10,
        -big_angle + line_angles,
        -big_angle + line_angles,
        color="#999999",
    )

    return p


def make_second_order_heatmap(df, top=10, name="", mirror=True, include=[]):
    """
    Plot a heat map of the second order sensitivity indices from a given
    dataframe.  If you are choosing a high value of `top` then making
    this plot gets expensive and it is recommended to set mirror to False.

    Parameters
    -----------
    df     : pandas dataframe
             dataframe with second order sensitivity indices. This
             dataframe should be formatted in the standard output format
             from a Sobol sensitivity analysis in SALib.
    top    : int, optional
             integer specifying the number of parameter interactions to
             plot (those with the 'top' greatest values are displayed).
    name   : str, optional
             string indicating the name of the output measure
             you are plotting.
    mirror : bool, optional
             boolean indicating whether you would like to plot the mirror
             image (reflection across the diagonal).  This mirror image
             contains the same information as plotted already, but will
             increase the computation time for large dataframes.
    include: list, optional
             a list of parameters that you would like to make sure are shown
             on the heat map (even if they are not in the `top` subset)

    Returns
    --------
    p : bokeh figure
        A Bokeh figure to be plotted
    """

    # Confirm that df contains second order sensitivity indices
    if (
        list(df.columns.values).sort()
        != ["Parameter_1", "Parameter_2", "S2", "S2_conf"].sort()
    ):
        raise TypeError("df must contain second order sensitivity data")

    # Make sure `top` != 0 (it must be at least 1, even if a list is
    # specified for `include`.
    if top <= 0:
        top = 1
        print("`top` cannot be <= 0; it has been set to 1")

    # Colormap to use for plot
    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ]

    # Slice the dataframe to include only the top parameters
    df_top = df.sort_values("S2", ascending=False).head(top)

    # Make a list of all the parameters that interact with each other
    labels = list(set([x for x in pd.concat([df_top.Parameter_1, df_top.Parameter_2])]))
    for item in include:
        if item not in labels:
            labels.append(item)
    xlabels = labels
    ylabels = labels

    # Use this to scale the heat map so the max sensitivity index is darkest
    maxval = np.max(df.S2)

    xlabel = []
    ylabel = []
    color = []
    s2 = []
    s2_conf = []
    for px in xlabels:
        for py in ylabels:
            xlabel.append(px)
            ylabel.append(py)
            # sens is a dataframe with S2 and S2_conf that is stored for
            # each box of the heat map
            sens = df[df.Parameter_1.isin([px]) & df.Parameter_2.isin([py])].loc[
                :, ["S2", "S2_conf"]
            ]
            # dfs can be empty if there are no corresponding pairs in the
            # source dataframe (for example a parameter interacting with
            # itself).
            if sens.empty and not mirror:
                s2.append(float("NaN"))
                s2_conf.append(float("NaN"))
                color.append("#b3b3b3")
            # This heat map is symmetric across the diagonal, so this elif
            # statement populates the mirror image if you've chosen to
            elif sens.empty and mirror:
                sens_mirror = df[
                    df.Parameter_1.isin([py]) & df.Parameter_2.isin([px])
                ].loc[:, ["S2", "S2_conf"]]
                if sens_mirror.empty:
                    s2.append(float("NaN"))
                    s2_conf.append(float("NaN"))
                    color.append("#b3b3b3")
                else:
                    s2.append(sens_mirror.S2.values[0])
                    s2_conf.append(sens_mirror.S2_conf.values[0])
                    color.append(
                        colors[
                            max(
                                0,
                                int(round((sens_mirror.S2.values[0] / maxval) * 7) + 1),
                            )
                        ]
                    )
            # This else handles the standard (un-mirrored) boxes of the plot
            else:
                s2.append(sens.S2.values[0])
                s2_conf.append(sens.S2_conf.values[0])
                color.append(
                    colors[max(0, int(round((sens.S2.values[0] / maxval) * 7) + 1))]
                )

    source = ColumnDataSource(
        data=dict(xlabel=xlabel, ylabel=ylabel, s2=s2, s2_conf=s2_conf, color=color)
    )

    # Initialize the plot
    plottools = "hover, save, box_zoom, wheel_zoom, reset"
    p = figure(
        title="Sobol second order sensitivities",
        x_range=list(reversed(labels)),
        y_range=labels,
        x_axis_location="above",
        plot_width=700,
        plot_height=700,
        toolbar_location="right",
        tools=plottools,
    )
    p.toolbar.autohide = True
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.1

    # Plot the second order data
    p.rect("xlabel", "ylabel", 1, 1, source=source, color="color", line_color=None)

    p.select_one(HoverTool).tooltips = [
        ("Interaction", "@xlabel-@ylabel"),
        ("S2", "@s2"),
        ("S2_conf", "@s2_conf"),
    ]

    return p


from bokeh.core.properties import Instance, String
from bokeh.io import show
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.util.compiler import TypeScript

TS_CODE = """
// This custom model wraps one part of the third-party vis.js library:
//
//     http://visjs.org/index.html
//
// Making it easy to hook up python data analytics tools (NumPy, SciPy,
// Pandas, etc.) to web presentations using the Bokeh server.

import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import {ColumnDataSource} from "models/sources/column_data_source"
import {LayoutItem} from "core/layout"
import * as p from "core/properties"

declare namespace vis {
  class Graph3d {
    constructor(el: HTMLElement, data: object, OPTIONS: object)
    setData(data: vis.DataSet): void
  }

  class DataSet {
    add(data: unknown): void
  }
}

// This defines some default options for the Graph3d feature of vis.js
// See: http://visjs.org/graph3d_examples.html for more details.
const OPTIONS = {
  width: '100%',
  height: '400px',
  style: 'surface',
  showPerspective: true,
  showGrid: true,
  showSurfaceGrid: false,
  tooltip: true,
  showShadow: false,
  keepAspectRatio: true,
  verticalRatio: 0.6,
  legendLabel: '',
  xLabel: 'x',
  yLabel: 'y',
  cameraPosition: {
    horizontal: -0.35,
    vertical: 0.22,
    distance: 1.8,
  },
  colormap: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
              "#4292c6", "#2171b5", "#08519c", "#08306b"],
};
// To create custom model extensions that will render on to the HTML canvas
// or into the DOM, we must create a View subclass for the model.
//
// In this case we will subclass from the existing BokehJS ``LayoutDOMView``
export class Surface3dView extends LayoutDOMView {
  model: Surface3d

  private _graph: vis.Graph3d

  initialize(): void {
    super.initialize()
    const url = "includes/vis-graph3d.min.js"
    const script = document.createElement("script")
    script.onload = () => this._init()
    script.async = false
    script.src = url
    document.head.appendChild(script)
  }

  private _init(): void {
    // Create a new Graph3s using the vis.js API. This assumes the vis.js has
    // already been loaded (e.g. in a custom app template). In the future Bokeh
    // models will be able to specify and load external scripts automatically.
    //
    // BokehJS Views create <div> elements by default, accessible as this.el.
    // Many Bokeh views ignore this default <div>, and instead do things like
    // draw to the HTML canvas. In this case though, we use the <div> to attach
    // a Graph3d to the DOM.
    OPTIONS.xLabel = this.model.xLabel
    OPTIONS.yLabel = this.model.yLabel
    this._graph = new vis.Graph3d(this.el, this.get_data(), OPTIONS)

    // Set a listener so that when the Bokeh data source has a change
    // event, we can process the new data
    this.connect(this.model.data_source.change, () => {
      this._graph.setData(this.get_data())
    })
  }

  // This is the callback executed when the Bokeh data has an change. Its basic
  // function is to adapt the Bokeh data source to the vis.js DataSet format.
  get_data(): vis.DataSet {
    const data = new vis.DataSet()
    const source = this.model.data_source
    for (let i = 0; i < source.get_length()!; i++) {
      data.add({
        x: source.data[this.model.x][i],
        y: source.data[this.model.y][i],
        z: source.data[this.model.z][i],
      })
    }
    return data
  }

  get child_models(): LayoutDOM[] {
    return []
  }

  _update_layout(): void {
    this.layout = new LayoutItem()
    this.layout.set_sizing(this.box_sizing())
  }
}

// We must also create a corresponding JavaScript BokehJS model subclass to
// correspond to the python Bokeh model subclass. In this case, since we want
// an element that can position itself in the DOM according to a Bokeh layout,
// we subclass from ``LayoutDOM``
export namespace Surface3d {
  export type Attrs = p.AttrsOf<Props>

  export type Props = LayoutDOM.Props & {
    x: p.Property<string>
    y: p.Property<string>
    z: p.Property<string>
    data_source: p.Property<ColumnDataSource>
    xLabel: p.Property<string>
    yLabel: p.Property<string>
  }
}

export interface Surface3d extends Surface3d.Attrs {}

export class Surface3d extends LayoutDOM {
  properties: Surface3d.Props
  __view_type__: Surface3dView

  constructor(attrs?: Partial<Surface3d.Attrs>) {
    super(attrs)
  }

  // The ``__name__`` class attribute should generally match exactly the name
  // of the corresponding Python class. Note that if using TypeScript, this
  // will be automatically filled in during compilation, so except in some
  // special cases, this shouldn't be generally included manually, to avoid
  // typos, which would prohibit serialization/deserialization of this model.
  static __name__ = "Surface3d"

  static {
    // This is usually boilerplate. In some cases there may not be a view.
    this.prototype.default_view = Surface3dView

    // The @define block adds corresponding "properties" to the JS model. These
    // should basically line up 1-1 with the Python model class. Most property
    // types have counterparts, e.g. ``bokeh.core.properties.String`` will be
    // ``String`` in the JS implementatin. Where the JS type system is not yet
    // as rich, you can use ``p.Any`` as a "wildcard" property type.
    this.define<Surface3d.Props>(({String, Ref}) => ({
      x:            [ String ],
      y:            [ String ],
      z:            [ String ],
      data_source:  [ Ref(ColumnDataSource) ],
      xLabel:       [ String ],
      yLabel:       [ String ],
    }))
  }
}
"""
# This custom extension model will have a DOM view that should layout-able in
# Bokeh layouts, so use ``LayoutDOM`` as the base class. If you wanted to create
# a custom tool, you could inherit from ``Tool``, or from ``Glyph`` if you
# wanted to create a custom glyph, etc.
class Surface3d(LayoutDOM):

    # The special class attribute ``__implementation__`` should contain a string
    # of JavaScript code that implements the browser side of the extension model.
    __implementation__ = TypeScript(TS_CODE)

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    # This is a Bokeh ColumnDataSource that can be updated in the Bokeh
    # server by Python code
    data_source = Instance(ColumnDataSource)

    xLabel = String()
    yLabel = String()

    # The vis.js library that we are wrapping expects data for x, y, and z.
    # The data will actually be stored in the ColumnDataSource, but these
    # properties let us specify the *name* of the column that should be
    # used for each field.
    x = String()

    y = String()

    z = String()
