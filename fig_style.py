import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import atan2, degrees


golden_mean = (np.sqrt(5.0) - 1.0) / 2.0

subplotlabelfont = {"fontweight": "bold", "fontsize": 10}


def format_axes(axes, position=(-0.2, 1.02)):
    for i, ax in enumerate(axes, start=97):
        ax.text(
            *position,
            chr(i),
            fontdict=subplotlabelfont,
            weight="bold",
            transform=ax.transAxes
        )


def figsize(scale, aspect_ratio=golden_mean):
    fig_width_pt = 246.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch

    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * aspect_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


rcparams = {  # setup matplotlib to use latex for output
    # "text.usetex": False,                # use LaTeX to write all text
    # "font.family": "sans-serif",
    # "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    # "font.sans-serif": ["Helvetica"],
    # "font.monospace": [],
    "axes.labelsize": 7,  # LaTeX default is 10pt font.
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "font.size": 7,
    "legend.fontsize": 7,  # Make the legend/label fonts a little smaller
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    # "mathtext.fontset": "custom",
    # "mathtext.rm": "Helvetica",
    # "mathtext.it": "Helvetica:italic",
    # "mathtext.bf": "Helvetica:bold",
    # "mathtext.sf": "Helvetica",
    # "mathtext.tt": "DejaVu Sans",
    # "mathtext.cal": "DejaVu Sans:italic",
}

mpl.rcParams.update(rcparams)

# I make my own newfig and savefig functions
def newfig(width, aspect_ratio=golden_mean):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, aspect_ratio))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, **kwargs):
    plt.savefig("{}.pdf".format(filename), bbox_inches="tight", **kwargs)


def savefig_old(filename, **kwargs):
    plt.savefig("{}.pdf".format(filename), bbox_inches="tight", **kwargs)


from math import atan2, degrees
import numpy as np

# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print("x label location is outside data range!")
        return

    # Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (
        xdata[ip] - xdata[ip - 1]
    )

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)

