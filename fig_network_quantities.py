import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports
import numpy as np
import networkx as nx

import hilbert_graph_tools as ht

from fig_style import *

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle
import bz2


def Bks(N):
    ks = np.arange(N + 2)
    return np.cos(np.pi * ks / (N + 1))


def B_from_k(k, N):
    return np.cos(np.pi * k / (N + 1))


def k_from_B(B, N):
    if np.abs(B) > 1:
        return np.nan
    elif B == -1.0:
        return N
    else:
        return np.where(Bks(N) < B)[0][0] - 1


#     else:
#         return int(np.ceil(np.arccos(B)/np.pi*(N+1)))

# Ns = [20, 60, 180]

# Bs = np.linspace(0, 1.2, 400)

# density = {n: 0*Bs for n in Ns}
# avg_disparity = {n: 0*Bs for n in Ns}

# for N in Ns:
#     for i, B in enumerate(Bs):
#         k = k_from_B(B, N)
#         if k <= N/2:
#             G = nx.read_graphml('graphs/{}_{}_concurrence.graphml'.format(N, k), node_type=int)

#             strengths_k = np.array(list(dict(G.degree(weight='weight')).values()))
#             disparities_k = ht.Y_i(G)
#             disparities_k = [disparities_k[i] for i in sorted(disparities_k.keys())]

#             density[N][i] = np.sum(strengths_k) / (N * (N-1))
#             avg_disparity[N][i] = np.mean(disparities_k)
#         else:
#             if np.abs(B) > 1:
#                 density[N][i] = 0.0
#                 avg_disparity[N][i] = 0.0

# with open("data/network_properties/data.pkl", "wb") as file:
#     pickle.dump({'Ns': Ns, 'Bs': Bs, 'density': density, 'avg_disparity': avg_disparity}, file)


with open("data/network_properties/data.pkl", "rb") as file:
    data = pickle.load(file)

Ns = data["Ns"]
Bs = data["Bs"]
density = data["density"]
avg_disparity = data["avg_disparity"]

# set cut viridis as colormap
viridisBig = plt.cm.get_cmap("viridis_r", 512)
cutviridis = ListedColormap(viridisBig(np.linspace(0.20, 0.95, 256)))
colormap = plt.cm.viridis
colormap = cutviridis

N = 180
# load data
with open(f"data/xx_data_180_concurrence.dat", "rb") as file:
    print(file.name)
    xx_data_180_concurrence = pickle.load(bz2.BZ2File(file, "r"))

# convert to tuples from dictionaries
disps_i = xx_data_180_concurrence["Disparities by node(i)"]
strs_i = xx_data_180_concurrence["Strengths (i)"]

Bfy = lambda ks, N: np.cos(np.pi * ks / (N))

fig = plt.figure(constrained_layout=True, figsize=figsize(2.3, aspect_ratio=0.6))

gs = GridSpec(
    2, 5, figure=fig, width_ratios=[0.94, 0.03, 0.03, 0.25, 1], wspace=0.2, hspace=0.3
)


ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax1.set_xticks([1, 45, 90, 135, 180])
ax2.set_xticks([1, 45, 90, 135, 180])
# plt.setp(ax1.get_xticklabels(), visible=False)

gs1 = gs[:, 2].subgridspec(3, 1, height_ratios=[0.3, 1, 0.3])

cbar_ax = fig.add_subplot(gs1[1])

ax3 = fig.add_subplot(gs[0, 4])
ax_distr = fig.add_subplot(gs[1, 4])

yidxs = [
    90,
    102,
    114,
    126,
    138,
    147,
    151,
    155,
    160,
    165,
    168,
    170,
    172,
    174,
    175,
    176,
    177,
    178,
    179,
]
print(yidxs)
# yidxs = np.arange(1, 10)
for idx in yidxs:
    ys = disps_i[idx]
    x, y = zip(*ys.items())  # from dict to tuple
    color = 2 - 2 * idx / (len(disps_i) - 1)
    plt1 = ax1.step(x, y, where="mid", color=colormap(color), linewidth=0.9)

ax1.set_yscale("log")

# yidxs = np.array(list(range(90,147,12)) + list(range(147,167,4)) + list(range(168,180,2))+[179])
# yidxs = np.array(list(range(90,180,12)) + [177, 178, 179])
for idx in reversed(yidxs):
    ys = strs_i[idx]
    x, y = zip(*ys.items())  # from dict to tuple
    color = 2 - 2 * idx / (len(strs_i) - 1)
    ax2.step(x, y, where="mid", color=colormap(color), linewidth=0.9)


# cbar_ax = fig.add_axes([0.22, 1.035, 0.6, 0.01])
sm = plt.cm.ScalarMappable(
    cmap=colormap, norm=plt.Normalize(vmin=0.5, vmax=0)
)  # colorbar
cbar = fig.colorbar(sm, pad=10, cax=cbar_ax, orientation="vertical")

cbar_ax.set_ylim(0.0, 0.5)
cbar_ax.set_ylabel("k/N")
cbar_ax.yaxis.set_label_coords(-5.0, 0.5)  # manually fix the position of label
cbar.set_ticks(np.round(np.linspace(0.0, 0.5, 6), 1))

# add second axis on the colorbar
# iticks = np.arange(0, 91, 15)
# biticks = Bfy(iticks,180)

biticks = [0, 0.2, 0.5, 0.7, 0.9, 1.0]
iticks = [k_from_B(b, N) for b in biticks]


cax2 = cbar_ax.twinx()
cax2.set_ylim(0, 1)
cax2.set_yticks(iticks)
cax2.set_yticklabels(np.round(biticks, 2))
cax2.set_ylabel("B")
# cax2.xaxis.set_label_coords(1, 0) # manually fix the position of label

#
# sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=.5, vmax=0)) # colorbar
# cbar = plt.colorbar(sm, ax=ax2)
# cbar.set_label("k/N")
# cbar.ax.yaxis.set_label_position("left")
#
## add second axis on the colorbar
# iticks = np.arange(0,91,15)
# biticks = Bfy(iticks,180)
# cax = cbar.ax
# cax2 = cax.twinx()
# cax2.set_ylim(1,0)
# cax2.set_yticks(iticks)
# cax2.set_yticklabels(np.round(biticks, 2))
# cax2.set_ylabel("B")

ax1.set_ylabel("$Y_i$")
ax2.set_ylabel("$s_i$")
ax2.set_xlabel("$i$")

cmap = plt.cm.inferno

for N in Ns:
    l = ax3.plot(Bs, np.array(density[N]) * (N - 1), label=f"$N={N}$", lw=1)

ax3.set_ylabel("$\\langle s_i \\rangle $")
ax3.set_xlabel("$B$")
ax3.legend(loc="lower left", ncol=3, columnspacing=1, handlelength=1)
# labelLine(l[0], x=.2)
ax4 = ax3.twinx()
ax4.spines["right"].set_visible(True)


for N in Ns:
    l = ax4.plot(Bs, avg_disparity[N], label=N, lw=1)

ax4.set_ylabel("$\\langle Y_i \\rangle$")
ax4.axvline(x=1, linestyle="--", color="k", lw=0.2)

ax4.annotate(
    "$\\langle s_i \\rangle $",
    xy=(0.8, 0.36),
    xycoords="data",
    xytext=(0.72, 0.45),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
)
# labelLine(l[0], x=.2)

ax4.annotate(
    "$\\langle Y_i \\rangle$",
    xy=(0.5, 0.5),
    xycoords="data",
    xytext=(0.55, 0.58),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
)


with open("data/local_distributions/local_scaled_distributions.pkl", "rb") as file:
    data = pickle.load(file)


with open("data/local_distributions/average_wasserstein_distances.pkl", "rb") as file:
    avg_dist = pickle.load(file)

nodes = range(1, 91)
plt.sca(ax_distr)
for d in nodes:
    plt.hist(
        data[d], bins=50, histtype="step", range=(0, 1.8), cumulative=True, density=True
    )

plt.xlim(0, 1.6)
plt.xlabel("$\\omega_{ij} d_i /s_i$")
plt.ylabel("$P(\\omega_{ij} d_i /s_i)$")

# Create inset of width 1.3 inches and height 0.9 inches
# at the default upper right location
axins = inset_axes(
    ax_distr,
    loc="upper center",
    width=0.9,
    height=0.55,
    bbox_to_anchor=(0.27, 1 - 0.3, 0.3, 0.3),
    bbox_transform=ax_distr.transAxes,
)

plt.sca(axins)
plt.plot(range(1, 11), avg_dist, ".-")
plt.xlabel("$k$")
plt.ylabel("$\\bar W$")
axins.set_xticks([1, 5, 10])


format_axes([ax1, ax2, ax3, ax_distr])


savefig(f"figures/fig_network_quantities")
