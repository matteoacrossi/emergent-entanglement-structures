import matplotlib.pyplot as plt
import numpy as np

from fig_style import *


def B(N, k):
    return np.cos(np.pi * k / (N + 1))


nrows = 3
ncols = 3

mew = 0.5
msz = 5

gold_mean = (np.sqrt(5.0) - 1.0) / 2.0
aspect_ratio = gold_mean * 1.15
scale = 0.8
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
fig_width = fig_width_pt * inches_per_pt * scale
fig_height = aspect_ratio * fig_width

fig = plt.figure(figsize=(fig_width * float(ncols), fig_height * float(nrows)))
G = plt.GridSpec(nrows, ncols)

ax = {}
al = {}
for i in range(0, nrows):
    ax[i] = {}
    for j in range(0, ncols):
        ax[i][j] = plt.subplot(G[i : (i + 1), j : (j + 1)])

colorsA = plt.cm.Set1(np.linspace(0, 1, 9))[1:]
colorsB = plt.cm.tab10(np.linspace(0, 1, 10))[5:]
colorsC = plt.cm.tab10(np.linspace(0, 1, 10))
symbols = ["o", "s", "v", "^", ">", "D", "*"]


#############################
# A
#############################
ca = ax[0][0]
Ns = [50, 200, 400, 600]
for i, N in enumerate(Ns):
    data = np.loadtxt(f"data/avd_vs_k/N_{N}.dat")
    ca.plot(data[:, 0] / N, data[:, 1], c=colorsA[i], label=N)  # f'$N = {N}$')

lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    title="$N$",
    loc="lower left",
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    columnspacing=1,
    handlelength=1,
)

Npeaks = 4
Npeaks_total = 5
mn, mx = 1.5, 30

Ks = {}
Bs = {}
peak_data = np.loadtxt(f"data/d_prima/peaks_{N}.dat")
for i in range(Npeaks_total - 1, Npeaks_total - Npeaks - 1, -1):
    k = peak_data[i, 1]
    Ks[Npeaks_total - i] = k / N
    Bs[Npeaks_total - i] = np.round(B(N, k), 2)
    ca.plot(
        [k / N, k / N],
        [mn, mx],
        c=colorsB[Npeaks_total - i - 1],
        ls="--",
        lw=1,
        label=f"$m = {Npeaks_total - i}$",
    )


ca.set_xlabel(r"$k / N$")
ca.set_ylabel(r"$\langle d_i \rangle$")
ca.set_xscale("log")
ca.set_yscale("log")
ca.set_xbound(0.04, 0.6)
ca.set_ybound(1.5, 30)


#############################
# B
#############################
ca = ax[0][1]
N = 600
data = np.loadtxt(f"data/d_prima/N_{N}.dat")
ca.plot(data[:, 0] / N, data[:, 1], c="#" + "9" * 6)

Npeaks = 4
Npeaks_total = 5
mn, mx = 3e-3, 2

Ks = {}
Bs = {}
peak_data = np.loadtxt(f"data/d_prima/peaks_{N}.dat")
for i in range(Npeaks_total - 1, Npeaks_total - Npeaks - 1, -1):
    k = peak_data[i, 1]
    Ks[Npeaks_total - i] = k / N
    Bs[Npeaks_total - i] = np.round(B(N, k), 2)
    ca.plot(
        [k / N, k / N],
        [mn, mx],
        c=colorsB[Npeaks_total - i - 1],
        ls="--",
        lw=1.0,
        label=f"$m = {Npeaks_total - i}$",
    )

lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    loc="lower left",
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    ncol=1,
    columnspacing=1,
    handlelength=1,
)

ca.set_xlabel(r"$k / N$")
ca.set_ylabel(r"$\Delta \langle d_i \rangle$")
ca.set_xscale("log")
ca.set_yscale("log")
ca.set_xbound(0.04, 0.6)
ca.set_ybound(mn, mx)


#############################
# C
#############################
ca = ax[0][2]
for p in range(Npeaks):
    data = np.loadtxt(f"data/peaks_vs_N/Peak_{p}.dat")
    ca.plot(
        data[:, 0],
        data[:, 1],
        c=colorsB[p],
        marker=symbols[p],
        markeredgewidth=mew,
        markeredgecolor="k",
        ms=msz,
        label=f"$m = {p+1}$",
    )

lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    loc=(0.03, 0.15),
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    ncol=1,
    columnspacing=1,
    handlelength=1,
)

ca.set_xlabel(r"$N$")
ca.set_ylabel(r"$\overline{B}_m(N)$")


#############################
# D
#############################
ca = ax[1][0]
mn, mx = 0.05, 3

Ns = [50, 200, 400, 600]
for i, N in enumerate(Ns):
    data = np.loadtxt(f"data/sigd_vs_k/N_{N}.dat")
    ca.plot(data[:, 0] / N, data[:, 1], c=colorsA[i], label=N)  # f'$N = {N}$')

# for k in Ks:
# 	ca.plot([Ks[k], Ks[k]], [mn, mx], c='#'+'9'*6, lw=1, ls='--')

lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    loc="lower left",
    title="$N$",
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    ncol=1,
    columnspacing=1,
    handlelength=1,
)


Npeaks = 4
Npeaks_total = 5

Ks = {}
Bs = {}
peak_data = np.loadtxt(f"data/d_prima/peaks_{N}.dat")
for i in range(Npeaks_total - 1, Npeaks_total - Npeaks - 1, -1):
    k = peak_data[i, 1]
    Ks[Npeaks_total - i] = k / N
    Bs[Npeaks_total - i] = np.round(B(N, k), 2)
    ca.plot(
        [k / N, k / N],
        [mn, mx],
        c=colorsB[Npeaks_total - i - 1],
        ls="--",
        lw=1.0,
        label=f"$m = {Npeaks_total - i}$",
    )


ca.set_xlabel(r"$k / N$")
ca.set_ylabel(r"$\sigma(d_i)$")
ca.set_xscale("log")
ca.set_yscale("log")
ca.set_xbound(0.04, 0.6)
ca.set_ybound(mn, mx)


#############################
# E
#############################
ca = ax[1][1]
for p in range(Npeaks):
    data = np.loadtxt(f"data/sigma_peaks_vs_N/On_peak_{p}.dat")
    ca.plot(
        data[:, 0],
        data[:, 1],
        c=colorsB[p],
        marker=symbols[p],
        markeredgewidth=mew,
        markeredgecolor="k",
        ms=msz,
        label=f"$m = {p+1}$",
    )

lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    loc="upper right",
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    ncol=1,
    columnspacing=1,
    handlelength=1,
)

ca.set_xlabel(r"$N$")
ca.set_ylabel(r"$\sigma( d_i )$")
# ca.set_xscale("log")
# ca.set_yscale("log")


#############################
# F
#############################
ca = ax[1][2]
for p in range(Npeaks):
    data = np.loadtxt(f"data/sigma_peaks_vs_N/Off_peak_{p}.dat")
    ca.plot(
        data[:, 0],
        data[:, 1],
        c=colorsB[p],
        marker=symbols[p],
        markeredgewidth=mew,
        markeredgecolor="k",
        ms=msz,
    )
ca.plot(
    data[:, 0],
    3 / np.sqrt(data[:, 0]),
    ls="--",
    c="#" + "9" * 6,
    lw=1,
    label=r"$\propto N^{-1 / 2}$",
)
lines, labels = ca.get_legend_handles_labels()
ca.legend(
    lines,
    labels,
    loc="lower left",
    shadow=False,
    fancybox=False,
    frameon=True,
    numpoints=1,
    ncol=1,
    columnspacing=1,
    handlelength=1,
)

ca.set_xlabel(r"$N$")
ca.set_ylabel(r"$\sigma( d_i )$")
ca.set_xscale("log")
ca.set_yscale("log")


#############################
# G - I
#############################
for i in range(3):
    ca = ax[2][i]
    p = i + 1
    for l in range(1, p + 2):
        data = np.loadtxt(f"data/spurious_vs_N/Peak_{p}_length_{l}.dat")
        ca.plot(
            data[:, 0],
            data[:, 1],
            c=colorsB[i],
            marker=symbols[l - 1],
            markeredgewidth=mew,
            markeredgecolor="k",
            ms=msz,
            label=f"$l = {l}$",
        )
    ca.plot(
        data[:, 0],
        3 / data[:, 0],
        ls="--",
        c="#" + "9" * 6,
        lw=1,
        label=r"$\propto 1 / N$",
    )
    lines, labels = ca.get_legend_handles_labels()
    ca.legend(
        lines,
        labels,
        loc="center right",
        shadow=False,
        fancybox=False,
        frameon=True,
        numpoints=1,
        ncol=i + 1,
        columnspacing=1,
        handlelength=1,
    )

    ca.set_xlabel(r"$N$")
    ca.set_ylabel(r"$\langle C \rangle_l$")
    ca.set_xscale("log")
    ca.set_yscale("log")


#############################
# Letters on corners
#############################
format_axes([ax[i][j] for i in range(nrows) for j in range(ncols)], (-0.25, 1.01))

# Adjust spacings
plt.subplots_adjust(wspace=0.34, hspace=0.4)

# Save figure
# tight_layout()
# G.tight_layout(fig, rect=[0, 0, 1, 0.96], pad = 0.3)
fig.savefig("figures/fig_topology.pdf", bbox_inches="tight")
