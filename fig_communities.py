import pickle

import matplotlib.pyplot as plt

from fig_style import *

fig, ax = plt.subplots(
    2, 1, constrained_layout=True, figsize=figsize(1, aspect_ratio=1.3)
)

DATA_PATH = "./data/communities/"
###########
# Subplot a
###########

plt.sca(ax[0])
N_values = [100, 200, 500, 600, 960]
for i, N in enumerate(N_values):
    with open(DATA_PATH + f"communities_N_{N}.pkl", "rb") as file:
        communities2 = pickle.load(file)

    ks = np.arange(len(communities2["weighted"]))
    communities2.keys()

    # plt.plot(ks[1:], lpa_communities[1:],  label='Async weighted')
    # plt.plot(ks[1:], louvain_communities[1:],  label='Async unweighted')
    ncu = [len(c) for c in communities2["unweighted"]]
    ncw = [len(c) for c in communities2["weighted"]]

    plt.plot(
        ks[1:] / N,
        np.array(ncu[1:]) / N,
        "C0--",
        alpha=1,
        lw=(i + 1) / len(N_values),
        label="",
    )  # f'{N} Unweighted')
    # We skip the first one for the weighted graph, because it sucks
    plt.plot(
        ks[2:] / N,
        np.array(ncw[2:]) / N,
        "C2--",
        alpha=1,
        lw=(i + 1) / len(N_values),
        label="",
    )  # f'{N} Weighted')
    plt.xlabel("$k / N$")
    plt.ylabel("$n_c / N$")
    # plt.legend()

plt.plot(ks[1:] / N, np.array(ncu[1:]) / N, "C0", label=f"Unweighted")
# We skip the first one for the weighted graph, because it sucks
plt.plot(ks[2:] / N, np.array(ncw[2:]) / N, "C2", label=f"Weighted")
plt.plot(ks[1:] / N, ks[1:] / N, "k--", lw=0.75, label="$k/N$")
plt.legend(loc=4)
# Numbers below provided by Guille
# [0.332, 0.1933, 0.138, 0.106, 0.086]
# For N = 2000:
plt.vlines(
    [0.0875, 0.107, 0.1375, 0.194, 0.333],
    0,
    0.5,
    color="black",
    linestyles="--",
    lw=0.5,
)
# plt.title("Number of communities for various values of N".format(N));


## SUBFIGURE b: communities size vs k
plt.sca(ax[1])

N = 50
data = np.load(DATA_PATH + f"community_sizes_{N}.npz")

sizes = data["sizes"]
avg_comm_size = data["avg_comm_size"]

ks = np.arange(1, int(N / 2) + 1)

plt.imshow(
    sizes[-1::-1, :] * 100,
    cmap=plt.cm.Greens,
    extent=[0.5, N / 2 + 0.5, 0.5, N + 0.5],
    aspect="auto",
)
plt.plot(np.array(ks), N / np.array(ks), "k--", lw=0.5)
plt.plot(np.array(ks), avg_comm_size, "r.", lw=1, markersize=3)
plt.ylabel("$s_c$")
plt.xlabel("$k$")
# plt.grid()
plt.ylim(0, 26)
plt.legend(["$N/k$", "$\\bar s_c$"])
cbar = plt.colorbar()
cbar.set_label("% of communities")
# savefig(f'community_sizes_N_{N}', dpi=300)

format_axes(ax)
fig.savefig("figures/fig_communities.pdf", dpi=300)
