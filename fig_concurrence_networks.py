from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from fig_style import *
import networkx as nx
from semi_sync_lpa import label_propagation_communities
import hilbert_graph_tools as ht


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

N = 20
kvalues = range(1, 11)

viridisBig = plt.cm.get_cmap("viridis_r", 512)
cutviridis = ListedColormap(viridisBig(np.linspace(0.20, 0.95, 256)))

fig, ax = plt.subplots(2, int(len(kvalues) / 2), figsize=[2.6 * len(kvalues) / 2, 5.0])

for i, k in enumerate(kvalues):
    G = nx.read_graphml(
        "data/graphs/{}_{}_concurrence.graphml".format(N, k), node_type=int
    )

    communities2 = label_propagation_communities(G, weight="weight")
    partition2 = {}
    for j, c in enumerate(communities2):
        for node in c:
            partition2[node] = j + 1

    plt.sca(ax.flatten()[i])
    ax.flatten()[i].set_title("$k$ = {}, $B_k$ = {:.2f}".format(k, B_from_k(k, N)))
    ht.draw_entanglement_graph(
        G,
        node_color=partition2,
        edge_scale_factor=0.55,
        node_scale_factor=0.25,
        cmap=cutviridis,
    )
    plt.axis("off")

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.96, 0.2, 0.01, 0.6])
mappable = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 0.75), cmap=cutviridis)
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label("Concurrence")

savefig("figures/fig_concurrence_graph_communities")
