import numpy as np
import download_graph as dg
from semi_sync_lpa import label_propagation_communities
from collections import Counter
from tqdm import tqdm

N = 50

ks = []
comm_sizes = []

for k in tqdm(range(1, int(N / 2) + 1), leave=False):
    # G = nx.read_graphml('graphs/{}_{}_concurrence.graphml'.format(N, k), node_type=int)
    G = dg.get_graph(N, k, "concurrence")
    ks.append(k)
    communities = list(label_propagation_communities(G, weight="weight"))
    comm_sizes.append(list(map(len, communities)))

avg_comm_size = list(map(np.mean, comm_sizes))

sizes = np.zeros((len(ks), N))

for i, k in enumerate(ks):
    ctr = Counter(comm_sizes[i])
    tot = sum(ctr.values())
    for s, num in ctr.items():
        sizes[i, s - 1] = num / tot

sizes = sizes.T

sizes[sizes == 0.0] = np.nan

np.savez(
    f"data/communities/community_sizes_{N}.npz",
    sizes=sizes,
    avg_comm_size=avg_comm_size,
)

