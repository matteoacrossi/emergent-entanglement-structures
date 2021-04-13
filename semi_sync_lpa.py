
import warnings
import networkx as nx
from collections import Counter

def label_propagation_communities(G, weight=None):
    """Generates community sets determined by label propagation

    Finds communities in `G` using a semi-synchronous label propagation
    method[1]_. This method combines the advantages of both the synchronous
    and asynchronous models. Not implemented for directed graphs.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph.
    weight : str


    Yields
    ------
    communities : generator
        Yields sets of the nodes in each community.

    Raises
    ------
    NetworkXNotImplemented
       If the graph is directed

    References
    ----------
    .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
       via semi-synchronous label propagation algorithms. In Business
       Applications of Social Network Analysis (BASNA), 2010 IEEE International
       Workshop on (pp. 1-8). IEEE.
    """
    coloring = _color_network(G)
    # Create a unique label for each node in the graph
    labeling = {v: k for k, v in enumerate(G)}
    while not _labeling_complete(labeling, G, weight=weight):
        # Update the labels of every node with the same color.
        for color, nodes in coloring.items():
            for n in nodes:
                _update_label(n, labeling, G, weight=weight)

    for label in set(labeling.values()):
        yield set((x for x in labeling if labeling[x] == label))

def _color_network(G):
    """Colors the network so that neighboring nodes all have distinct colors.

       Returns a dict keyed by color to a set of nodes with that color.
    """
    coloring = dict()  # color => set(node)
    colors = nx.coloring.greedy_color(G)
    for node, color in colors.items():
        if color in coloring:
            coloring[color].add(node)
        else:
            coloring[color] = set([node])
    return coloring


def _labeling_complete(labeling, G, weight=None):
    """Determines whether or not LPA is done.

       Label propagation is complete when all nodes have a label that is
       in the set of highest frequency labels amongst its neighbors.

       Nodes with no neighbors are considered complete.
    """
    return all(labeling[v] in _most_frequent_labels(v, labeling, G, weight)
               for v in G if len(G[v]) > 0)


def _most_frequent_labels(node, labeling, G, weight=None):
    """Returns a set of all labels with maximum frequency in `labeling`.

       Input `labeling` should be a dict keyed by node to labels.
    """
    if not G[node]:
        # Nodes with no neighbors are themselves a community and are labeled
        # accordingly, hence the immediate if statement.
        return {labeling[node]}

    # Compute the frequencies of all neighbours of node
    #freqs = Counter({labeling[q]: (G[node][q][weight] if weight else 1) for q in G[node]})
    #freqs = Counter(labeling[q] for q in G[node])
    #print(node, freqs)

    freqs = Counter()
    for v in G[node]:
        freqs.update({labeling[v]: G.edges[node, v][weight]
                        if weight else 1})

    #freqs2 = Counter({labeling[q]: (G[node][q][weight] if weight else 1) for q in G[node]})
    #print(node, freqs)

    max_freq = max(freqs.values())
    return {label for label, freq in freqs.items() if freq == max_freq}


def _update_label(node, labeling, G, weight=None):
    """Updates the label of a node using the Prec-Max tie breaking algorithm

       The algorithm is explained in: 'Community Detection via Semi-Synchronous
       Label Propagation Algorithms' Cordasco and Gargano, 2011
    """
    old = labeling[node]
    high_labels = _most_frequent_labels(node, labeling, G, weight=weight)
    if len(high_labels) == 1:
        labeling[node] = high_labels.pop()
    elif len(high_labels) > 1:
        # Prec-Max
        if labeling[node] not in high_labels:
            labeling[node] = max(high_labels)

    # if labeling[node] != old:
    #     print(node, f"{old} -> {labeling[node]}")


# def label_propagation_communities(G,
#                                   weight=None,
#                                   maxiter=None,
#                                   initial_labeling=None):
#     """Generates community sets determined by label propagation
#     Finds communities in `G` using a semi-synchronous label propagation
#     method[1]_. This method combines the advantages of both the synchronous
#     and asynchronous models. Not implemented for directed graphs.

#     Parameters
#     ----------
#     G : graph
#         An undirected NetworkX graph.
#     weight : str
#         The edge attribute representing the weight of an edge. If None, each
#         edge is assumed to have weight one. In this algorithm, the weight of
#         an edge is used in determining the frequency with which a label appears
#         among the neighbors of a node: a higher weight means the label appears
#         more often.
#     maxiter : int
#         Maximum number of iterations before stopping. If None, continue until
#         converged
#     initial_labeling : dict
#         Specify an initial labeling in the form `{node: label}`
#         If None, each node will be given a different label.

#     Yields
#     ------
#     communities : generator
#         Yields sets of the nodes in each community.
#     Raises
#     ------
#     NetworkXNotImplemented
#        If the graph is directed
#     References
#     ----------
#     .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
#        via semi-synchronous label propagation algorithms. In Business
#        Applications of Social Network Analysis (BASNA), 2010 IEEE International
#        Workshop on (pp. 1-8). IEEE.
#     """
#     coloring = _color_network(G)
#     if initial_labeling is None:
#         # Create a unique label for each node in the graph
#         labeling = {v: k for k, v in enumerate(G)}
#     else:
#         labeling = initial_labeling

#     it = 0
#     while not _labeling_complete(labeling, G, weight=weight):
#         # Update the labels of every node with the same color.
#         for color, nodes in coloring.items():
#             for n in nodes:
#                 _update_label(n, labeling, G, weight=weight)

#         it += 1

#         if maxiter is not None:
#             if it == maxiter:
#                 warnings.warn("Maximum iterations reached")
#                 break

#     for label in set(labeling.values()):
#         yield {x for x in labeling if labeling[x] == label}




# def _color_network(G):
#     """Colors the network so that neighboring nodes all have distinct colors.
#        Returns a dict keyed by color to a set of nodes with that color.
#     """
#     coloring = dict()  # color => set(node)
#     colors = nx.coloring.greedy_color(G)
#     for node, color in colors.items():
#         if color in coloring:
#             coloring[color].add(node)
#         else:
#             coloring[color] = {node}
#     return coloring


# def _labeling_complete(labeling, G, weight=None):
#     """Determines whether or not LPA is done.
#        Label propagation is complete when all nodes have a label that is
#        in the set of highest frequency labels amongst its neighbors.
#        Nodes with no neighbors are considered complete.
#     """

# #    print("Converged labels:", sum (labeling[v] in _most_frequent_labels(v, labeling, G)
# #              for v in G if len(G[v]) > 0))
#     return all(labeling[v] in _most_frequent_labels(v, labeling, G, weight=weight)
#                for v in G if len(G[v]) > 0)

# def _most_frequent_labels(node, labeling, G, weight=None):
#     """Returns a set of all labels with maximum frequency in `labeling`.
#        Input `labeling` should be a dict keyed by node to labels.
#     """
#     if not G[node]:
#         # Nodes with no neighbors are themselves a community and are labeled
#         # accordingly, hence the immediate if statement.
#         return {labeling[node]}

# #             label_freq = Counter()
# #             for v in G[node]:
# #                 label_freq.update({labels[v]: G.edges[node, v][weight]
# #                                    if weight else 1})

#     # Compute the frequencies of all neighbours of node
#     freqs = Counter({labeling[q]: (G[node][q][weight] if weight else 1) for q in G[node]})

#     max_freq = max(freqs.values())
#     aa = {label for label, freq in freqs.items() if freq == max_freq}

#     return aa


# def _update_label(node, labeling, G, weight=None):
#     """Updates the label of a node using the Prec-Max tie breaking algorithm
#        The algorithm is explained in: 'Community Detection via Semi-Synchronous
#        Label Propagation Algorithms' Cordasco and Gargano, 2011
#     """
#     old = labeling[node]
#     high_labels = _most_frequent_labels(node, labeling, G, weight=weight)
#     if len(high_labels) == 1:
#         labeling[node] = high_labels.pop()
#     elif len(high_labels) > 1:
#         # Prec-Max
#         if labeling[node] not in high_labels:
#             labeling[node] = max(high_labels)

#     # if labeling[node] != old:
#     #     print(node, f"{old} -> {labeling[node]}")
