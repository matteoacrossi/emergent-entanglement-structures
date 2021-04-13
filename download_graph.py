"""
Download graph file from UNIMI server
"""
import io
from contextlib import closing
import requests
import scipy.sparse as sp
import networkx as nx

_BASEURL = "http://qtech2.fisica.unimi.it:8000/"

def get_adjacency(N: int, k: int, quantity: str):
    """Return the adjacency matrix for the graph of quantity
    (e.g. 'concurrence', 'mutual_information', ...), for the given N and k
    """
    url = _BASEURL + 'graphs/graphs/{}_{}_{}.npz'.format(N, k, quantity)

    response = requests.get(url)

    # If something is wrong raise an exception
    response.raise_for_status()

    # Read the bytes of the response directly from memory
    with closing(response):
        matrix = sp.load_npz(io.BytesIO(response.content))
    return matrix

def get_graph(N: int, k: int, quantity: str) -> nx.Graph:
    """Return the NetworkX graph for quantity
    (e.g. 'concurrence', 'mutual_information', ...), for the given N and k
    """
    adj = get_adjacency(N, k, quantity)
    return nx.from_scipy_sparse_matrix(adj)
