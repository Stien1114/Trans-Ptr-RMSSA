# topology_loader.py
"""
Returns (networkx.Graph, number of nodes) based on topology name.
To add a topology later, just add a line to _EDGE_MAP.
"""

import networkx as nx

_EDGE_MAP = {
    "NSF": [
        (0, 1, 1100), (0, 2, 1600), (0, 7, 2800),
        (1, 2, 600), (1, 3, 1000),
        (2, 5, 2000),
        (3, 4, 600), (3, 10, 2400),
        (4, 5, 1100), (4, 6, 800),
        (5, 9, 1200), (5, 12, 2000),
        (6, 7, 700),
        (7, 8, 700),
        (8, 9, 900), (8, 11, 500), (8, 13, 500),
        (10, 11, 800), (10, 13, 800),
        (11, 12, 300),
        (12, 13, 300),
    ],
    "N6S9": [
        (0, 1, 1000), (0, 2, 1200),
        (1, 2, 600), (1, 3, 800), (1, 4, 1000),
        (2, 4, 800),
        (3, 4, 600), (3, 5, 1000),
        (4, 5, 1200),
    ],
    "EURO16": [
        (0, 1, 514), (0, 3, 540),
        (1, 2, 393), (1, 6, 600), (1, 4, 594),
        (2, 3, 259), (2, 8, 474),
        (3, 7, 552),
        (4, 5, 507),
        (5, 6, 218), (5, 9, 327),
        (6, 8, 271),
        (7, 8, 592), (7, 11, 381),
        (8, 10, 456),
        (9, 10, 522), (9, 12, 720),
        (10, 11, 757), (10, 14, 534),
        (11, 15, 420),
        (12, 13, 783),
        (13, 14, 400),
        (14, 15, 376),
    ],
}

def load_topology(name: str) -> tuple[nx.Graph, int]:
    """name can be upper or lower case; returns (G, num_nodes)."""
    name = name.upper()
    if name not in _EDGE_MAP:
        raise ValueError(f"Unsupported topology {name}. Currently supported: {list(_EDGE_MAP)}")
    G = nx.Graph()
    for s, d, dist in _EDGE_MAP[name]:
        G.add_edge(s, d, distance=dist)
    return G, G.number_of_nodes()