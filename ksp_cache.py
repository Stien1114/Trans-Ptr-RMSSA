# ksp_cache.py
"""
Pre-compute and persist the K-shortest paths results and path features for topology (s,d) pairs.
The first run calculates and writes to a local .pkl file; subsequent runs load directly.
"""

import pickle, os
from pathlib import Path
from itertools import islice
import networkx as nx
from typing import Dict, List, Tuple
import multiprocessing as _mp
import numpy as np
from math import ceil

_CACHE_DIR = Path(__file__).parent / "ksp_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_DEFAULT_CACHE_SIZE = 50


def _cache_file(topo_name: str, cache_size: int) -> Path:
    return _CACHE_DIR / f"{topo_name.lower()}_k{cache_size}_with_features.pkl"


def _calculate_modulation_format(path_length: float) -> int:
    """Calculate modulation format based on path length."""
    if path_length <= 600:
        return 4  # DP-16QAM
    elif path_length <= 1200:
        return 3  # DP-8QAM
    elif path_length <= 3500:
        return 2  # DP-QPSK
    elif path_length <= 6300:
        return 1  # DP-BPSK
    else:
        return 0  # Out of range


def calculate_fs_requirement(traffic: float, modulation: int) -> int:
    """
    Calculate the required number of Frequency Slots (FS) based on traffic and modulation format.
    This function should be consistent with the calculation method in RMSSA_environment.py.
    """
    if modulation == 0:
        return 320  # Out of transmission range, return max value

    # Determine capacity per carrier based on modulation format
    capacity_per_carrier = {
        1: 50,  # BPSK
        2: 100,  # QPSK
        3: 150,  # 8-QAM
        4: 200  # 16-QAM
    }

    # Calculate number of optical carriers needed
    carriers_needed = ceil(traffic / capacity_per_carrier[modulation])

    # Calculate FS needed (3 FS per carrier + 1 guard band)
    fs_needed = carriers_needed * 3 + 1

    return fs_needed


def _compute_path_features(paths: List[List[int]], graph: nx.Graph,
                           k: int = 3) -> Dict[str, float]:
    """
    Compute path feature statistics.
    Returns:
    - avg_length: Average length of k shortest paths
    - min_length: Length of the shortest path
    - max_length: Length of the longest path (among k)
    - avg_hops: Average number of hops
    - min_hops: Minimum number of hops
    - max_hops: Maximum number of hops (among k)
    - avg_modulation: Average modulation format (1-4)
    - best_modulation: Best modulation format (of the shortest path)
    """
    if not paths:
        return {
            'avg_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'avg_hops': 0.0,
            'min_hops': 0.0,
            'max_hops': 0.0,
            'avg_modulation': 0.0,
            'best_modulation': 0.0
        }

    # Take only the first k paths
    k_paths = paths[:k] if len(paths) >= k else paths

    lengths = []
    hops = []
    modulations = []

    for path in k_paths:
        # Calculate path length
        path_length = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = graph.get_edge_data(u, v)
            if edge_data:
                # Try to get distance or weight attribute
                path_length += edge_data.get('distance', edge_data.get('weight', 0))

        lengths.append(path_length)
        hops.append(len(path) - 1)  # Hops = Nodes - 1

        # Calculate modulation format
        mod = _calculate_modulation_format(path_length)
        modulations.append(mod if mod > 0 else 1)  # Avoid 0

    return {
        'avg_length': np.mean(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'avg_hops': np.mean(hops),
        'min_hops': float(np.min(hops)),
        'max_hops': float(np.max(hops)),
        'avg_modulation': np.mean(modulations),
        'best_modulation': modulations[0] if modulations else 0  # Modulation of the first (shortest) path
    }


def build_or_load(graph: nx.Graph, topo_name: str,
                  cache_size: int = _DEFAULT_CACHE_SIZE,
                  k_paths: int = 3):
    """
    Return enhanced cache: contains paths and features.
    Reads directly if cache file exists; otherwise computes and writes it.
    """
    f = _cache_file(topo_name, cache_size)

    if f.exists():
        with f.open("rb") as fh:
            cache_data = pickle.load(fh)

        # Compatible with old version cache
        if isinstance(cache_data, dict) and 'paths' in cache_data:
            _log(f"[KSP-CACHE] Loaded existing enhanced cache: {f}")
            return cache_data
        else:
            # Old version cache, need to recompute
            _log(f"[KSP-CACHE] Detected old version cache, recomputing features...")
            os.remove(f)

    _log(f"[KSP-CACHE] Computing paths and features {topo_name} k={cache_size} ...")

    paths_cache = {}
    features_cache = {}

    nodes = list(graph.nodes)
    total_pairs = len(nodes) * (len(nodes) - 1)
    processed = 0

    for s in graph.nodes:
        for d in graph.nodes:
            if s == d:
                continue

            try:
                # Use weight="distance" to get shortest path
                # If edge has no distance attribute, fallback to weight
                # gen = nx.shortest_simple_paths(graph, s, d, weight="distance")
                gen = nx.shortest_simple_paths(graph, s, d)
                paths = list(islice(gen, cache_size))
                paths_cache[(s, d)] = paths

                # Compute path features
                features = _compute_path_features(paths, graph, k_paths)
                features_cache[(s, d)] = features

            except nx.NetworkXNoPath:
                paths_cache[(s, d)] = []
                features_cache[(s, d)] = {
                    'avg_length': 0.0,
                    'min_length': 0.0,
                    'max_length': 0.0,
                    'avg_hops': 0.0,
                    'min_hops': 0.0,
                    'max_hops': 0.0,
                    'avg_modulation': 0.0,
                    'best_modulation': 0.0
                }

            processed += 1
            if processed % 50 == 0 and os.getenv("KSP_CACHE_VERBOSE", "0") == "1":
                _log(f"[KSP-CACHE] Progress: {processed}/{total_pairs}")

    # Combine cache
    cache_data = {
        'paths': paths_cache,
        'features': features_cache
    }

    with f.open("wb") as fh:
        pickle.dump(cache_data, fh)

    _log(f"[KSP-CACHE] Computation complete and written to enhanced cache {f}")
    return cache_data


def _log(msg: str):
    """Log only from the main process"""
    if _mp.current_process().name == "MainProcess":
        print(msg)