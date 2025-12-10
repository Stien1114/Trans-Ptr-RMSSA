# RMSSA_environment.py - Optical Network Resource Allocation Environment
"""
This module implements the optical network simulation environment for RMSSA problems.
It provides:
- OpticalNetwork class for network state management
- First-Fit spectrum allocation algorithm
- Multi-process parallel evaluation
- Various request ordering heuristics
"""

from random import random
import torch
import networkx as nx
from math import ceil
from topology_loader import load_topology
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import pickle
import ksp_cache
import psutil


class OpticalNetwork:
    """
    Represents an SDM-EON (Space Division Multiplexing Elastic Optical Network).
    
    Attributes:
        num_cores: Number of cores per fiber (default: 4)
        num_edges: Number of directed edges in the network
        graph: NetworkX graph representing the topology
        network_state: 3D tensor tracking spectrum availability [edges, cores, slots]
    """
    
    def __init__(self, nx_graph, num_cores=4, device='cpu'):
        """
        Initialize the Optical Network.
        
        Args:
            nx_graph: NetworkX graph with 'distance' edge attributes
            num_cores: Number of spatial cores per fiber
            device: Computation device (cpu/cuda)
        """
        self.num_cores = num_cores
        self.num_edges = len(nx_graph.edges()) * 2  # Bidirectional

        # Create undirected graph
        self.graph = nx_graph

        # Create mapping from node pairs to edge indices, and record distance for each edge
        self.edge_mapping = {}
        self.edge_distances = {}
        for edge_index, (s, d, distance) in enumerate(nx_graph.edges(data='distance')):
            self.edge_mapping[(s, d)] = edge_index * 2
            self.edge_mapping[(d, s)] = edge_index * 2 + 1
            self.edge_distances[(s, d)] = distance
            self.edge_distances[(d, s)] = distance
            self.graph.add_edge(s, d, distance=distance)

        # Initialize network state: 1 = available, 0 = occupied
        # Shape: [num_edges, num_cores, num_frequency_slots]
        self.network_state = torch.ones((self.num_edges, 4, 320))

    def get_edge_index(self, s, d):
        """Get the edge index for a given source-destination pair."""
        return self.edge_mapping.get((s, d), None)

    def update_network_state(self, s, d, core, fs_list):
        """Mark frequency slots as occupied on a specific edge and core."""
        edge_index = self.get_edge_index(s, d)
        if edge_index is not None:
            for fs in fs_list:
                self.network_state[edge_index, core, fs] = 0

    def check_fs_status(self, s, d, core, fs):
        """Check if a frequency slot is available."""
        edge_index = self.get_edge_index(s, d)
        if edge_index is not None:
            return self.network_state[edge_index, core, fs]
        return None

    def get_edge_distance(self, s, d):
        """Get the physical distance of an edge."""
        return self.edge_distances.get((s, d), None)

    def find_min_fs_with_continuous_availability(self, path, core, required_fs_count):
        """
        Find the minimum starting frequency slot index that has continuous availability
        across all edges in the path for the specified core.
        
        Args:
            path: List of (source, destination) tuples representing edges
            core: Core index
            required_fs_count: Number of contiguous slots required
            
        Returns:
            Minimum starting slot index, or None if not found
        """
        if not path or required_fs_count <= 0:
            return None

        for edge_index, (s, d) in enumerate(path):
            if edge_index == 0:
                edge_fs = self.network_state[self.get_edge_index(s, d), core, :]
                continuous_available_fs = edge_fs.clone()
            else:
                edge_fs = self.network_state[self.get_edge_index(s, d), core, :]
                continuous_available_fs *= edge_fs

        count = 0
        for i in range(320):
            if continuous_available_fs[i] == 1:
                count += 1
                if count >= required_fs_count:
                    return i - required_fs_count + 1
            else:
                count = 0

        return None

    def monitor_network_status(self):
        """Get the maximum used frequency slot index across all edges and cores."""
        max_used_fs = torch.max(torch.where(self.network_state == 0, torch.arange(320), torch.tensor(0)))
        status = max_used_fs.item()
        return status

    def allocate_communication_request(self, s, d, tr, k):
        """
        Allocate resources for a communication request using First-Fit algorithm.
        
        Args:
            s: Source node
            d: Destination node
            tr: Traffic demand in Gbps
            k: Maximum number of candidate paths to consider
        """
        global G, PATH_CACHE

        k_shortest_paths = PATH_CACHE.get((s, d), [])
        if not k_shortest_paths:
            return "No paths available."
        if len(k_shortest_paths) == 0:
            return "No paths available."

        fs_requirements = []
        filtered_paths = [path for path in k_shortest_paths
                          if sum(G.get_edge_data(u, v).get('distance', 0) for u, v in zip(path, path[1:])) < 6300]
        filtered_paths = filtered_paths[:k]

        for path in filtered_paths[:k]:
            path_length = sum(G.get_edge_data(u, v).get('distance', 0) for u, v in zip(path, path[1:]))
            if path_length > 6300:
                continue
            # Distance-adaptive modulation selection
            if path_length <= 600:
                fs_needed = ceil(tr / 200) * 3 + 1  # DP-16QAM
            elif path_length <= 1200:
                fs_needed = ceil(tr / 150) * 3 + 1  # DP-8QAM
            elif path_length <= 3500:
                fs_needed = ceil(tr / 100) * 3 + 1  # DP-QPSK
            else:
                fs_needed = ceil(tr / 50) * 3 + 1   # DP-BPSK

            fs_requirements.append(fs_needed)

        best_max_used_fs = float('inf')
        best_path = None
        best_max_used_fs_inpath = float('inf')

        # First-Fit: Find the allocation that minimizes FSmax
        for path, fs_needed in zip(filtered_paths, fs_requirements):
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            for core in range(self.num_cores):
                min_fs = self.find_min_fs_with_continuous_availability(edges, core, fs_needed)
                if min_fs is not None:
                    temp_network_state = self.network_state.clone()
                    for s, d in edges:
                        temp_network_state[self.get_edge_index(s, d), core, min_fs:min_fs + fs_needed] = 0
                    max_used_fs = torch.max(torch.where(temp_network_state == 0, torch.arange(320), torch.tensor(0)))
                    max_used_fs_inpath = min_fs + fs_needed

                    if max_used_fs <= best_max_used_fs and max_used_fs_inpath < best_max_used_fs_inpath:
                        best_max_used_fs = max_used_fs
                        best_max_used_fs_inpath = max_used_fs_inpath
                        best_path = (path, core, min_fs, fs_needed)

        if best_path:
            path, core, min_fs, fs_needed = best_path
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            for s, d in edges:
                self.network_state[self.get_edge_index(s, d), core, min_fs:min_fs + fs_needed] = 0


class PerformanceConfig:
    """Performance Configuration Manager for multi-process execution."""

    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()

    def _calculate_optimal_workers(self):
        """Calculate optimal number of worker processes."""
        phys = psutil.cpu_count(logical=False) or (self.cpu_count // 2)
        # Default 90% physical cores; allow environment variable override
        override = os.getenv("RMSSA_WORKERS")
        if override:
            return max(1, int(override))
        return max(1, int(phys * 0.9))

    def setup_environment(self):
        """Setup optimization environment variables for multi-processing."""
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['MALLOC_ARENA_MAX'] = '2'

        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)

    def print_info(self):
        """Print system configuration info."""
        print(f"=== System Performance Configuration ===")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Optimal Workers: {self.optimal_workers}")
        print("========================================")


# Global Variables
K_PATHS = 3
G = None
NUM_NODES = None
TOPO_NAME = None
PATH_CACHE = {}
_perf_config = None


def get_performance_config():
    """Get or create performance configuration singleton."""
    global _perf_config
    if _perf_config is None:
        _perf_config = PerformanceConfig()
        _perf_config.setup_environment()
    return _perf_config


def trf_l2s(dicts_list):
    """Sort requests by traffic demand in descending order (Large to Small)."""
    new_dict = []
    for d in dicts_list:
        valid_items = []
        for key, value in d.items():
            if isinstance(value, (list, tuple)) and len(value) > 2:
                valid_items.append((key, value))
            else:
                print(f"Warning: Value {value} for key {key} cannot be sorted, skipped")
        sorted_items = sorted(valid_items, key=lambda item: item[1][2], reverse=True)
        new_dict.append(dict(sorted_items))
    return new_dict


def trf_s2l(dicts_list):
    """Sort requests by traffic demand in ascending order (Small to Large)."""
    new_dict = []
    for dicts in dicts_list:
        d = dict(sorted(dicts.items(), key=lambda item: item[1][2]))
        new_dict.append(d)
    return new_dict


def reader(tensor):
    """Convert tensor to list of request dictionaries."""
    shape = tensor.shape
    output_dict = {}

    R_list = []
    for i in range(shape[0]):
        value = tensor[i, :, :].tolist()
        output_dict[i] = value
        R = {}
        for r in range(shape[1]):
            R[r] = output_dict[i][r]
        R_list.append(R)

    return R_list


def _init_pool_processes(graph, k_paths, topo_name):
    """
    Initialize worker process with topology data.
    Called when each subprocess starts.
    """
    global G, K_PATHS, TOPO_NAME, PATH_CACHE
    G = graph
    K_PATHS = k_paths
    TOPO_NAME = topo_name.upper()

    cache_data = ksp_cache.build_or_load(G, TOPO_NAME, k_paths=k_paths)
    PATH_CACHE = cache_data.get('paths', cache_data)


def process_communication_requests(all_requests):
    """
    Process a set of communication requests using First-Fit allocation.
    
    Args:
        all_requests: Dictionary mapping request IDs to [source, dest, traffic] lists
        
    Returns:
        Maximum used frequency slot index after all allocations
    """
    global G
    if G is None:
        raise RuntimeError(
            "Global topology G is None. "
            "Please run set_topology() before calling multi_process."
        )

    # Create an independent OpticalNetwork instance for this subprocess
    network = OpticalNetwork(G, device="cpu")

    # Iterate and allocate all requests
    for s, d, tr in all_requests.values():
        network.allocate_communication_request(s, d, tr, K_PATHS)

    # Return maximum frequency slot position
    max_used_fs = network.monitor_network_status()
    return max_used_fs


def set_topology(topo_name: str, k_paths: int = 3):
    """
    Initialize topology and cache K-Shortest Paths.
    
    Args:
        topo_name: Topology name ('NSF', 'N6S9', or 'EURO16')
        k_paths: Number of shortest paths to cache
        
    Returns:
        Number of nodes in the topology
    """
    global G, NUM_NODES, K_PATHS, TOPO_NAME, PATH_CACHE
    from topology_loader import load_topology

    perf_config = get_performance_config()
    perf_config.print_info()

    G, NUM_NODES = load_topology(topo_name)
    K_PATHS = int(k_paths)
    TOPO_NAME = topo_name.upper()

    cache_data = ksp_cache.build_or_load(G, TOPO_NAME, k_paths=K_PATHS)
    PATH_CACHE = cache_data.get('paths', cache_data)

    print(f"[RMSSA_environment] Topology {TOPO_NAME} (|V|={NUM_NODES})  "
          f"k_paths={K_PATHS}  Cache entries={len(PATH_CACHE)}")
    return NUM_NODES


def multi_process(sample_dict_list):
    """
    Evaluate multiple request sets in parallel using ProcessPoolExecutor.
    
    Args:
        sample_dict_list: List of request dictionaries to evaluate
        
    Returns:
        Tensor of FSmax values for each sample
    """
    global G

    perf_config = get_performance_config()

    if G is None:
        raise RuntimeError('Please call set_topology() to initialize G first')

    max_workers = perf_config.optimal_workers

    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_pool_processes,
            initargs=(G, K_PATHS, TOPO_NAME),
    ) as pool:
        results = list(pool.map(process_communication_requests, sample_dict_list))

    return torch.tensor(results)


def benchmark_performance():
    """Run performance benchmark on the current topology."""
    print("=== Performance Benchmark ===")

    import time
    import numpy as np

    # Create test data
    print("Generating test data...")
    test_samples = []
    num_samples = 100
    requests_per_sample = 50

    for i in range(num_samples):
        sample = {}
        for j in range(requests_per_sample):
            s = np.random.randint(0, NUM_NODES)
            d = np.random.randint(0, NUM_NODES)
            while d == s:
                d = np.random.randint(0, NUM_NODES)
            tr = np.random.randint(100, 1000)
            sample[j] = [s, d, tr]
        test_samples.append(sample)

    print(f"Test Config: {num_samples} samples, {requests_per_sample} requests per sample")

    # Run benchmark
    start_time = time.time()
    results = multi_process(test_samples)
    end_time = time.time()

    # Statistics
    elapsed_time = end_time - start_time
    throughput = len(test_samples) / elapsed_time
    avg_result = results.float().mean().item()

    print(f"=== Test Results ===")
    print(f"Total Time: {elapsed_time:.2f} s")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Avg FS Usage: {avg_result:.2f}")
    print(f"Avg Time per Sample: {elapsed_time / len(test_samples) * 1000:.2f} ms")
    print("====================")

    return {
        'elapsed_time': elapsed_time,
        'throughput': throughput,
        'avg_result': avg_result,
        'samples_processed': len(test_samples)
    }


def cleanup_environment():
    """Cleanup global environment resources."""
    global G, PATH_CACHE, _perf_config
    G = None
    PATH_CACHE = {}
    _perf_config = None


# ============================================================================
# Heuristic Ordering Functions
# ============================================================================

def dic_hop(dicts):
    """Calculate hop-count statistics for all requests."""
    all_paths = {}
    hop = {}
    min_hop = {}
    max_hop = {}
    ave_hop = {}
    for r in dicts:
        all_paths[r] = list(nx.all_simple_paths(G, source=dicts[r][0], target=dicts[r][1]))
        hop[r] = {}
        current_max_hop = 0
        current_min_hop = 10000
        sum_hop = 0
        for path in range(len(all_paths[r])):
            hop[r][path] = len(all_paths[r][path])
            sum_hop += hop[r][path]
            if current_max_hop < hop[r][path]:
                current_max_hop = hop[r][path]
            if current_min_hop > hop[r][path]:
                current_min_hop = hop[r][path]
        ave_hop[r] = sum_hop / len(hop[r]) if hop[r] else 0
        min_hop[r] = current_min_hop
        max_hop[r] = current_max_hop
    return min_hop, max_hop, ave_hop


def ave_hop_s2l(dicts):
    """Sort requests by average hop count, ascending."""
    dic_ave_hop = dic_hop(dicts)[2]
    d = sorted(dic_ave_hop.items(), key=lambda item: item[1])
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def ave_hop_l2s(dicts):
    """Sort requests by average hop count, descending."""
    dic_ave_hop = dic_hop(dicts)[2]
    d = sorted(dic_ave_hop.items(), key=lambda item: item[1], reverse=True)
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def min_hop_s2l(dicts):
    """Sort requests by minimum hop count, ascending."""
    dic_min_hop = dic_hop(dicts)[0]
    d = sorted(dic_min_hop.items(), key=lambda item: item[1])
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def min_hop_l2s(dicts):
    """Sort requests by minimum hop count, descending."""
    dic_min_hop = dic_hop(dicts)[0]
    d = sorted(dic_min_hop.items(), key=lambda item: item[1], reverse=True)
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def max_hop_s2l(dicts):
    """Sort requests by maximum hop count, ascending."""
    dic_max_hop = dic_hop(dicts)[1]
    d = sorted(dic_max_hop.items(), key=lambda item: item[1])
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def max_hop_l2s(dicts):
    """Sort requests by maximum hop count, descending."""
    dic_max_hop = dic_hop(dicts)[1]
    d = sorted(dic_max_hop.items(), key=lambda item: item[1], reverse=True)
    new_dic = {}
    for key in range(len(d)):
        new_dic[key] = dicts[d[key][0]]
    return new_dic


def random_dic(dicts):
    """Randomly shuffle request order."""
    import random
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    current_key = 0
    for key in dict_key_ls:
        if current_key < len(dicts):
            new_dic[current_key] = dicts.get(key)
            current_key += 1
    return new_dic
