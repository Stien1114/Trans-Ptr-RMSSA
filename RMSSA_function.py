"""Defines the main task for the RMSSA

Each request in the list must be assigned once and only once

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RMSSA_environment
import ksp_cache
from typing import Optional
from topology_loader import load_topology
from math import ceil


class RMSSADataset(Dataset):
    """
    Improved dataset class, correctly handling 11-dimensional features.
    """
    STATIC_SIZE = 11

    def __init__(self, topo_nodes: int,
                 request_size: int = 100,
                 num_samples: int = 10,
                 seed: Optional[int] = None,
                 topo_name: str = 'NSF',
                 k_paths: int = 3,
                 use_enhanced_features: bool = True):

        if seed is None:
            seed = np.random.randint(123456789)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.topo_nodes = topo_nodes
        self.request_size = request_size
        self.num_samples = num_samples
        self.topo_name = topo_name
        self.k_paths = k_paths
        self.use_enhanced_features = use_enhanced_features

        # Load enhanced KSP cache
        if use_enhanced_features:
            G, _ = load_topology(topo_name)
            cache_data = ksp_cache.build_or_load(
                G, topo_name.upper(), cache_size=50, k_paths=k_paths
            )
            self.paths_cache = cache_data.get('paths', {})
            self.features_cache = cache_data.get('features', {})
            self.STATIC_SIZE = 11
        else:
            self.STATIC_SIZE = 3

        # Generate dataset
        basic_dataset_list = []
        enhanced_dataset_list = []

        for _ in range(num_samples):
            basic_requests = []
            enhanced_requests = []

            while len(basic_requests) < request_size:
                s = int(torch.randint(low=0, high=self.topo_nodes, size=(1,)).item())
                d = int(torch.randint(low=0, high=self.topo_nodes, size=(1,)).item())
                if s != d:
                    tr = int(torch.randint(low=100, high=1000, size=(1,)).item())

                    # Basic features
                    basic_features = [s, d, tr]
                    basic_requests.append(basic_features)

                    if use_enhanced_features:
                        # Get pre-computed path features
                        path_features = self.features_cache.get((s, d), {
                            'avg_length': 0.0,
                            'min_length': 0.0,
                            'max_length': 0.0,
                            'avg_hops': 0.0,
                            'min_hops': 0.0,
                            'max_hops': 0.0,
                            'avg_modulation': 0.0,
                            'best_modulation': 0.0
                        })

                        # Calculate FS requirement
                        fs_min = self._calculate_fs(tr, path_features['best_modulation'])
                        fs_avg = self._calculate_fs(tr, path_features['avg_modulation'])
                        worst_modulation = ksp_cache._calculate_modulation_format(path_features['max_length'])
                        fs_max = self._calculate_fs(tr, worst_modulation if worst_modulation > 0 else 1)

                        # Combine all features - keep node IDs as original values!
                        enhanced_features = [
                            s, d, tr,  # Keep original values, no normalization here
                            path_features['avg_length'],
                            path_features['min_length'],
                            path_features['avg_hops'],
                            path_features['min_hops'],
                            path_features['avg_modulation'],
                            fs_min,
                            fs_avg,
                            fs_max
                        ]
                        enhanced_requests.append(enhanced_features)

            basic_tensor = torch.tensor(basic_requests, dtype=torch.float32)
            basic_dataset_list.append(basic_tensor)

            if use_enhanced_features:
                enhanced_tensor = torch.tensor(enhanced_requests, dtype=torch.float32)
                enhanced_dataset_list.append(enhanced_tensor)

        # Stack and permute
        self.basic_dataset = torch.stack(basic_dataset_list, dim=0)
        self.basic_dataset = self.basic_dataset.permute(0, 2, 1).float()

        if use_enhanced_features:
            self.dataset = torch.stack(enhanced_dataset_list, dim=0)
            self.dataset = self.dataset.permute(0, 2, 1).float()
            # Normalize only selected features, keep node IDs unchanged
            self._normalize_features_selective()
        else:
            self.dataset = self.basic_dataset

        # Record normalization params for model usage
        self.normalization_params = self._compute_normalization_params()

        # Compute baseline
        self._compute_baseline()
        self.dynamic_baseline = self.heuristic_baseline

    def _normalize_features_selective(self):
        """Selective normalization - keep node IDs unchanged"""
        # Do not normalize first 3 dimensions (s, d, tr)
        # Only normalize path and FS related features

        for feat_idx in range(3, 11):
            feat_data = self.dataset[:, feat_idx, :]

            if feat_idx in [3, 4]:  # Path length
                max_val = feat_data.max()
                min_val = feat_data.min()
                if max_val > min_val:
                    self.dataset[:, feat_idx, :] = (feat_data - min_val) / (max_val - min_val)
            elif feat_idx in [5, 6]:  # Hops
                max_hops = feat_data.max()
                if max_hops > 0:
                    self.dataset[:, feat_idx, :] = feat_data / max_hops
            elif feat_idx == 7:  # Modulation format
                self.dataset[:, feat_idx, :] = (feat_data - 1) / 3.0
            elif feat_idx in [8, 9, 10]:  # FS requirement
                self.dataset[:, feat_idx, :] = feat_data / 320.0

    def _compute_normalization_params(self):
        """Compute normalization parameters for model usage"""
        params = {}
        for feat_idx in range(2, 11):  # Start from tr
            feat_data = self.dataset[:, feat_idx, :]
            params[feat_idx] = {
                'mean': feat_data.mean().item(),
                'std': feat_data.std().item() + 1e-8,
                'min': feat_data.min().item(),
                'max': feat_data.max().item()
            }
        return params

    def _calculate_fs(self, traffic: float, modulation: float) -> float:
        """Calculate FS requirement based on traffic and modulation format"""
        if modulation == 0:
            return 320

        modulation = int(ceil(modulation))
        modulation = min(4, max(1, modulation))

        capacity_per_carrier = {
            1: 50, 2: 100, 3: 150, 4: 200
        }

        carriers_needed = ceil(traffic / capacity_per_carrier[modulation])
        fs_needed = carriers_needed * 3 + 1

        return float(fs_needed)

    def _compute_baseline(self):
        """Compute heuristic baseline"""
        requests_for_all_samples = []

        for i in range(self.num_samples):
            sample_basic = self.basic_dataset[i]
            sample_t = sample_basic.permute(1, 0)

            request_dict = {}
            for idx_r, r in enumerate(sample_t):
                s, d, tr = r.tolist()
                request_dict[idx_r] = [int(s), int(d), tr]
            requests_for_all_samples.append(request_dict)

        sorted_requests_all = RMSSA_environment.trf_l2s(requests_for_all_samples)
        fsmax_list = RMSSA_environment.multi_process(sorted_requests_all)
        print(f"Heuristic baseline: mean={fsmax_list.float().mean():.2f}")

        self.heuristic_baseline = fsmax_list.float()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        static = self.dataset[idx]
        basic_static = self.basic_dataset[idx]
        x0 = []

        if hasattr(self, 'dynamic_baseline'):
            baseline = self.dynamic_baseline[idx]
        else:
            baseline = self.heuristic_baseline[idx]

        return (static, basic_static, x0, baseline)


def update_mask(mask, chosen_idx):
    """Marks the assigned request, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, assign_indices):
    """
    Calculate reward (using basic features)

    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. s, d, tr) data
    assign_indices: torch.IntTensor of size (batch_size, num_request)

    Returns
    -------
    FS index max
    """
    # If static is 11-dim, take only the first 3 dims
    if static.size(1) > 3:
        static = static[:, :3, :]

    idx = assign_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    request_list = RMSSA_environment.reader(tour)
    FS_max = RMSSA_environment.multi_process(request_list)
    FS_max = FS_max.to("cuda:0" if torch.cuda.is_available() else "cpu").float()
    return FS_max.detach()


def reward_validate(static, assign_indices):
    """Reward calculation during validation"""
    # If static is 11-dim, take only the first 3 dims
    if static.size(1) > 3:
        static = static[:, :3, :]

    idx = assign_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    request_list = RMSSA_environment.reader(tour)
    FS_max = RMSSA_environment.multi_process(request_list)
    FS_max = FS_max.to("cuda:0" if torch.cuda.is_available() else "cpu").float()
    return FS_max.detach()


def render(static, assign_indices, save_path):
    """Plots """

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(assign_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = assign_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)