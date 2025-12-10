# Trans-Ptr: A Transformer-Pointer DRL Model for Static Resource Allocation in SDM-EONs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of our paper:

> **A Transformer–Pointer DRL Model for Static Resource Allocation Problems in SDM-EONs**
> 
> Sibo Chen, Jiading Wang, and Maiko Shigeno
> 
> *Journal of Optical Communications and Networking (JOCN)*

## Overview

We propose a deep reinforcement learning approach for solving the static **RMSSA** (Routing, Modulation, Spectrum and Space Assignment) problem in **SDM-EONs** (Space Division Multiplexing Elastic Optical Networks). Our method reformulates the combinatorial optimization problem as a learned ordering task:

- **Transformer Encoder**: Extracts inter-request relationships through multi-head self-attention
- **Pointer Network Decoder**: Generates optimized request orderings autoregressively
- **First-Fit Executor**: Performs deterministic spectrum-space assignment following the learned order

### Key Contributions

1. **Theoretical Foundation**: We extend the order-expressiveness property from SA to SSA, proving that First-Fit can achieve optimal solutions under appropriate request ordering (Theorem 1)

2. **Neural Architecture**: A Transformer-Pointer network that learns to discover effective ordering strategies directly from data

3. **Practical Performance**: Achieves solution quality comparable to ILP/metaheuristics with inference speed approaching simple heuristics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RMSSA Environment                         │
├─────────────────────────────────────────────────────────────┤
│  Request Generator → Preprocessor → 11-dim Features          │
│         ↓                                                    │
│  ┌─────────────────┐     ┌────────────────────┐             │
│  │ Transformer     │     │ Pointer Network    │             │
│  │ Encoder (6L×8H) │ ──→ │ Decoder (LSTM)     │             │
│  └─────────────────┘     └────────────────────┘             │
│         ↓                         ↓                          │
│    Request Embeddings       Ordering π                      │
│                                   ↓                          │
│                          First-Fit Executor                  │
│                                   ↓                          │
│                         Resource Allocation                  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Trans-Ptr-RMSSA.git
cd Trans-Ptr-RMSSA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Trans-Ptr-RMSSA/
├── model.py              # Neural network architectures
│                         # - RequestTransformerEncoder
│                         # - PointerDecoder / HistoryAwarePointerDecoder
│                         # - DRL4RMSSA (main model)
│
├── trainer.py            # Training pipeline
│                         # - REINFORCE with baseline
│                         # - Performance tracking
│                         # - Early stopping
│
├── RMSSA_environment.py  # Optical network simulation
│                         # - OpticalNetwork class
│                         # - First-Fit allocation
│                         # - Multi-process execution
│
├── RMSSA_function.py     # Dataset and reward computation
│                         # - RMSSADataset (11-dim features)
│                         # - Reward calculation
│
├── topology_loader.py    # Network topology definitions
│                         # - NSF (14 nodes, 21 links)
│                         # - N6S9 (6 nodes, 9 links)
│                         # - EURO16 (16 nodes, 23 links)
│
├── ksp_cache.py          # K-Shortest Path caching
│                         # - Pre-computation of paths
│                         # - Path feature extraction
│
├── requirements.txt      # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## Usage

### Training

```bash
# Train on NSF topology with default settings
python trainer.py --topology NSF --nodes 100

# Train on EURO16 with custom hyperparameters
python trainer.py --topology EURO16 \
    --nodes 100 \
    --hidden 256 \
    --layers 6 \
    --batch_size 100 \
    --actor_lr 5e-5

# Train without enhanced features (3-dim only)
python trainer.py --topology NSF --no-use-enhanced-features
```

### Testing

```bash
# Test from checkpoint
python trainer.py --topology NSF --test --checkpoint path/to/checkpoint
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--topology` | NSF | Network topology (NSF/N6S9/EURO16) |
| `--nodes` | 100 | Number of requests per instance |
| `--hidden` | 256 | Hidden layer dimension |
| `--layers` | 6 | Number of Transformer layers |
| `--batch_size` | 100 | Training batch size |
| `--actor_lr` | 5e-5 | Actor learning rate |
| `--train-size` | 500 | Number of training samples |
| `--valid-size` | 500 | Number of validation samples |
| `--k_paths` | 5 | K-Shortest paths for routing |
| `--use-enhanced-features` | True | Use 11-dim features |
| `--use-history` | True | Use history-aware decoder |

## Experimental Results

Performance comparison on three topologies (FSmax: lower is better):

| Topology | R Size | Trans-Ptr | Trans-Ptr(Mix) | FF-G(tr_desc) | SA | ILP |
|----------|--------|-----------|----------------|---------------|-----|------|
| **NSF** | 50 | 56.13 | **56.05** | 56.09 | 56.07 | 56.03 |
| | 100* | 65.37 | **64.49** | 67.28 | 66.72 | 61.51 |
| | 150 | 87.36 | **86.37** | 89.24 | 88.67 | 92.10 |
| **N6S9** | 50 | 33.14 | **32.61** | 33.41 | 32.97 | 31.36 |
| | 100* | 53.64 | **53.17** | 54.36 | 53.88 | 50.33 |
| | 150 | 76.19 | **75.57** | 76.73 | 75.86 | 83.26 |
| **EURO16** | 50 | 32.41 | **31.84** | 33.09 | 32.03 | 30.09 |
| | 100* | 49.24 | **48.68** | 50.35 | 49.27 | 55.29 |
| | 150 | 68.00 | **67.52** | 69.11 | 68.04 | 80.13 |

*\* indicates training size; models generalize to other sizes without retraining*

## Feature Description

The 11-dimensional input features for each request:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | s | Source node ID |
| 1 | d | Destination node ID |
| 2 | tr | Traffic demand (Gbps) |
| 3 | Lavg | Average path length |
| 4 | Lmin | Minimum path length |
| 5 | Havg | Average hop count |
| 6 | Hmin | Minimum hop count |
| 7 | Mavg | Average modulation level |
| 8 | FSmin | Minimum FS requirement |
| 9 | FSavg | Average FS requirement |
| 10 | FSmax | Maximum FS requirement |

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{chen2025transformer,
  title={A Transformer--Pointer DRL Model for Static Resource Allocation Problems in SDM-EONs},
  author={Chen, Sibo and Wang, Jiading and Shigeno, Maiko},
  journal={Journal of Optical Communications and Networking},
  year={2025},
  publisher={Optica Publishing Group}
}
```

## Acknowledgments

This work was supported by the Japan Science and Technology Agency (JST) through the Next Generation (SPRING) program, Grant Number JPMJSP2124.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Sibo Chen** - s2330125@u.tsukuba.ac.jp
- Systems and Information Engineering, University of Tsukuba, Japan
