# Trans-Ptr: A Transformer-Pointer DRL Model for Static Resource Allocation in SDM-EONs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of our paper:

> **A Transformer–Pointer DRL Model for Static Resource Allocation Problems in SDM-EONs**
> 
> Sibo Chen, Jiading Wang, and Maiko Shigeno
> 
## Overview

We propose a deep reinforcement learning approach for solving the static **RMSSA** (Routing, Modulation, Spectrum and Space Assignment) problem in **SDM-EONs** (Space Division Multiplexing Elastic Optical Networks). Our method reformulates the combinatorial optimization problem as a learned ordering task:

- **Transformer Encoder**: Extracts inter-request relationships through multi-head self-attention
- **Pointer Network Decoder**: Generates optimized request orderings autoregressively
- **First-Fit Executor**: Performs deterministic spectrum-space assignment following the learned order

### Key Contributions

1. **Theoretical Foundation**: We extend the order-expressiveness property from SA to SSA, proving that First-Fit can achieve optimal solutions under appropriate request ordering

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

n trainer.py --topology NSF --no-use-enhanced-features
```

### Testing

```bash
# Test from checkpoint
python trainer.py --topology NSF --test --checkpoint path/to/checkpoint
```

```

## Acknowledgments

This work was supported by the Japan Science and Technology Agency (JST) through the Next Generation (SPRING) program, Grant Number JPMJSP2124.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Sibo Chen** - s2330125@u.tsukuba.ac.jp
- Systems and Information Engineering, University of Tsukuba, Japan
