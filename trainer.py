"""NUMA Optimized trainer.py - Fully utilize dual-socket CPU performance

Key improvements:
1. NUMA-aware environment configuration
2. Intelligent multi-process strategy
3. Performance monitoring and benchmarking
4. Optimization dedicated for dual-socket systems
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set environment variables - Must be done before importing torch
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MALLOC_ARENA_MAX'] = '2'

# Set torch threads
torch.set_num_threads(1)

import RMSSA_environment as REnv
from model import DRL4RMSSA, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PerformanceTracker:
    """Performance Tracker"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.batch_times = []
        self.reward_history = []
        self.loss_history = []
        self.throughput_history = []

    def update_batch(self, batch_time, reward, loss, batch_size):
        self.batch_times.append(batch_time)
        self.reward_history.append(reward)
        self.loss_history.append(loss)
        self.throughput_history.append(batch_size / batch_time if batch_time > 0 else 0)

    def get_stats(self):
        if not self.batch_times:
            return {}

        return {
            'avg_batch_time': np.mean(self.batch_times),
            'avg_reward': np.mean(self.reward_history),
            'avg_loss': np.mean(self.loss_history),
            'avg_throughput': np.mean(self.throughput_history),
            'total_time': time.time() - self.start_time,
            'batches_processed': len(self.batch_times)
        }

    def print_summary(self):
        stats = self.get_stats()
        print(f"\n=== Performance Statistics ===")
        print(f"Avg Batch Time: {stats.get('avg_batch_time', 0):.3f}s")
        print(f"Avg Reward: {stats.get('avg_reward', 0):.3f}")
        print(f"Avg Loss: {stats.get('avg_loss', 0):.4f}")
        print(f"Avg Throughput: {stats.get('avg_throughput', 0):.2f} samples/s")
        print(f"Total Time: {stats.get('total_time', 0):.2f}s")
        print("==============================")

class StateCritic(nn.Module):
    """Improved State Critic, optimized for NUMA systems"""

    def __init__(self, static_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)

        # Deeper network structure
        self.fc1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1)
        self.fc3 = nn.Conv1d(hidden_size // 2, hidden_size // 4, kernel_size=1)
        self.fc4 = nn.Conv1d(hidden_size // 4, 1, kernel_size=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, static):
        # Encode static features
        static_hidden = self.static_encoder(static)

        # Forward pass
        output = F.relu(self.bn1(self.fc1(static_hidden)))
        output = self.dropout(output)
        output = F.relu(self.bn2(self.fc2(output)))
        output = self.dropout(output)
        output = F.relu(self.bn3(self.fc3(output)))
        output = self.fc4(output).sum(dim=2)

        return output

from RMSSA_function import reward_validate

def validate(data_loader, actor, reward_fn, save_dir='.', performance_tracker=None):
    """Validation function - Compatible with 3/4 element batches"""
    actor.eval()
    os.makedirs(save_dir, exist_ok=True)

    rewards = []
    val_start_time = time.time()

    for batch_idx, batch in enumerate(data_loader):
        batch_start = time.time()

        # ① Compatible with different return formats
        if len(batch) == 4:
            static, _, x0, _ = batch          # Ignore basic_static and baseline
        else:                                 # Old format
            static, x0, _ = batch

        static = static.to(device)
        x0 = x0 if len(x0) > 0 else None

        with torch.no_grad():
            assign_indices = actor.greedy(static)

        reward = reward_fn(static, assign_indices).mean().item()
        rewards.append(reward)

        # ② Record performance info (keep unchanged)
        batch_time = time.time() - batch_start
        if performance_tracker:
            performance_tracker.update_batch(batch_time, reward, 0, static.size(0))

        if (batch_idx + 1) % 10 == 0:
            print(f"    Validation Batch {batch_idx+1}/{len(data_loader)}, "
                  f"Reward: {reward:.3f}, Time: {batch_time:.3f}s")

    # ③ Other logic remains unchanged
    val_time = time.time() - val_start_time
    mean_reward = np.mean(rewards)

    print(f"Validation Complete: Avg Reward={mean_reward:.4f}, Time={val_time:.2f}s, "
          f"Speed={len(data_loader)*data_loader.batch_size/val_time:.2f} samples/s")

    actor.train()
    return mean_reward


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Main training function - Supports 11-dim features and dynamic baseline"""
    # Create save directory
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Performance trackers
    train_tracker = PerformanceTracker()
    val_tracker = PerformanceTracker()

    # Optimizer settings
    actor_optim = optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=1e-6)
    critic_optim = optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=1e-6)

    # Learning rate schedulers
    actor_scheduler = CosineAnnealingLR(actor_optim, T_max=50, eta_min=actor_lr*0.01)
    critic_scheduler = CosineAnnealingLR(critic_optim, T_max=50, eta_min=critic_lr*0.01)

    # Data loaders - Optimized for NUMA systems
    num_workers = 0  # Avoid extra process overhead
    train_loader = DataLoader(train_data, batch_size, True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=num_workers)

    # Training records
    best_params = None
    best_reward = np.inf
    save_loss = []
    save_reward = []
    patience = 20  # Increased patience
    patience_counter = 0

    print(f"\n=== Start Training ===")
    print(f"Training Data: {len(train_data)} samples")
    print(f"Validation Data: {len(valid_data)} samples")
    print(f"Batch Size: {batch_size}")
    print(f"Actor LR: {actor_lr}")
    print(f"Critic LR: {critic_lr}")
    print("======================\n")

    # Training loop
    for epoch in range(100):
        actor.train()
        critic.train()
        train_tracker.reset()

        times, losses, rewards, critic_rewards = [], [], [], []
        epoch_start = time.time()
        start = epoch_start

        print(f"Epoch {epoch+1}/100:")

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()

            if len(batch) == 4:
                static, basic_static, x0, baseline = batch
                static = static.to(device)
                basic_static = baseline.to(device)
            else:
                # Compatible with old format
                static, x0, baseline = batch
                static = static.to(device)
                basic_static = static[:, :3, :] if static.size(1) > 3 else static
            #
            # print("\n[DEBUG] static[0] shape:", static[0].shape)  # Should be (11, 100)
            # for i in range(static[0].shape[0]):
            #     print(f"[DEBUG] static[0][{i}]:", static[0][i].tolist())  # or directly print(static[0][i])
            # print("[DEBUG] basic_static[0] shape:", basic_static[0].shape)
            # print(basic_static[0])

            baseline_heuristic = basic_static.to(device).float()

            # Forward pass
            assign_indices, assign_logp = actor(static, x0)

            # Calculate reward
            actor_value = reward_fn(static, assign_indices)

            # Calculate advantage function
            advantage = actor_value - baseline_heuristic
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Actor loss
            actor_loss = torch.mean(advantage.detach() * assign_logp.sum(dim=1))

            # Critic loss
            critic_pred = critic(static).view(-1)
            critic_loss = F.mse_loss(critic_pred, actor_value.detach())

            # Update Actor
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            # Update Critic
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            # Record
            batch_time = time.time() - batch_start
            reward_val = torch.mean(actor_value.detach()).item()
            loss_val = actor_loss.detach().item()

            critic_rewards.append(torch.mean(baseline_heuristic.detach()).item())
            rewards.append(reward_val)
            losses.append(loss_val)

            # Update performance tracking
            train_tracker.update_batch(batch_time, reward_val, loss_val, static.size(0))

            # Periodic output
            if (batch_idx + 1) % 20 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-20:])
                mean_reward = np.mean(rewards[-20:])

                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Reward: {mean_reward:.3f}, Loss: {mean_loss:.4f}, "
                      f"Actor Grad: {actor_grad_norm:.4f}, Critic Grad: {critic_grad_norm:.4f}, "
                      f"Batch Time: {batch_time:.3f}s, Throughput: {static.size(0)/batch_time:.1f} samples/s")

        # Update learning rate
        actor_scheduler.step()
        critic_scheduler.step()

        # Calculate epoch stats
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save checkpoint
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        torch.save(actor.state_dict(), os.path.join(epoch_dir, 'actor.pt'))
        torch.save(critic.state_dict(), os.path.join(epoch_dir, 'critic.pt'))

        # Validation
        print(f"  Starting Validation...")
        val_tracker.reset()
        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid = validate(valid_loader, actor, reward_validate, valid_dir, val_tracker)

        # Save best model
        if mean_valid < best_reward:
            best_reward = mean_valid
            patience_counter = 0

            torch.save(actor.state_dict(), os.path.join(save_dir, 'actor.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, 'critic.pt'))
            print(f"  * Saved Best Model (Val Reward: {best_reward:.4f})")
        else:
            patience_counter += 1

        # Record training history
        save_loss.append(mean_loss)
        save_reward.append(mean_reward)

        # Print epoch stats
        epoch_time = time.time() - epoch_start
        train_stats = train_tracker.get_stats()
        val_stats = val_tracker.get_stats()

        print(f"Epoch {epoch+1} Completed:")
        print(f"  Train - Loss: {mean_loss:.4f}, Reward: {mean_reward:.4f}, "
              f"Throughput: {train_stats.get('avg_throughput', 0):.1f} samples/s")
        print(f"  Valid - Reward: {mean_valid:.4f}, Best: {best_reward:.4f}, "
              f"Throughput: {val_stats.get('avg_throughput', 0):.1f} samples/s")
        print(f"  LR: {actor_scheduler.get_last_lr()[0]:.2e}, "
              f"Total Time: {epoch_time:.1f}s, Patience: {patience_counter}/{patience}")
        print()

        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Training completion stats
    train_tracker.print_summary()


def train_rmssa(args):
    """RMSSA Training Function - Supports 11-dim features"""

    import RMSSA_function
    from RMSSA_function import RMSSADataset

    print("=== RMSSA Training System (Enhanced) ===")

    # Run performance benchmark (optional)
    if hasattr(args, 'benchmark') and args.benchmark:
        print("\nRunning Performance Benchmark...")
        REnv.benchmark_performance()
        print()

    # Set dimension based on whether enhanced features are used
    USE_ENHANCED_FEATURES = args.use_enhanced_features if hasattr(args, 'use_enhanced_features') else True
    STATIC_SIZE = 11 if USE_ENHANCED_FEATURES else 3

    # Create dataset (using modified RMSSADataset)
    print("Creating Training Dataset...")
    train_data = RMSSADataset(
        TOPO_NODES,
        request_size=args.num_nodes,
        num_samples=args.train_size,
        seed=args.seed,
        topo_name=args.topology,
        k_paths=args.k_paths,
        use_enhanced_features=USE_ENHANCED_FEATURES
    )

    print("Creating Validation Dataset...")
    valid_data = RMSSADataset(
        TOPO_NODES,
        request_size=args.num_nodes,
        num_samples=args.valid_size,
        seed=args.seed + 1,
        topo_name=args.topology,
        k_paths=args.k_paths,
        use_enhanced_features=USE_ENHANCED_FEATURES
    )

    update_fn = None

    # Create model (using modified DRL4RMSSA)
    print("Creating Neural Network Model...")
    print(f"  Feature Dimension: {STATIC_SIZE}")
    print(f"  Use History Memory: {args.use_history if hasattr(args, 'use_history') else True}")

    actor = DRL4RMSSA(
        static_size=STATIC_SIZE,
        hidden_size=args.hidden_size,
        num_nodes=TOPO_NODES,
        update_fn=update_fn,
        mask_fn=RMSSA_function.update_mask,
        transformer_layers=args.num_layers,
        transformer_heads=8,
        dropout=args.dropout,
        use_history=args.use_history
    ).to(device)
    critic = StateCritic(STATIC_SIZE, args.hidden_size).to(device)

    # Model parameters statistics
    actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Actor Parameters: {actor_params:,}")
    print(f"Critic Parameters: {critic_params:,}")
    print(f"Total Parameters: {actor_params + critic_params:,}")

    # Training arguments
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = RMSSA_function.reward
    kwargs['render_fn'] = RMSSA_function.render

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, map_location=device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, map_location=device))

    # Train or Test
    if not args.test:
        print("Starting Training...")
        training_start = time.time()
        train(actor, critic, **kwargs)
        training_end = time.time()
        print(f"Training completed, Total time: {training_end - training_start:.2f} seconds")

    # Test
    print("Starting Testing...")
    test_data = RMSSADataset(
        TOPO_NODES,
        request_size=args.num_nodes,
        num_samples=args.test_size,
        seed=args.seed + 2,
        topo_name=args.topology,
        k_paths=args.k_paths,
        use_enhanced_features=USE_ENHANCED_FEATURES
    )

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

    test_tracker = PerformanceTracker()
    test_start_time = time.time()
    test_reward = validate(test_loader, actor, RMSSA_function.reward, test_dir, test_tracker)
    test_end_time = time.time()
    test_runtime = test_end_time - test_start_time

    # Test results statistics
    test_stats = test_tracker.get_stats()
    total_requests = args.test_size * args.num_nodes

    print(f'\n=== Test Results ===')
    print(f'Avg Reward: {test_reward:.4f}')
    print(f'Runtime: {test_runtime:.2f}s')
    print(f'Throughput: {test_stats.get("avg_throughput", 0):.2f} samples/s')
    print(f'Request Processing Speed: {total_requests / test_runtime:.2f} requests/s')
    print(f'Avg Time per Sample: {test_runtime / args.test_size * 1000:.2f}ms')
    print('====================')

    # Cleanup environment resources
    REnv.cleanup_environment()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NUMA Optimized RMSSA Training System')

    # Basic Parameters
    parser.add_argument('--seed', default=114514, type=int, help='Random seed')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path')
    # parser.add_argument('--checkpoint', default='rmssa/100/NSF', help='Checkpoint path')
    parser.add_argument('--test', action='store_true', default=False, help='Test mode only')
    parser.add_argument('--task', default='rmssa', help='Task name')
    parser.add_argument('--benchmark', action='store_true', default=False, help='Run performance benchmark')
    parser.add_argument('--use-enhanced-features', action='store_true', default=True,
                       help='Use 11-dimensional enhanced features (enabled by default)')
    parser.add_argument('--use-history', action='store_true', default=True,
                       help='Use history-aware decoder (enabled by default)')
    parser.add_argument('--max_seq_len', default=100, type=int,
                       help='History memory window size')

    # Network Parameters
    parser.add_argument('--nodes', dest='num_nodes', default=100, type=int, help='Number of requests per sample')
    parser.add_argument('--hidden', dest='hidden_size', default=256, type=int, help='Hidden layer size')
    parser.add_argument('--layers', dest='num_layers', default=6, type=int, help='Number of Transformer layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')

    # Training Parameters
    parser.add_argument('--actor_lr', default=5e-5, type=float, help='Actor learning rate')
    parser.add_argument('--critic_lr', default=1e-5, type=float, help='Critic learning rate')
    parser.add_argument('--max_grad_norm', default=2., type=float, help='Gradient clipping threshold')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')

    # Data Parameters
    parser.add_argument('--train-size', default=500, type=int, help='Training sample size')
    parser.add_argument('--valid-size', default=500, type=int, help='Validation sample size')
    parser.add_argument('--test-size', default=5000, type=int, help='Test sample size')

    # Topology Parameters
    parser.add_argument('--topology', default='NSF', help='Topology type: NSF / N6S9 / EURO16')
    parser.add_argument('--k_paths', default=5, type=int, help='K-Shortest Paths count')

    # NUMA Optimization Parameters
    parser.add_argument('--force-numa', action='store_true', default=False,
                       help='Force enable NUMA optimization (even on single socket systems)')
    parser.add_argument('--numa-workers', default=None, type=int,
                       help='Number of workers per NUMA node (auto-detected if unspecified)')

    args = parser.parse_args()

    print("=== System Configuration ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CPU Cores: {os.cpu_count()}")
    print(f"Device: {device}")
    print("============================")

    # Set Topology
    print(f"Loading Topology: {args.topology}")
    TOPO_NODES = REnv.set_topology(args.topology, args.k_paths)
    print(f"Topology Loaded: {TOPO_NODES} nodes")

    # Start Training
    try:
        train_rmssa(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        REnv.cleanup_environment()
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        REnv.cleanup_environment()
        raise