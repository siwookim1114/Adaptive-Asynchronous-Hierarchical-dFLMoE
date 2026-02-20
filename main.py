"""
Complete orchestration of federated learning system with:
- Multiple clients with heterogeneous data
- Hierarchical clustering
- Expert routing with FST
- Hybrid loss training
- Trust-based collaboration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

import time
import threading
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict
import numpy as np

# Import all components
from infra.cluster import ClusterManager
from client_node import ClientNode
from models.body_encoder import SimpleCNNBody
from models.head import Head
from utils.data_utils import DataPartitioner, get_dataset, verify_partitioning


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Federated Learning with Adaptive MoE')
    
    # System settings
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num_clusters', type=int, default=3, help='Number of clusters')
    parser.add_argument('--rounds', type=int, default=100, help='Number of training rounds')
    
    # Model settings
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--top_k_experts', type=int, default=3, help='Top-K experts to select')
    
    # Training settings
    parser.add_argument('--alpha', type=float, default=0.6, help='Hybrid loss weight (local)')
    parser.add_argument('--warmup_rounds', type=int, default=10, help='Rounds for alpha warm-up (1.0 to target alpha)')
    parser.add_argument('--local_epochs', type=int, default=5, help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_head', type=float, default=0.001, help='Learning rate for head')
    parser.add_argument('--lr_body', type=float, default=0.001, help='Learning rate for body')
    parser.add_argument('--lr_router', type=float, default=0.001, help='Learning rate for router')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='LR decay factor per round')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout in expert heads')

    # Data settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
    parser.add_argument('--partition_method', type=str, default='dirichlet',
                        choices=['iid', 'dirichlet', 'label_sharding'],
                        help='Data partitioning method')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5, help='Dirichlet alpha (lower=more non-IID)')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Classes per client (for label_sharding)')
    
    # Other settings
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name: str, root: str = './data'):
    """
    Load dataset with proper transforms (train augmentation + test normalization)

    Args:
        dataset_name: 'cifar10' or 'mnist'
        root: Data directory

    Returns:
        (train_dataset, test_dataset)
    """
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize 28x28 to 32x32 to match SimpleCNNBody
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    partition_method: str,
    batch_size: int,
    seed: int = 42,
    non_iid_alpha: float = 0.5,
    classes_per_client: int = 2,
    val_split: float = 0.1
):
    """
    Create partitioned data loaders using DataPartitioner from data_utils.py

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_clients: Number of clients
        partition_method: 'iid', 'dirichlet', or 'label_sharding'
        batch_size: Batch size
        seed: Random seed
        non_iid_alpha: Dirichlet alpha (for dirichlet method)
        classes_per_client: Classes per client (for label_sharding method)
        val_split: Validation split ratio

    Returns:
        (train_loaders, val_loaders, test_loader, train_partitioner)
    """
    # Partition the FULL training dataset first, then split each client's share into train/val.
    # This ensures each client's val set has the same class distribution as their train set.
    partitioner = DataPartitioner(
        train_dataset,
        num_clients=num_clients,
        method=partition_method,
        classes_per_client=classes_per_client,
        alpha=non_iid_alpha,
        seed=seed
    )

    # Print distribution
    partitioner.print_distribution()

    # For each client: split their indices into train (90%) / val (10%)
    train_loaders = []
    val_loaders = []

    for i in range(num_clients):
        client_indices = partitioner.client_indices[i]
        np.random.shuffle(client_indices)

        n_val = int(len(client_indices) * val_split)
        val_idx = client_indices[:n_val]
        train_idx = client_indices[n_val:]

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

        print(f"  Client {i}: {len(train_idx)} train, {len(val_idx)} val")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loaders, val_loaders, test_loader, partitioner


def create_models(feature_dim: int, num_classes: int, device: torch.device, dataset: str = 'cifar10', dropout: float = 0.0):
    """
    Create body encoder and head

    Args:
        feature_dim: Feature dimension
        num_classes: Number of classes
        device: Device
        dataset: Dataset name (determines input channels)
        dropout: Dropout probability for head

    Returns:
        (body_encoder, head)
    """
    input_channels = 1 if dataset == 'mnist' else 3
    body_encoder = SimpleCNNBody(input_channels=input_channels, output_dim=feature_dim).to(device)
    head = Head(input_dim=feature_dim, output_dim=num_classes, dropout=dropout).to(device)

    return body_encoder, head


def run_federated_learning(args):
    """
    Main federated learning orchestration
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("FEDERATED LEARNING WITH ADAPTIVE HIERARCHICAL MoE")
    print("="*70 + "\n")
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of clusters: {args.num_clusters}")
    print(f"Training rounds: {args.rounds}")
    print(f"Alpha (local weight): {args.alpha}")
    print(f"Warmup rounds: {args.warmup_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Top-K experts: {args.top_k_experts}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"LR decay: {args.lr_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Partition method: {args.partition_method}")
    if args.partition_method == 'dirichlet':
        print(f"Dirichlet alpha: {args.non_iid_alpha}")
    elif args.partition_method == 'label_sharding':
        print(f"Classes per client: {args.classes_per_client}")
    print()

    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}\n")

    # Partition data and create loaders using DataPartitioner
    print("Partitioning data...")
    train_loaders, val_loaders, test_loader, train_partitioner = create_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_clients=args.num_clients,
        partition_method=args.partition_method,
        batch_size=args.batch_size,
        seed=args.seed,
        non_iid_alpha=args.non_iid_alpha,
        classes_per_client=args.classes_per_client
    )
    print(f"  Created {len(train_loaders)} train loaders")
    print(f"  Created {len(val_loaders)} val loaders")

    print()

    # Create cluster manager
    print("Initializing cluster manager...")
    cluster_manager = ClusterManager(
        num_clusters=args.num_clusters
    )
    print(f"  Clusters: {args.num_clusters}\n")
    
    # Create clients
    print("Creating clients...")
    clients: List[ClientNode] = []
    
    for i in range(args.num_clients):
        client = ClientNode(
            client_id=f"client_{i}",
            cluster_manager=cluster_manager,
            host='127.0.0.1',
            port=0,  # Auto-assign
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_experts=args.num_clients,
            max_cache_size=args.num_clients,
            staleness_lambda=0.001,
            top_k_experts=args.top_k_experts,
            alpha=args.alpha,
            warmup_rounds=args.warmup_rounds,
            local_epochs=args.local_epochs,
            learning_rate_head=args.lr_head,
            learning_rate_body=args.lr_body,
            learning_rate_router=args.lr_router,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay
        )
        
        # Create models for this client
        body_encoder, head = create_models(args.feature_dim, args.num_classes, device, args.dataset, args.dropout)
        client.set_model(body_encoder, head)
        
        clients.append(client)
        
        if args.verbose:
            print(f"  Client {i}: {client.transport.port}")
    
    print(f"  Created {len(clients)} clients\n")
    
    # Register peer addresses for transport communication
    print("Registering peer addresses...")
    for i, client_i in enumerate(clients):
        for j, client_j in enumerate(clients):
            if i != j:
                _, port = client_j.transport.get_address()
                # Use localhost for local testing
                client_i.transport.register_peer(client_j.client_id, '127.0.0.1', port)
    time.sleep(1)  # Wait for registrations
    print("  All peer addresses registered\n")
    
    # Perform initial clustering
    print("Performing initial clustering...")
    cluster_manager.perform_clustering()
    
    # Print cluster assignments
    for cluster_id in range(args.num_clusters):
        members = cluster_manager.get_cluster_members(cluster_id)
        print(f"  Cluster {cluster_id}: {len(members)} members")
    print()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Statistics tracking
    global_stats = {
        'round': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': [],
        'avg_experts_used': []
    }
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}/{args.rounds}")
        print(f"{'='*70}\n")
        
        round_start_time = time.time()
        
        # Train all clients concurrently (asynchronous federated learning)
        # Each client trains in its own thread. While clients train, expert
        # packages from other clients can arrive via transport handlers,
        # enabling true asynchronous expert exchange.
        round_metrics = []

        def train_one_client(idx, client, train_loader, val_loader):
            """Train a single client in a thread."""
            try:
                metrics = client.train_round(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    use_experts=True
                )
                return idx, metrics, None
            except Exception as e:
                return idx, None, e

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_clients) as executor:
            futures = [
                executor.submit(train_one_client, i, client, train_loaders[i], val_loaders[i])
                for i, client in enumerate(clients)
            ]
            for future in concurrent.futures.as_completed(futures):
                idx, metrics, error = future.result()
                if error:
                    print(f"  Client {idx} error: {error}")
                elif metrics:
                    round_metrics.append(metrics)
                    if args.verbose or idx == 0:
                        print(f"Client {idx}: "
                              f"Train Loss={metrics['train_loss']:.4f}, "
                              f"Train Acc={metrics['train_acc']:.3f}, "
                              f"Val Acc={metrics['val_acc']:.3f}, "
                              f"Trust={metrics['trust_score']:.3f}, "
                              f"Cache={metrics['cache_size']}, "
                              f"Experts Used={metrics['experts_used']}")
        
        # Re-cluster every 5 rounds (after real features are computed)
        # Round 1: first recluster with real features (initial was random)
        if round_num == 1 or round_num % 5 == 0:
            cluster_manager.perform_clustering()

        # Aggregate statistics
        if len(round_metrics) > 0:
            avg_train_loss = np.mean([m['train_loss'] for m in round_metrics])
            avg_train_loss_local = np.mean([m['train_loss_local'] for m in round_metrics])
            avg_train_loss_moe = np.mean([m['train_loss_moe'] for m in round_metrics])
            avg_train_acc = np.mean([m['train_acc'] for m in round_metrics])
            avg_val_loss = np.mean([m['val_loss'] for m in round_metrics])
            avg_val_acc = np.mean([m['val_acc'] for m in round_metrics])
            avg_experts_used = np.mean([m['experts_used'] for m in round_metrics])
            
            # Global test every 5 rounds (and always on last round)
            if round_num % 5 == 0 or round_num == args.rounds:
                test_acc = evaluate_global(clients, test_loader, device)
                global_stats['test_acc'].append(test_acc)
            else:
                test_acc = None
            
            # Store statistics
            global_stats['round'].append(round_num)
            global_stats['train_loss'].append(avg_train_loss)
            global_stats['train_acc'].append(avg_train_acc)
            global_stats['val_loss'].append(avg_val_loss)
            global_stats['val_acc'].append(avg_val_acc)
            global_stats['avg_experts_used'].append(avg_experts_used)
            
            round_time = time.time() - round_start_time
            
            # Print round summary
            print(f"\n{'='*70}")
            print(f"ROUND {round_num} SUMMARY")
            print(f"{'='*70}")
            print(f"Avg Train Loss (total): {avg_train_loss:.4f}  [local: {avg_train_loss_local:.4f}, moe: {avg_train_loss_moe:.4f}]")
            print(f"Average Train Acc:      {avg_train_acc:.3f}")
            print(f"Average Val Loss:       {avg_val_loss:.4f}")
            print(f"Average Val Acc:        {avg_val_acc:.3f}")
            print(f"Avg Experts Used:   {avg_experts_used:.1f}")
            if 'current_alpha' in round_metrics[0]:
                print(f"Current Alpha:      {round_metrics[0]['current_alpha']:.3f}")
            if test_acc is not None:
                print(f"Global Test Acc:    {test_acc:.3f}")
            print(f"Round Time:         {round_time:.2f}s")
            print(f"{'='*70}\n")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70 + "\n")

    final_test_acc = evaluate_global(clients, test_loader, device)
    global_stats['final_test_acc'] = final_test_acc
    print(f"Final Global Test Accuracy: {final_test_acc:.3f}\n")

    # Collect per-client final statistics
    client_final_stats = []
    print("Final Statistics:")
    for i, client in enumerate(clients):
        stats = client.get_statistics()
        client_info = {
            'client_id': i,
            'rounds_completed': stats['rounds_completed'],
            'experts_sent': stats['experts_sent'],
            'experts_received': stats['experts_received'],
            'experts_registered': stats['experts_registered'],
            'experts_used': stats['experts_used'],
            'trust_score': client.trust_score,
            'validation_accuracy': client.validation_accuracy,
            'cache_size': stats['cache']['size']
        }
        client_final_stats.append(client_info)

        print(f"\nClient {i}:")
        print(f"  Rounds: {stats['rounds_completed']}")
        print(f"  Experts Sent: {stats['experts_sent']}")
        print(f"  Experts Received: {stats['experts_received']}")
        print(f"  Experts Registered: {stats['experts_registered']}")
        print(f"  Experts Used: {stats['experts_used']}")
        print(f"  Trust Score: {client.trust_score:.3f}")
        print(f"  Cache Size: {stats['cache']['size']}")

    global_stats['client_final_stats'] = client_final_stats

    # Shutdown
    print("\n" + "="*70)
    print("SHUTTING DOWN")
    print("="*70 + "\n")

    for client in clients:
        client.shutdown()

    print("Training complete!\n")

    return global_stats


def evaluate_global(
    clients: List[ClientNode],
    test_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluate global test accuracy using router-based MoE inference.

    Each client produces its prediction purely via its router:
    1. Router selects top-K experts from all available (including local head)
    2. Applies per-expert FST and weighted combination
    3. This MoE output IS the prediction (no fixed alpha combination)

    These MoE predictions are converted to softmax probabilities
    and trust-weighted ensembled across all clients.

    Args:
        clients: List of clients
        test_loader: Test data loader
        device: Device

    Returns:
        Ensemble test accuracy
    """
    num_classes = clients[0].num_classes
    correct = 0
    total = 0

    # Prepare all clients: set eval mode and build expert heads from cache
    client_expert_heads = []
    for client in clients:
        if client.body_encoder is not None:
            client.body_encoder.eval()
            client.head.eval()
            client.router.eval()

        # Build expert heads from this client's cache (once, not per batch)
        expert_heads = {}
        if client.cache.size() > 0:
            expert_packages = client.cache.get_available_experts(exclude_id=client.client_id)
            for pkg in expert_packages:
                eid = int(pkg.client_id.split('_')[-1]) if '_' in pkg.client_id else int(pkg.client_id)
                head_temp = Head(input_dim=client.feature_dim, output_dim=client.num_classes, dropout=client.dropout)
                head_temp.load_state_dict(pkg.head_state_dict)
                head_temp = head_temp.to(device)
                head_temp.eval()
                expert_heads[eid] = head_temp

        # Include local head in expert pool (already in eval mode)
        expert_heads[client.local_expert_id] = client.head

        client_expert_heads.append(expert_heads)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            ensemble_probs = torch.zeros(x.size(0), num_classes, device=device)
            total_weight = 0.0

            for i, client in enumerate(clients):
                if client.body_encoder is None or client.head is None:
                    continue

                features = client.body_encoder(x)

                expert_heads = client_expert_heads[i]
                if len(expert_heads) > 0:
                    # Pure MoE: router selects from all experts including local
                    predictions = client.router.forward_moe(
                        features, expert_heads, k=client.top_k_experts
                    )
                else:
                    predictions = client.head(features)

                probs = torch.softmax(predictions, dim=1)
                weight = client.trust_score
                ensemble_probs += weight * probs
                total_weight += weight

            if total_weight > 0:
                ensemble_probs /= total_weight

            _, predicted = ensemble_probs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return correct / total if total > 0 else 0.0


def save_results(args, global_stats):
    """
    Save comprehensive results to a text file in results/ directory

    Includes:
    - Full configuration
    - Per-round metrics (train loss/acc, val loss/acc, test acc, experts used)
    - Per-client final statistics
    - Final test accuracy
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = results_dir / f"results_{args.dataset}_{args.partition_method}_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEDERATED LEARNING WITH ADAPTIVE HIERARCHICAL MoE - RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Dataset:            {args.dataset}\n")
        f.write(f"  Num Clients:        {args.num_clients}\n")
        f.write(f"  Num Clusters:       {args.num_clusters}\n")
        f.write(f"  Training Rounds:    {args.rounds}\n")
        f.write(f"  Partition Method:   {args.partition_method}\n")
        if args.partition_method == 'dirichlet':
            f.write(f"  Dirichlet Alpha:    {args.non_iid_alpha}\n")
        elif args.partition_method == 'label_sharding':
            f.write(f"  Classes/Client:     {args.classes_per_client}\n")
        f.write(f"  Alpha (local wt):   {args.alpha}\n")
        f.write(f"  Warmup Rounds:      {args.warmup_rounds}\n")
        f.write(f"  Local Epochs:       {args.local_epochs}\n")
        f.write(f"  Top-K Experts:      {args.top_k_experts}\n")
        f.write(f"  Batch Size:         {args.batch_size}\n")
        f.write(f"  LR Head:            {args.lr_head}\n")
        f.write(f"  LR Body:            {args.lr_body}\n")
        f.write(f"  LR Router:          {args.lr_router}\n")
        f.write(f"  Weight Decay:       {args.weight_decay}\n")
        f.write(f"  LR Decay:           {args.lr_decay}\n")
        f.write(f"  Dropout:            {args.dropout}\n")
        f.write(f"  Feature Dim:        {args.feature_dim}\n")
        f.write(f"  Seed:               {args.seed}\n")
        f.write(f"  Device:             {args.device}\n\n")

        # Per-round metrics table
        f.write("PER-ROUND METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Round':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
                f"{'Val Loss':>8} | {'Val Acc':>7} | {'Test Acc':>8} | {'Experts':>7}\n")
        f.write("-" * 70 + "\n")

        test_acc_idx = 0
        for i, round_num in enumerate(global_stats['round']):
            train_loss = global_stats['train_loss'][i]
            train_acc = global_stats['train_acc'][i]
            val_loss = global_stats['val_loss'][i]
            val_acc = global_stats['val_acc'][i]
            experts = global_stats['avg_experts_used'][i]

            # Test acc is recorded every 5 rounds and on the last round
            if (round_num % 5 == 0 or round_num == len(global_stats['round'])) and test_acc_idx < len(global_stats['test_acc']):
                test_acc_str = f"{global_stats['test_acc'][test_acc_idx]:.4f}"
                test_acc_idx += 1
            else:
                test_acc_str = "   -   "

            f.write(f"{round_num:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
                    f"{val_loss:>8.4f} | {val_acc:>7.4f} | {test_acc_str:>8} | {experts:>7.1f}\n")

        f.write("\n")

        # Final test accuracy
        f.write("FINAL RESULTS\n")
        f.write("-" * 70 + "\n")
        final_test = global_stats.get('final_test_acc', 0.0)
        f.write(f"  Final Global Test Accuracy: {final_test:.4f}\n")

        # Best metrics
        if len(global_stats['val_acc']) > 0:
            best_val_acc = max(global_stats['val_acc'])
            best_val_round = global_stats['round'][global_stats['val_acc'].index(best_val_acc)]
            f.write(f"  Best Val Accuracy:          {best_val_acc:.4f} (Round {best_val_round})\n")

        if len(global_stats['test_acc']) > 0:
            best_test_acc = max(global_stats['test_acc'])
            f.write(f"  Best Test Accuracy:         {best_test_acc:.4f}\n")

        if len(global_stats['train_loss']) > 0:
            final_train_loss = global_stats['train_loss'][-1]
            f.write(f"  Final Train Loss:           {final_train_loss:.4f}\n")

        f.write("\n")

        # Per-client final statistics
        client_stats = global_stats.get('client_final_stats', [])
        if len(client_stats) > 0:
            f.write("PER-CLIENT FINAL STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Client':>7} | {'Trust':>6} | {'Val Acc':>7} | "
                    f"{'Sent':>5} | {'Recv':>5} | {'Registered':>10} | "
                    f"{'Used':>5} | {'Cache':>5}\n")
            f.write("-" * 70 + "\n")

            for cs in client_stats:
                f.write(f"{cs['client_id']:>7} | {cs['trust_score']:>6.3f} | "
                        f"{cs['validation_accuracy']:>7.3f} | "
                        f"{cs['experts_sent']:>5} | {cs['experts_received']:>5} | "
                        f"{cs['experts_registered']:>10} | "
                        f"{cs['experts_used']:>5} | {cs['cache_size']:>5}\n")

            f.write("\n")

        f.write("=" * 70 + "\n")

    return results_file


def main():
    """Main entry point"""
    args = parse_args()

    try:
        global_stats = run_federated_learning(args)

        # Save comprehensive results
        print("Saving results...")
        results_file = save_results(args, global_stats)
        print(f"Results saved to {results_file}\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()