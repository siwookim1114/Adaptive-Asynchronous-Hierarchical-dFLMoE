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
from pathlib import Path
from typing import List, Dict
import numpy as np

# Import all components
from infra.cluster import ClusterManager
from client_node import ClientNode
from models.body_encoder import BodyEncoder
from models.head import Head


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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_head', type=float, default=0.001, help='Learning rate for head')
    parser.add_argument('--lr_body', type=float, default=0.001, help='Learning rate for body')
    parser.add_argument('--lr_router', type=float, default=0.001, help='Learning rate for router')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
    parser.add_argument('--data_heterogeneity', type=str, default='iid', choices=['iid', 'non-iid'], help='Data distribution')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID')
    
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
    Load dataset
    
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


def partition_data_iid(dataset: Dataset, num_clients: int):
    """
    Partition data IID across clients
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        
    Returns:
        List of indices for each client
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    samples_per_client = num_samples // num_clients
    client_indices = []
    
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start:end].tolist())
    
    return client_indices


def partition_data_non_iid(dataset: Dataset, num_clients: int, num_classes: int, alpha: float = 0.5):
    """
    Partition data non-IID using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        num_classes: Number of classes
        alpha: Dirichlet concentration parameter (smaller = more heterogeneous)
        
    Returns:
        List of indices for each client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset doesn't have targets or labels")
    
    # Organize indices by class
    class_indices = {k: np.where(labels == k)[0] for k in range(num_classes)}
    
    # Sample proportions for each client from Dirichlet
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        # Sample proportions
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split class k indices according to proportions
        indices = class_indices[k]
        np.random.shuffle(indices)
        
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, splits)
        
        # Assign to clients
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices


def create_data_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    client_indices: List[List[int]],
    batch_size: int,
    val_split: float = 0.1
):
    """
    Create data loaders for each client
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        client_indices: Indices for each client
        batch_size: Batch size
        val_split: Validation split ratio
        
    Returns:
        (train_loaders, val_loaders, test_loader)
    """
    train_loaders = []
    val_loaders = []
    
    for indices in client_indices:
        # Split into train/val
        n_val = int(len(indices) * val_split)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Create subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    
    # Global test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loaders, val_loaders, test_loader


def create_models(feature_dim: int, num_classes: int, device: torch.device):
    """
    Create body encoder and head
    
    Args:
        feature_dim: Feature dimension
        num_classes: Number of classes
        device: Device
        
    Returns:
        (body_encoder, head)
    """
    body_encoder = BodyEncoder(output_dim=feature_dim).to(device)
    head = Head(input_dim=feature_dim, output_dim=num_classes).to(device)
    
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
    print(f"Top-K experts: {args.top_k_experts}")
    print(f"Data distribution: {args.data_heterogeneity}\n")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}\n")
    
    # Partition data
    print("Partitioning data...")
    if args.data_heterogeneity == 'iid':
        client_indices = partition_data_iid(train_dataset, args.num_clients)
    else:
        client_indices = partition_data_non_iid(
            train_dataset, args.num_clients, args.num_classes, args.non_iid_alpha
        )
    
    # Print data distribution
    for i, indices in enumerate(client_indices):
        print(f"  Client {i}: {len(indices)} samples")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loaders, val_loaders, test_loader = create_data_loaders(
        train_dataset, test_dataset, client_indices, args.batch_size
    )
    print(f"  Created {len(train_loaders)} train loaders")
    print(f"  Created {len(val_loaders)} val loaders\n")
    
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
            learning_rate_head=args.lr_head,
            learning_rate_body=args.lr_body,
            learning_rate_router=args.lr_router
        )
        
        # Create models for this client
        body_encoder, head = create_models(args.feature_dim, args.num_classes, device)
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
        
        # Train each client
        round_metrics = []
        
        for i, client in enumerate(clients):
            try:
                metrics = client.train_round(
                    train_loader=train_loaders[i],
                    val_loader=val_loaders[i],
                    criterion=criterion,
                    device=device,
                    use_experts=True
                )
                
                round_metrics.append(metrics)
                
                if args.verbose or i == 0:
                    print(f"Client {i}: "
                          f"Train Loss={metrics['train_loss']:.4f}, "
                          f"Train Acc={metrics['train_acc']:.3f}, "
                          f"Val Acc={metrics['val_acc']:.3f}, "
                          f"Trust={metrics['trust_score']:.3f}, "
                          f"Cache={metrics['cache_size']}, "
                          f"Experts Used={metrics['experts_used']}")
            
            except Exception as e:
                print(f"  Client {i} error: {e}")
                continue
        
        # Aggregate statistics
        if len(round_metrics) > 0:
            avg_train_loss = np.mean([m['train_loss'] for m in round_metrics])
            avg_train_acc = np.mean([m['train_acc'] for m in round_metrics])
            avg_val_loss = np.mean([m['val_loss'] for m in round_metrics])
            avg_val_acc = np.mean([m['val_acc'] for m in round_metrics])
            avg_experts_used = np.mean([m['experts_used'] for m in round_metrics])
            
            # Global test every 10 rounds
            if round_num % 10 == 0:
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
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            print(f"Average Train Acc:  {avg_train_acc:.3f}")
            print(f"Average Val Loss:   {avg_val_loss:.4f}")
            print(f"Average Val Acc:    {avg_val_acc:.3f}")
            print(f"Avg Experts Used:   {avg_experts_used:.1f}")
            if test_acc is not None:
                print(f"Global Test Acc:    {test_acc:.3f}")
            print(f"Round Time:         {round_time:.2f}s")
            print(f"{'='*70}\n")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70 + "\n")
    
    final_test_acc = evaluate_global(clients, test_loader, device)
    print(f"Final Global Test Accuracy: {final_test_acc:.3f}\n")
    
    # Print final statistics
    print("Final Statistics:")
    for i, client in enumerate(clients):
        stats = client.get_statistics()
        print(f"\nClient {i}:")
        print(f"  Rounds: {stats['rounds_completed']}")
        print(f"  Experts Sent: {stats['experts_sent']}")
        print(f"  Experts Received: {stats['experts_received']}")
        print(f"  Experts Registered: {stats['experts_registered']}")
        print(f"  Experts Used: {stats['experts_used']}")
        print(f"  Trust Score: {client.trust_score:.3f}")
        print(f"  Cache Size: {stats['cache']['size']}")
    
    # Shutdown
    print("\n" + "="*70)
    print("SHUTTING DOWN")
    print("="*70 + "\n")
    
    for client in clients:
        client.shutdown()
    
    print("Training complete!\n")
    
    return global_stats


def evaluate_global(clients: List[ClientNode], test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate global test accuracy (average of all clients)
    
    Args:
        clients: List of clients
        test_loader: Test data loader
        device: Device
        
    Returns:
        Average test accuracy
    """
    accuracies = []
    
    for client in clients:
        if client.body_encoder is None or client.head is None:
            continue
        
        client.body_encoder.eval()
        client.head.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                features = client.body_encoder(x)
                logits = client.head(features)
                
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        accuracy = correct / total
        accuracies.append(accuracy)
    
    return np.mean(accuracies) if len(accuracies) > 0 else 0.0


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        global_stats = run_federated_learning(args)
        
        # Save results
        print("Saving results...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = results_dir / f"results_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("Federated Learning Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Clients: {args.num_clients}\n")
            f.write(f"  Clusters: {args.num_clusters}\n")
            f.write(f"  Rounds: {args.rounds}\n")
            f.write(f"  Alpha: {args.alpha}\n")
            f.write(f"  Top-K: {args.top_k_experts}\n")
            f.write(f"  Data: {args.data_heterogeneity}\n\n")
            
            f.write("Results:\n")
            for i, round_num in enumerate(global_stats['round']):
                f.write(f"Round {round_num}: "
                       f"Train Acc={global_stats['train_acc'][i]:.3f}, "
                       f"Val Acc={global_stats['val_acc'][i]:.3f}\n")
            
            if len(global_stats['test_acc']) > 0:
                f.write(f"\nFinal Test Acc: {global_stats['test_acc'][-1]:.3f}\n")
        
        print(f"Results saved to {results_file}\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()