"""
Complete orchestration of federated learning system with:
- TRUE asynchronous training (no round barrier — each client trains independently)
- Hierarchical clustering with time-based reclustering
- Expert routing with FST and trust-weighted scoring
- Dynamic adaptive cache with staleness eviction
- Hybrid loss training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

import time
import threading
import queue
import argparse
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
    parser.add_argument('--rounds', type=int, default=100, help='Number of training rounds per client')

    # Model settings
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--top_k_experts', type=int, default=3, help='Top-K experts to select')

    # Training settings
    parser.add_argument('--alpha', type=float, default=0.5, help='Hybrid loss weight (local)')
    parser.add_argument('--warmup_rounds', type=int, default=10, help='Rounds for alpha warm-up (1.0 to target alpha)')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_head', type=float, default=0.001, help='Learning rate for head')
    parser.add_argument('--lr_body', type=float, default=0.001, help='Learning rate for body')
    parser.add_argument('--lr_router', type=float, default=0.001, help='Learning rate for router')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='LR decay factor per round')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout in expert heads')

    # Async and adaptive settings
    parser.add_argument('--staleness_lambda', type=float, default=0.005,
                        help='Staleness decay rate for e^(-lambda*t) scoring')
    parser.add_argument('--max_expert_age', type=float, default=300.0,
                        help='Max expert age in seconds before cache eviction')
    parser.add_argument('--cross_cluster_interval', type=float, default=60.0,
                        help='Seconds between cross-cluster expert exchanges')
    parser.add_argument('--eval_interval', type=float, default=300.0,
                        help='Seconds between global evaluations')
    parser.add_argument('--recluster_interval', type=float, default=150.0,
                        help='Seconds between reclustering')

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
    Main federated learning orchestration with TRUE asynchronous training.

    Each client runs its own independent training loop in a separate thread.
    There is NO global round barrier — faster clients proceed without waiting
    for slower ones. Expert packages arrive asynchronously via TCP transport.

    Periodic operations (evaluation, reclustering) are time-based, not
    round-based, fitting the asynchronous model.

    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("ASYNC FEDERATED LEARNING WITH ADAPTIVE HIERARCHICAL MoE")
    print("="*70 + "\n")

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of clusters: {args.num_clusters}")
    print(f"Training rounds (per client): {args.rounds}")
    print(f"Alpha (local weight): {args.alpha}")
    print(f"Warmup rounds: {args.warmup_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Top-K experts: {args.top_k_experts}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"LR decay: {args.lr_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Staleness lambda: {args.staleness_lambda}")
    print(f"Max expert age: {args.max_expert_age}s")
    print(f"Cross-cluster interval: {args.cross_cluster_interval}s")
    print(f"Eval interval: {args.eval_interval}s")
    print(f"Recluster interval: {args.recluster_interval}s")
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
            staleness_lambda=args.staleness_lambda,
            top_k_experts=args.top_k_experts,
            alpha=args.alpha,
            warmup_rounds=args.warmup_rounds,
            local_epochs=args.local_epochs,
            learning_rate_head=args.lr_head,
            learning_rate_body=args.lr_body,
            learning_rate_router=args.lr_router,
            weight_decay=args.weight_decay,
            lr_decay=args.lr_decay,
            max_expert_age=args.max_expert_age,
            cross_cluster_interval=args.cross_cluster_interval
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

    # ================================================================
    # TRUE ASYNCHRONOUS TRAINING
    # Each client runs its own independent training loop in a thread.
    # No global round barrier — faster clients proceed without waiting.
    # ================================================================
    print("\n" + "="*70)
    print("STARTING ASYNCHRONOUS TRAINING")
    print("="*70 + "\n")

    # Shared communication
    stats_queue = queue.Queue()       # Clients report metrics here
    eval_pause = threading.Event()    # Cleared during evaluation to pause clients
    eval_pause.set()                  # Not paused initially
    all_training_done = threading.Event()  # Set when all clients finish — signals keep-alive loops to exit

    # Wire eval_pause to all clients so they check it at epoch boundaries
    for client in clients:
        client.eval_pause = eval_pause

    # Statistics tracking
    global_stats = {
        'timestamps': [],         # Wall-clock time of each evaluation
        'eval_round_counts': [],  # Per-client round counts at each eval
        'test_acc': [],
        'avg_train_loss': [],
        'avg_train_acc': [],
        'avg_val_loss': [],
        'avg_val_acc': [],
        'avg_cache_size': [],
        'avg_experts_used': [],
    }

    # Per-client latest metrics (updated from queue)
    client_latest_metrics = [None] * args.num_clients
    client_round_counts = [0] * args.num_clients

    training_start_time = time.time()

    def client_training_loop(idx, client, train_loader, val_loader):
        """Independent training loop for a single client.

        This function runs in its own thread. The client trains for
        args.rounds local rounds at its own pace, with no synchronization
        barrier with other clients. Expert packages arrive asynchronously
        via the transport layer during training.

        After training completes, the client enters a keep-alive phase:
        it periodically re-shares its final (best) expert package so that
        peers' caches don't evict it due to staleness. This continues
        until all clients have finished training. Without keep-alive,
        fast-finishing clients' experts get evicted, degrading the expert
        pool for still-training clients.
        """
        for local_round in range(1, args.rounds + 1):
            # eval_pause is checked at each epoch boundary inside
            # _train_epoch_corrected(), so clients pause quickly

            try:
                metrics = client.train_round(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    use_experts=True
                )
                stats_queue.put(('metrics', idx, local_round, metrics))
            except Exception as e:
                stats_queue.put(('error', idx, local_round, str(e)))

        stats_queue.put(('done', idx, 0, None))

        # Keep-alive: re-share final expert every 30s to prevent intra-cluster eviction.
        # The expert timestamp is refreshed on each intra-cluster share, so cluster
        # peers' caches keep it alive until all training completes.
        #
        # allow_cross_cluster=False is intentional: after a client finishes its
        # training rounds, its expert is frozen (no more gradient updates). Continuing
        # to force cross-cluster exchange every 30s would refresh timestamps on
        # post-training experts, defeating max_age eviction and locking the cache at
        # 9/9. By stopping cross-cluster here, post-training cross-cluster experts
        # age out naturally — restoring staleness discrimination and allowing the
        # router to prefer fresher, still-training intra-cluster experts.
        # Intra-cluster sharing is unaffected (no knowledge loss within cluster).
        reshare_interval = 30  # seconds
        while not all_training_done.wait(timeout=reshare_interval):
            try:
                eval_pause.wait()  # Pause during evaluation
                client._drain_pending_registrations()
                client.share_expert(force_all_targets=True, allow_cross_cluster=False)
            except Exception as e:
                print(f"  [Client {idx}] Keep-alive share error: {e}")

    # Launch all client threads independently
    threads = []
    for i, client in enumerate(clients):
        t = threading.Thread(
            target=client_training_loop,
            args=(i, client, train_loaders[i], val_loaders[i]),
            daemon=True,
            name=f"Client-{i}"
        )
        threads.append(t)
        t.start()

    # ================================================================
    # MONITORING LOOP
    # Collects metrics from clients, triggers periodic evaluation
    # and reclustering based on wall-clock time (not round counts).
    # ================================================================
    clients_done = 0
    last_eval_time = training_start_time
    last_recluster_time = training_start_time
    eval_count = 0

    # First recluster after initial features are computed (~30s)
    first_recluster_done = False

    print(f"[Async] All {args.num_clients} client threads launched. Monitoring...\n")

    while clients_done < args.num_clients:
        # Drain metrics from queue (non-blocking)
        try:
            msg_type, idx, rnd, data = stats_queue.get(timeout=1.0)

            if msg_type == 'metrics':
                client_round_counts[idx] = rnd
                client_latest_metrics[idx] = data
                if args.verbose or idx == 0:
                    print(f"  [Client {idx}] Round {rnd}: "
                          f"Train Loss={data['train_loss']:.4f}, "
                          f"Train Acc={data['train_acc']:.3f}, "
                          f"Val Acc={data['val_acc']:.3f}, "
                          f"Cache={data['cache_size']}, "
                          f"Experts={data['experts_used']}")
            elif msg_type == 'error':
                print(f"  [Client {idx}] Round {rnd} ERROR: {data}")
            elif msg_type == 'done':
                clients_done += 1
                print(f"  [Client {idx}] FINISHED all {args.rounds} rounds "
                      f"({clients_done}/{args.num_clients} done)")

        except queue.Empty:
            pass

        now = time.time()
        elapsed = now - training_start_time

        # First recluster: after ~30s when clients have real features
        if not first_recluster_done and elapsed > 30:
            print(f"\n[Async] First recluster at t={elapsed:.0f}s")
            cluster_manager.perform_clustering()
            first_recluster_done = True
            last_recluster_time = now

        # Periodic reclustering (time-based)
        if now - last_recluster_time >= args.recluster_interval:
            print(f"\n[Async] Reclustering at t={elapsed:.0f}s "
                  f"(rounds: {client_round_counts})")
            cluster_manager.perform_clustering()
            last_recluster_time = now

        # Periodic evaluation (time-based)
        if now - last_eval_time >= args.eval_interval:
            eval_count += 1
            print(f"\n{'='*70}")
            print(f"[EVAL {eval_count}] t={elapsed:.0f}s | "
                  f"Client rounds: {client_round_counts}")
            print(f"{'='*70}")

            try:
                # Pause all clients at their next epoch boundary
                # (clients check eval_pause.wait() at start of each epoch)
                # INSIDE try so that Ctrl+C during sleep still triggers finally→set()
                eval_pause.clear()
                time.sleep(3)  # Wait for in-flight batches to finish (~1-2s each)
                # Set eval mode on all clients
                for client in clients:
                    if client.body_encoder is not None:
                        client.body_encoder.eval()
                        client.head.eval()
                        client.router.eval()

                # Evaluate
                test_acc = evaluate_global(clients, test_loader, device)

                # Collect current aggregate metrics
                active_metrics = [m for m in client_latest_metrics if m is not None]
                if len(active_metrics) > 0:
                    avg_train_loss = np.mean([m['train_loss'] for m in active_metrics])
                    avg_train_acc = np.mean([m['train_acc'] for m in active_metrics])
                    avg_val_loss = np.mean([m['val_loss'] for m in active_metrics])
                    avg_val_acc = np.mean([m['val_acc'] for m in active_metrics])
                    avg_cache = np.mean([m['cache_size'] for m in active_metrics])
                    avg_experts = np.mean([m['experts_used'] for m in active_metrics])
                else:
                    avg_train_loss = avg_train_acc = avg_val_loss = avg_val_acc = 0.0
                    avg_cache = avg_experts = 0.0

                # Store
                global_stats['timestamps'].append(elapsed)
                global_stats['eval_round_counts'].append(list(client_round_counts))
                global_stats['test_acc'].append(test_acc)
                global_stats['avg_train_loss'].append(avg_train_loss)
                global_stats['avg_train_acc'].append(avg_train_acc)
                global_stats['avg_val_loss'].append(avg_val_loss)
                global_stats['avg_val_acc'].append(avg_val_acc)
                global_stats['avg_cache_size'].append(avg_cache)
                global_stats['avg_experts_used'].append(avg_experts)

                print(f"\n{'~'*70}")
                print(f"  RESULTS:")
                print(f"  Global Test Acc:  {test_acc:.4f}")
                print(f"  Avg Train Loss:   {avg_train_loss:.4f}")
                print(f"  Avg Train Acc:    {avg_train_acc:.3f}")
                print(f"  Avg Val Loss:     {avg_val_loss:.4f}")
                print(f"  Avg Val Acc:      {avg_val_acc:.3f}")
                print(f"  Avg Cache Size:   {avg_cache:.1f}")
                print(f"  Avg Experts Used: {avg_experts:.0f}")
                print(f"{'~'*70}")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"[EVAL {eval_count}] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # ALWAYS restore train mode and resume clients
                for client in clients:
                    if client.body_encoder is not None:
                        client.body_encoder.train()
                        client.head.train()
                        client.router.train()

                eval_pause.set()  # Resume all client threads
                last_eval_time = now

    # Signal keep-alive loops to exit now that all clients finished training
    all_training_done.set()

    # Wait for all threads to finish (keep-alive loops exit within 30s)
    for t in threads:
        t.join(timeout=60)

    # ================================================================
    # FINAL EVALUATION
    # ================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70 + "\n")

    # Set eval mode
    for client in clients:
        if client.body_encoder is not None:
            client.body_encoder.eval()
            client.head.eval()
            client.router.eval()

    final_test_acc = evaluate_global(clients, test_loader, device)
    total_training_time = time.time() - training_start_time

    global_stats['final_test_acc'] = final_test_acc
    global_stats['total_training_time'] = total_training_time

    print(f"Final Global Test Accuracy: {final_test_acc:.4f}")
    if len(global_stats['test_acc']) > 0:
        print(f"Best Test Accuracy:         {max(global_stats['test_acc']):.4f}")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.1f}min)")
    print(f"Client round counts: {client_round_counts}\n")

    # Collect per-client final statistics
    client_final_stats = []
    print("Final Per-Client Statistics:")
    for i, client in enumerate(clients):
        stats = client.get_statistics()
        client_info = {
            'client_id': i,
            'rounds_completed': stats['rounds_completed'],
            'experts_sent': stats['experts_sent'],
            'experts_received': stats['experts_received'],
            'experts_relayed': stats['experts_relayed'],
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
        print(f"  Experts Relayed: {stats['experts_relayed']}")
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
    Evaluate global test accuracy using confidence-weighted MoE ensemble.

    Each client produces its prediction purely via its router's MoE:
    1. Router selects top-K experts from all available (including local head)
    2. Applies per-expert FST and weighted combination
    3. This MoE output IS the prediction

    These MoE predictions are converted to softmax probabilities
    and trust-weighted ensembled across all clients. Per-sample confidence
    (max class probability) naturally down-weights uncertain clients.

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

    # Prepare all clients: build expert heads from cache
    client_expert_heads = []
    for client in clients:
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

        # Include local head in expert pool
        expert_heads[client.local_expert_id] = client.head

        client_expert_heads.append(expert_heads)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # Per-sample confidence-weighted ensemble:
            # Clients uncertain about a sample (poor body encoder features for
            # that class) produce near-uniform softmax -> low confidence -> low
            # weight. Clients confident about a sample dominate the ensemble.
            ensemble_probs = torch.zeros(x.size(0), num_classes, device=device)
            total_weight = torch.zeros(x.size(0), 1, device=device)

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
                # Per-sample confidence: max class probability (B, 1)
                confidence = probs.max(dim=1, keepdim=True)[0]
                # Trust x confidence -> uncertain clients contribute less
                weight = client.trust_score * confidence  # (B, 1)
                ensemble_probs += weight * probs
                total_weight += weight

            ensemble_probs = ensemble_probs / total_weight.clamp(min=1e-8)

            _, predicted = ensemble_probs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return correct / total if total > 0 else 0.0


def save_results(args, global_stats):
    """
    Save comprehensive results to a text file in results/ directory

    Includes:
    - Full configuration (including async/adaptive params)
    - Per-evaluation metrics (time-based, not round-based)
    - Per-client final statistics
    - Final test accuracy
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = results_dir / f"results_{args.dataset}_{args.partition_method}_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ASYNC FEDERATED LEARNING WITH ADAPTIVE HIERARCHICAL MoE - RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Dataset:            {args.dataset}\n")
        f.write(f"  Num Clients:        {args.num_clients}\n")
        f.write(f"  Num Clusters:       {args.num_clusters}\n")
        f.write(f"  Rounds (per client):{args.rounds}\n")
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
        f.write(f"  Staleness Lambda:   {args.staleness_lambda}\n")
        f.write(f"  Max Expert Age:     {args.max_expert_age}s\n")
        f.write(f"  Cross-Cluster Int:  {args.cross_cluster_interval}s\n")
        f.write(f"  Eval Interval:      {args.eval_interval}s\n")
        f.write(f"  Recluster Interval: {args.recluster_interval}s\n")
        f.write(f"  Feature Dim:        {args.feature_dim}\n")
        f.write(f"  Seed:               {args.seed}\n")
        f.write(f"  Device:             {args.device}\n\n")

        # Per-evaluation metrics table
        f.write("PER-EVALUATION METRICS (time-based)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Eval':>5} | {'Time(s)':>8} | {'Test Acc':>8} | {'Train Loss':>10} | "
                f"{'Train Acc':>9} | {'Val Acc':>7} | {'Cache':>5} | {'Experts':>7} | {'Rounds':>30}\n")
        f.write("-" * 120 + "\n")

        for i in range(len(global_stats.get('timestamps', []))):
            ts = global_stats['timestamps'][i]
            test_acc = global_stats['test_acc'][i]
            train_loss = global_stats['avg_train_loss'][i]
            train_acc = global_stats['avg_train_acc'][i]
            val_acc = global_stats['avg_val_acc'][i]
            cache = global_stats['avg_cache_size'][i]
            experts = global_stats['avg_experts_used'][i]
            rounds = global_stats['eval_round_counts'][i]

            # Summarize round counts
            min_r = min(rounds)
            max_r = max(rounds)
            avg_r = np.mean(rounds)
            rounds_str = f"min={min_r} avg={avg_r:.0f} max={max_r}"

            f.write(f"{i+1:>5} | {ts:>8.1f} | {test_acc:>8.4f} | {train_loss:>10.4f} | "
                    f"{train_acc:>9.4f} | {val_acc:>7.4f} | {cache:>5.1f} | {experts:>7.0f} | {rounds_str:>30}\n")

        f.write("\n")

        # Final test accuracy
        f.write("FINAL RESULTS\n")
        f.write("-" * 70 + "\n")
        final_test = global_stats.get('final_test_acc', 0.0)
        f.write(f"  Final Global Test Accuracy: {final_test:.4f}\n")

        if len(global_stats.get('test_acc', [])) > 0:
            best_test_acc = max(global_stats['test_acc'])
            f.write(f"  Best Test Accuracy:         {best_test_acc:.4f}\n")

        if len(global_stats.get('avg_val_acc', [])) > 0:
            best_val_acc = max(global_stats['avg_val_acc'])
            f.write(f"  Best Avg Val Accuracy:      {best_val_acc:.4f}\n")

        if len(global_stats.get('avg_train_loss', [])) > 0:
            final_train_loss = global_stats['avg_train_loss'][-1]
            f.write(f"  Final Avg Train Loss:       {final_train_loss:.4f}\n")

        # Timing
        total_time = global_stats.get('total_training_time', 0.0)
        f.write(f"  Total Training Time:        {total_time:.2f}s ({total_time/60:.1f}min)\n")

        f.write("\n")

        # Per-client final statistics
        client_stats = global_stats.get('client_final_stats', [])
        if len(client_stats) > 0:
            f.write("PER-CLIENT FINAL STATISTICS\n")
            f.write("-" * 85 + "\n")
            f.write(f"{'Client':>7} | {'Trust':>6} | {'Val Acc':>7} | "
                    f"{'Rounds':>6} | {'Sent':>5} | {'Recv':>5} | {'Relay':>5} | {'Registered':>10} | "
                    f"{'Used':>5} | {'Cache':>5}\n")
            f.write("-" * 85 + "\n")

            for cs in client_stats:
                f.write(f"{cs['client_id']:>7} | {cs['trust_score']:>6.3f} | "
                        f"{cs['validation_accuracy']:>7.3f} | "
                        f"{cs['rounds_completed']:>6} | "
                        f"{cs['experts_sent']:>5} | {cs['experts_received']:>5} | "
                        f"{cs['experts_relayed']:>5} | "
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
