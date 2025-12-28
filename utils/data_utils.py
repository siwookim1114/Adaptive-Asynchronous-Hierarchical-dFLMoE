import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
from pathlib import Path

class DataPartitioner:
    """
    Partition dataset across clients
    """
    def __init__(self, dataset: Dataset, num_clients: int, method: str = "label_sharding", classes_per_client: int = 2, alpha: float = 0.5, seed: int = 42):
        """
        Args:
            dataset: PyTorch Dataset
            num_clients:  Number of clients
            method: Partitioning Method
                - label_sharding: Each client gets only specific classes
                - dirichlet: Each client gets all classes with different proportions
                - iid: Equal random split (baseline)
            
            classes_per_client: Number of classes per client (for label_sharding)
            alpha: Dirichlet concentration parameter (for dirichlet)
                    Lower = more heterogeneous (0.1 = extreme, 0.5 = moderate, 10.0 = nearly IID)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.method = method
        self.classes_per_client = classes_per_client
        self.alpha = alpha
        self.seed = seed

        valid_methods = ['label_sharding', 'dirichlet', 'iid']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
        
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Getting labels
        if hasattr(dataset, "targets"):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            self.labels = np.array(dataset.labels)
        else:
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])

        self.num_classes = len(np.unique(self.labels))

        # Label Sharding: Track which classes belong to which client
        self.client_classes = None
        
        # Partition data based on method
        self.client_indices = self.partition_data()

    def partition_data(self) -> List[np.ndarray]:
        """
        Partition data according to selected method
        
        Returns:
            List of index arrays for each client
        """
        if self.method == "label_sharding":
            return self.label_sharding_partition()
        elif self.method == "dirichlet":
            return self.dirichlet_partition()
        elif self.method == "iid":
            return self.iid_partition()
        
    # Label Sharding
    def label_sharding_partition(self) -> List[np.ndarray]:
        """
        Label Sharding: Each client gets ONLY specific classes

        Example for CIFAR-10 with 5 clients, 2 classes each:
        - Client 0: Only [0, 1]
        - Client 1: Only [2, 3]
        - Client 2: Only [4, 5]
        - Client 3: Only [6, 7]
        - Client 4: Only [8, 9]

        Returns:
            List of index arrays
        """

        # Assign classes to clients
        self.client_classes = self.assign_classes()
        client_indices = [[] for _ in range(self.num_clients)]

        for client_id in range(self.num_clients):
            # Get assigned classes
            assigned_classes = self.client_classes[client_id]
            # Collecting all samples from these classes
            for class_id in assigned_classes:
                class_indices = np.where(self.labels == class_id)[0]
                client_indices[client_id].extend(class_indices)

            # Converting to array and shuffling
            client_indices[client_id] = np.array(client_indices[client_id])
            np.random.shuffle(client_indices[client_id])
    
        return client_indices
        
    def assign_classes(self) -> Dict[int, List[int]]:
        """
        Assign classes to each client for label sharding

        Strategy:
        - Shuffle classes randomly
        - Assign consecutive chunks to clients
        - Handle wraparound if needed

        Returns:
            Dictionary: {client_id: [class_0, class_1, ...]}
        """
        all_classes = list(range(self.num_classes))
        np.random.shuffle(all_classes)

        client_classes = {}

        for client_id in range(self.num_clients):
            start_idx = (client_id * self.classes_per_client) % self.num_classes
            assigned = []
            for i in range(self.classes_per_client):
                class_idx = (start_idx + i) % self.num_classes
                assigned.append(all_classes[class_idx])
            
            client_classes[client_id] = sorted(assigned)
        
        return client_classes
    
    def dirichlet_partition(self) -> List[np.ndarray]:
        """
        Dirichlet: Each client gets all classes with different proportions
        
        Example with alpha=0.5:
        - Client 0: 40% class 0, 5% class 1, 30% class 2, ...
        - Client 1: 5% class 0, 35% class 1, 8% class 2, ...
        
        Returns:
            List of index arrays
        """
        client_indices = [[] for _ in range(self.num_clients)]
        
        # For each class, distribute samples across clients
        for class_id in range(self.num_classes):
            # Get all indices of this class
            class_indices = np.where(self.labels == class_id)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Convert proportions to split points
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, proportions)
            
            # Assign to clients
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split)
        
        # Convert to arrays and shuffle
        client_indices = [np.array(indices) for indices in client_indices]
        for indices in client_indices:
            np.random.shuffle(indices)
        
        return client_indices

    def iid_partition(self) -> List[np.ndarray]:
        """
        IID: Random equal split (baseline for comparision)

        Returns:
            List of index arrays
        """

        n_samples = len(self.dataset)
        indices = np.random.permutation(n_samples)
        return np.array_split(indices, self.num_clients)
    
    def get_client_dataset(self, client_id: int) -> Subset:
        """Get dataset subset for a specific client"""
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"client_id must be 0-{self.num_clients -1}, got {client_id}")
        indices = self.client_indices[client_id]
        return Subset(self.dataset, indices)
    
    def get_client_loader(self, client_id: int, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for a specific client"""
        dataset = self.get_client_dataset(client_id)
        return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    def get_statistics(self) -> Dict[int, Dict]:
        """
        Get detailed statistics about data distribution
        
        Returns:
            Dictionary with stats per client:
            {
                client_id: {
                    'num_samples': int,
                    'assigned_classes': List[int] (only for label_sharding),
                    'class_distribution': {class_id: count}
                }
            }
        """
        stats = {}
        
        for client_id in range(self.num_clients):
            indices = self.client_indices[client_id]
            client_labels = self.labels[indices]
            
            # Count samples per class
            class_counts = {}
            for class_id in range(self.num_classes):
                count = int(np.sum(client_labels == class_id))
                if count > 0:
                    class_counts[int(class_id)] = count
            
            client_stats = {
                'num_samples': len(indices),
                'class_distribution': class_counts
            }
            
            # Add assigned classes for label sharding
            if self.method == 'label_sharding' and self.client_classes is not None:
                client_stats['assigned_classes'] = self.client_classes[client_id]
            
            stats[client_id] = client_stats
        
        return stats
    
    def print_distribution(self):
        """Print detailed distribution information"""
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        if self.method == 'label_sharding':
            print(f"LABEL SHARDING DISTRIBUTION ({self.classes_per_client} classes/client)")
        elif self.method == 'dirichlet':
            print(f"DIRICHLET DISTRIBUTION (alpha={self.alpha})")
        elif self.method == 'iid':
            print("IID DISTRIBUTION (Baseline)")
        print("=" * 70)
        print(f"Total clients: {self.num_clients}")
        print(f"Total classes: {self.num_classes}")
        print(f"Total samples: {len(self.dataset)}")
        print("=" * 70)
        
        # Per-client stats
        for client_id in range(self.num_clients):
            s = stats[client_id]
            print(f"\nClient {client_id}:")
            print(f"   Total samples: {s['num_samples']}")
            
            # For label sharding, show assigned classes
            if 'assigned_classes' in s:
                print(f"   Assigned classes: {s['assigned_classes']}")
            
            # Show class distribution
            if self.method == 'label_sharding':
                print(f"   Class counts: {s['class_distribution']}")
            else:
                # For dirichlet/iid, show top 3 classes
                sorted_classes = sorted(s['class_distribution'].items(), 
                                      key=lambda x: x[1], reverse=True)
                top_3 = sorted_classes[:3]
                print(f"   Top 3 classes: {top_3}")
                print(f"   All classes: {s['class_distribution']}")
        
        print("\n" + "=" * 70 + "\n")
    
def get_dataset(dataset_name: str, data_dir: str, train: bool = True) -> Dataset:
    """
    Get dataset by name with proper transforms

    Args:
        dataset_name: 'cifar10', 'mnist', 'cifar100'
        data_dir: Directory to store/load data
        train: Whether to get training set
    
    Returns:
        PyTorch dataset with transforms applied
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents = True, exist_ok = True)
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return datasets.CIFAR10(root = data_dir, train = train, download = True, transform = transform)

    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)

    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: cifar10, cifar100, mnist")

def create_client_dataloaders(config) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Create data loaders for all clients and test set
        
    Args:
        config: Configuration object with attributes:
            - data.dataset: Dataset name
            - data.data_dir: Data directory
            - data.partitioning_method: 'label_sharding', 'dirichlet', or 'iid'
            - data.classes_per_client: For label_sharding
            - data.non_iid_alpha: For dirichlet
            - system.num_clients: Number of clients
            - system.seed: Random seed
            - training.batch_size: Training batch size
            - evaluation.test_batch_size: Evaluation batch size
        
    Returns:
        Tuple of (train_loaders, val_loaders, test_loader):
        - train_loaders: List of training DataLoaders (one per client)
        - val_loaders: List of validation DataLoaders (one per client)
        - test_loader: Shared test DataLoader (same for all clients)
    """
    print("\n" + "=" * 70)
    print("CREATING CLIENT DATALOADERS")
    print("=" * 70)
    
    # Get datasets
    train_dataset = get_dataset(config.data.dataset, config.data.data_dir, train=True)
    test_dataset = get_dataset(config.data.dataset, config.data.data_dir, train=False)
    
    print(f"Dataset: {config.data.dataset}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Split train into train/val (90/10)
    n_train = len(train_dataset)
    n_val = int(0.1 * n_train)
    indices = np.random.permutation(n_train)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    actual_train_dataset = Subset(train_dataset, train_indices)
    val_dataset_full = Subset(train_dataset, val_indices)
    
    print(f"Split: {len(train_indices)} train, {len(val_indices)} validation")
    
    # Get partitioning parameters
    method = getattr(config.data, 'partitioning_method', 'label_sharding')
    classes_per_client = getattr(config.data, 'classes_per_client', 2)
    alpha = getattr(config.data, 'non_iid_alpha', 0.5)
    
    print(f"\nPartitioning method: {method}")
    if method == 'label_sharding':
        print(f"Classes per client: {classes_per_client}")
    elif method == 'dirichlet':
        print(f"Dirichlet alpha: {alpha}")
    
    # Partition training data
    train_partitioner = DataPartitioner(
        actual_train_dataset,
        num_clients=config.system.num_clients,
        method=method,
        classes_per_client=classes_per_client,
        alpha=alpha,
        seed=config.system.seed
    )
    
    # Partition validation data (same strategy)
    val_partitioner = DataPartitioner(
        val_dataset_full,
        num_clients=config.system.num_clients,
        method=method,
        classes_per_client=classes_per_client,
        alpha=alpha,
        seed=config.system.seed + 1
    )
    
    # Print distribution
    train_partitioner.print_distribution()
    
    # Create DataLoaders
    train_loaders = [
        train_partitioner.get_client_loader(i, config.training.batch_size, shuffle=True)
        for i in range(config.system.num_clients)
    ]
    
    val_loaders = [
        val_partitioner.get_client_loader(i, config.evaluation.test_batch_size, shuffle=False)
        for i in range(config.system.num_clients)
    ]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.test_batch_size,
        shuffle=False
    )
    
    print("✅ DataLoaders created successfully!\n")
    
    return train_loaders, val_loaders, test_loader

def verify_partitioning(partitioner: DataPartitioner):
    """
    Verify that partitioning is correct
    
    Checks:
    1. All samples assigned exactly once
    2. No duplicate assignments
    3. Total samples match dataset size
    
    Args:
        partitioner: DataPartitioner instance
    """
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    # Collect all assigned indices
    all_indices = []
    for client_id in range(partitioner.num_clients):
        all_indices.extend(partitioner.client_indices[client_id])
    
    all_indices = np.array(all_indices)
    
    # Check 1: Total count
    print(f"Dataset size: {len(partitioner.dataset)}")
    print(f"Assigned samples: {len(all_indices)}")
    
    # Check 2: No duplicates
    unique_count = len(np.unique(all_indices))
    print(f"Unique samples: {unique_count}")
    
    # Check 3: All samples covered
    assert len(all_indices) == len(partitioner.dataset), "Not all samples assigned!"
    assert unique_count == len(partitioner.dataset), "Duplicate assignments detected!"
    
    # Check 4: Valid indices
    assert all_indices.min() >= 0, "Invalid negative index!"
    assert all_indices.max() < len(partitioner.dataset), "Index out of bounds!"
    
    print(f"\n✅ Verification passed!")
    print(f"   • All {len(partitioner.dataset)} samples assigned")
    print(f"   • No duplicates")
    print(f"   • All indices valid")
    print(f"{'='*70}\n")

def test_data_utils():
    """
    Comprehensive test of data_utils module
    
    Tests:
    1. Dataset loading
    2. Label sharding partitioning
    3. Dirichlet partitioning
    4. IID partitioning
    5. DataLoader creation
    6. Correctness verification
    """
    print("COMPREHENSIVE DATA_UTILS TEST")
    
    # Test 1: Dataset Loading
    print("TEST 1: Dataset Loading")
    print("-" * 70)
    train_data = get_dataset('cifar10', './data', train=True)
    test_data = get_dataset('cifar10', './data', train=False)
    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    print(f"✓ Sample shape: {train_data[0][0].shape}")
    print(f"✓ Sample label: {train_data[0][1]}")
    print()
    
    # Test 2: Label Sharding
    print("TEST 2: Label Sharding Partitioning")
    print("-" * 70)
    ls_part = DataPartitioner(
        train_data,
        num_clients=5,
        method='label_sharding',
        classes_per_client=2,
        seed=42
    )
    ls_part.print_distribution()
    verify_partitioning(ls_part)
    
    # Test 3: Dirichlet
    print("TEST 3: Dirichlet Partitioning")
    print("-" * 70)
    dir_part = DataPartitioner(
        train_data,
        num_clients=5,
        method='dirichlet',
        alpha=0.5,
        seed=42
    )
    dir_part.print_distribution()
    verify_partitioning(dir_part)
    
    # Test 4: IID
    print("TEST 4: IID Partitioning")
    print("-" * 70)
    iid_part = DataPartitioner(
        train_data,
        num_clients=5,
        method='iid',
        seed=42
    )
    iid_part.print_distribution()
    verify_partitioning(iid_part)
    
    # Test 5: DataLoader
    print("TEST 5: DataLoader Creation")
    print("-" * 70)
    loader = ls_part.get_client_loader(0, batch_size=64)
    batch = next(iter(loader))
    print(f"✓ Batch shape: {batch[0].shape}")
    print(f"✓ Labels shape: {batch[1].shape}")
    print(f"✓ First 10 labels: {batch[1][:10].tolist()}")
    print(f"✓ Total batches: {len(loader)}")
    print()
    
    # Test 6: Statistics
    print("TEST 6: Statistics Extraction")
    print("-" * 70)
    stats = ls_part.get_statistics()
    print(f"✓ Clients with stats: {len(stats)}")
    print(f"✓ Client 0 stats: {stats[0]}")
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run tests
    test_data_utils()