"""
Peer Cache Module for Expert Storage and Management

This module implements efficient storage and retrieval of expert packages received from peer clients, with automatic staleness tracking and eviction.
"""

import torch
import time
import os
from typing import Dict, List, Optional, Tuple
import threading

class ExpertPackage:
    """
    Expert package containing head weights and metadata

    Attributes:
        client_id: Unique id for each client
        head_state_dict: Head parameters dict 
        timestamp: Timestamp when package was created
        trust_score: Current trust score (0.1 - 2.0)
        validation_accuracy: Accuracy on source validation set
        representative_features: Average feature vector for similarity
        num_samples: Number of training samples used
    """
    def __init__(self, client_id: str, head_state_dict: Dict, timestamp: float, trust_score: float, validation_accuracy: float, representative_features: torch.Tensor, num_samples: int):
        """
        Initialize expert package
        """
        self.client_id = client_id
        self.head_state_dict = head_state_dict
        self.timestamp = timestamp
        self.trust_score = trust_score
        self.validation_accuracy = validation_accuracy
        self.representative_features = representative_features
        self.num_samples = num_samples

    def get_staleness(self, lambda_decay: float = 0.001) -> float:
        """
        Compute staleness factor: e^(-lambda * delta_t)

        Args:
            lambda_decay: Decay rate parameter (per second)

        Returns:
            Staleness factor in [0, 1]
        """
        elapsed = time.time() - self.timestamp
        return float(torch.exp(torch.tensor(-lambda_decay * elapsed)))
    
    def get_age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.timestamp

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"ExpertPackage(client_id={self.client_id}, "
                f"trust={self.trust_score:.3f}, "
                f"acc={self.validation_accuracy:.3f}, "
                f"age={self.get_age_seconds():.1f}s)")
    
class PeerCache:
    """
    Thread locked cache for storing and managing expert packages

    Features:
    - Efficient storage and retrieval
    - Automatic staleness tracking
    - Eviction Policies
    - Query by trust, similarity, or staleness
    - Optional disk persistence
    - Thread-safe operations using RLock
    """
    def __init__(self, max_cache_size: int = 100, max_age_seconds: float = 3600.0, staleness_decay: float = 0.001, cache_dir: Optional[str] = None, auto_evict: bool = True):
        """
        Initialize peer cache

        Args:
            max_cache_size: Maximum number of experts to cache
            max_age_seconds: Maximum age before auto-eviction
            staleness_decay: Decay rate lambda for staleness computation
            cache_dir: Directory for persistent storage (None = memory only)
            auto_evict: Automatically evict stale experts
        """
        self.max_cache_size = max_cache_size
        self.max_age_seconds = max_age_seconds
        self.staleness_decay = staleness_decay
        self.cache_dir = cache_dir
        self.auto_evict = auto_evict

        # Storage
        self.cache: Dict[str, ExpertPackage] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Statistics
        self.stats = {
            "adds": 0,
            "updates": 0,
            "gets": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "rejections": 0
        }

        # Create cache directory
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok = True)
    
    def load_from_disk(self) -> int:
        """
        Load cached packages from disk

        Returns:
            Number of packages loaded
        """
        if self.cache_dir is None or not os.path.exists(self.cache_dir):
            return 0
        
        count = 0
        with self.lock:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".pt"):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        save_dict = torch.load(filepath)
                        package = ExpertPackage(**save_dict)
                        self.cache[package.client_id] = package
                        count += 1
                    except Exception as e:
                        print(f"[PeerCache] Failed to load {filename}: {e}")
        return count
    
    def evict_stale_unsafe(self, max_age_seconds: Optional[float] = None) -> int:
        """Internal: evict stale experts"""
        if max_age_seconds is None:
            max_age_seconds = self.max_age_seconds
        current_time = time.time()
        to_evict = []

        for id, package in self.cache.items():
            age = current_time - package.timestamp    # Need to declare current_time instead of using method to avoid inconsistency in time and compare it to the same reference point
            if age > max_age_seconds:
                to_evict.append(id)
            
        for id in to_evict:
            if id in self.cache:
                del self.cache[id]
                self.stats["evictions"] += 1

        return len(to_evict)

    def evict_one_unsafe(self) -> Optional[str]:
        """Internal: evict oldest expert"""
        if not self.cache:
            return None
        
        # Find Oldest Expert 
        oldest_id = min(self.cache.items(), key = lambda x: x[1].timestamp)[0]

        if oldest_id in self.cache:
            del self.cache[oldest_id]
            self.stats["evictions"] += 1
        return oldest_id
    
    def save_package_unsafe(self, package: ExpertPackage):
        """Internal: save package to disk"""
        if self.cache_dir is None:
            return
        
        filepath = os.path.join(self.cache_dir, f"{package.client_id}.pt")

        save_dict = {
            "client_id": package.client_id,
            "head_state_dict": package.head_state_dict,
            "timestamp": package.timestamp,
            "trust_score": package.trust_score,
            "validation_accuracy": package.validation_accuracy,
            "representative_features": package.representative_features,
            "num_samples": package.num_samples
        }
        torch.save(save_dict, filepath)

    def add(self, package: ExpertPackage) -> bool:
        """
        Add or update expert package

        Args:
            package: ExpertPackage to add
        Returns:
            True if added/updated, False if rejected
        """
        with self.lock:
            client_id = package.client_id

            # Auto evict if enabled
            if self.auto_evict:
                self.evict_stale_unsafe()
            
            # Check if already cached
            if client_id in self.cache:
                old_package = self.cache[client_id]
            
                # Only update if newer
                if package.timestamp <= old_package.timestamp:
                    self.stats["rejections"] += 1
                    return False
                
                self.cache[client_id] = package
                self.stats["updates"] += 1
            else:
                # Check size limit
                if len(self.cache) >= self.max_cache_size:
                    evicted = self.evict_one_unsafe()
                    if evicted is None:
                        self.stats["rejections"] += 1
                        return False
            
                self.cache[client_id] = package
                self.stats["adds"] += 1

            # Persist if enabled
            if self.cache_dir is not None:
                self.save_package_unsafe(package)
            return True
        
    def get(self, client_id: str) -> Optional[ExpertPackage]:
        """
        Retrieve expert package by client ID

        Args:
            client_id: Client identifier
        Returns:
            ExpertPackage if found, None otherwise
        """
        with self.lock:
            self.stats["gets"] += 1
            if client_id in self.cache:
                self.stats["hits"] += 1
                return self.cache[client_id]
            else:
                self.stats["misses"] += 1
                return None
    
    def get_all(self) -> Dict[str, ExpertPackage]:
        """Get all cached experts"""
        with self.lock:
            return self.cache
    
    def get_available_experts(self, exclude_id: Optional[str] = None) -> List[str]:
        """
        Get list of available expert IDs

        Args:
            exclude_id: Client ID to exclude (typically own ID)
        Returns:
            List of available expert client IDs
        """
        with self.lock:
            available = list(self.cache.keys())
            if available is not None:
                available = [id for id in available if id != exclude_id]
            return available
    
    def get_by_staleness(self, min_staleness: float = 0.5, exclude_id: Optional[str] = None) -> List[Tuple[str, ExpertPackage]]:
        """
        Get experts above staleness threshold

        Args:
            min_staleness: Minimum staleness factor (0 to 1)
            exclude_id: Client ID to exclude
        
        Returns:
            List of (client_id, package) sorted by staleness descending
        """
        with self.lock:
            experts = []
            for id, package in self.cache.items():
                if exclude_id is not None and id == exclude_id:
                    continue
                staleness = package.get_staleness(self.staleness_decay)
                if staleness > min_staleness:
                    experts.append((id, package, staleness))
            
            # Sort by staleness descending (freshest updated package first) -> x[2] = staleness
            experts.sort(key=lambda x: x[2], reverse = True)  
            return [(id, package) for id, package, _ in  experts]
    
    def get_by_trust(self, min_trust: float = 0.5, exclude_id: Optional[str] = None) -> List[Tuple[str, ExpertPackage]]:
        """
        Get experts above trust threshold

        Args:
            min_trust: Minimum trust score
            exclude_id: Client ID to exclude

        Returns:
            List of (client_id, package) sorted by trust descending
        """
        with self.lock:
            experts = []
            for id, package in self.cache.items():
                if exclude_id is not None and id == exclude_id:
                    continue
                if package.trust_score >= min_trust:
                    experts.append((id, package))
            
            # Sort by trust descending
            experts.sort(key = lambda x: x[1].trust_score, reverse = True)
            return experts
    
    def get_best_experts(self, k: int = 3, exclude_id: Optional[str] = None) -> List[Tuple[str, ExpertPackage]]:
        """
        Get top-K experts by combined score (trust * staleness)

        Args:
            k: Number of experts to return
            exclude_id: Client ID to exclude
        
        Returns:
            List of top-K (client_id, package) tuples
        """
        with self.lock:
            experts = []
            for id, package in self.cache.items():
                if exclude_id is not None and id == exclude_id:
                    continue
                staleness = package.get_staleness(self.staleness_decay)
                combined_score = package.trust_score * staleness
                experts.append((id, package, combined_score))
            
            # Sort by score descending
            experts.sort(key = lambda x: x[2], reverse = True)

            # Return top-K
            return [(id, package) for id, package, _ in experts[:k]]

    def update_trust(self, client_id: str, new_trust: float) -> bool:
        """
        Update trust score for cached expert

        Args:
            client_id: Client identifier
            new_trust: New trust score (will be clamped to [0.1, 2.0])

        Returns:
            True if updated, False if not found
        """
        with self.lock:
            if client_id not in self.cache:
                return False
            
            # Clamp trust score
            new_trust = max(0.1, min(2.0, new_trust))
            self.cache[client_id].trust_score = new_trust

            # Persist if enabled
            if self.cache_dir is not None:
                self.save_package_unsafe(self.cache[client_id])
            return True
    
    def remove(self, client_id: str) -> bool:
        """
        Remove expert from cache

        Args:
            client_id: Client Identifier
        
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if client_id not in self.cache:
                return False
            del self.cache[client_id]

            # Remove from disk
            if self.cache_dir is not None:
                filepath = os.path.join(self.cache_dir, f"{client_id}.pt")
                if os.path.exists(filepath):
                    os.remove(filepath)
            return True
    
    def evict_stale(self, max_age_seconds: Optional[float] = None) -> int:
        """
        Evict experts older than threshold

        Args:
            max_age_seconds: Maximum age (use default if None)
        
        Returns:
            Number of experts evicted
        """
        with self.lock:
            return self.evict_stale_unsafe(max_age_seconds)
    
    def clear(self):
        """Clear all cached experts"""
        with self.lock:
            self.cache.clear()

            # Clear disk cache
            if self.cache_dir is not None:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith(".pt"):
                        os.remove(os.path.join(self.cache_dir, filename))
    
    def size(self) -> int:
        """Get number of cached experts"""
        with self.lock:
            return len(self.cache)
    
    def is_full(self) -> bool:
        """Check if cache is at capacity"""
        with self.lock:
            return len(self.cache) >= self.max_cache_sizeL
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            if not self.cache:
                return {
                    "size": 0,
                    "max_size": self.max_cache_size,
                    "utilization": 0.0,
                    "hit_rate": 0.0,
                    "avg_age_seconds": 0.0,
                    "avg_trust": 0.0,
                    "avg_staleness": 0.0,
                    **self.stats
                }
            ages = [package.get_age_seconds() for package in self.cache.values()]
            trusts = [package.trust_score for package in self.cache.values()]
            stalenesses = [package.get_staleness(self.staleness.decay) for package in self.cache.values()]
            hit_rate = (self.stats["hits"] / self.stats["gets"] if self.stats["gets"] > 0 else 0.0)

            return {
                "size": len(self.cache),
                "max_size": self.max_cache_size,
                "utilization": len(self.cache) / self.max_cache_size,
                "hit_rate": hit_rate,
                "avg_age_seconds": sum(ages) / len(ages),
                "avg_trust": sum(trusts) / len(trusts),
                "avg_staleness": sum(stalenesses) / len(stalenesses),
                **self.stats
            }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_satistics()
        print(f"\n{'='*60}")
        print("PEER CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Size: {stats['size']}/{stats['max_size']} "
              f"({stats['utilization']:.1%})")
        print(f"Adds: {stats['adds']}, Updates: {stats['updates']}")
        print(f"Gets: {stats['gets']} (Hit Rate: {stats['hit_rate']:.1%})")
        print(f"Evictions: {stats['evictions']}, Rejections: {stats['rejections']}")
        print(f"Avg Age: {stats['avg_age_seconds']:.1f}s")
        print(f"Avg Trust: {stats['avg_trust']:.3f}")
        print(f"Avg Staleness: {stats['avg_staleness']:.3f}")
        print(f"{'='*60}\n")

    