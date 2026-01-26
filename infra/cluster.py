"""
Cluster Manager for Hierarchical Organization

This module organizes federated clients into hierarchical clusters to reduce communication overhead from O(N^2) -> O(NlogN)

Features:
- Similarity-based clustering (using representative features)
- Hierarchical structure (clients -> clusters -> cluster heads)
- Adaptive re-clustering (periodic reorganization)
- Cluster head selection (highest trust score)
- Communication routing (within cluster vs across clusters)
"""
import numpy as np
import torch
import threading
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
import math

class ClientInfo:
    """
    Information about a client in the cluster system

    Attributes:
        client_id: Unique identifier for the client
        features: Representative features (embedding) for similarity
        trust_score: Trust score of this client
        cluster_id: Which cluster thisc lient belongs to
        last_updated: Timestamp of last update
        is_cluster_head: Whether this client is a cluster head
    """
    def __init__(self, client_id: str, features: torch.Tensor, trust_score: float, cluster_id: Optional[int] = None):
        """Initialize client info"""
        self.client_id = client_id
        self.features = features   # Representative features
        self.trust_score = trust_score
        self.cluster_id = cluster_id
        self.last_updated = time.time()
        self.is_cluster_head = False

    def update(self, features: torch.Tensor, trust_score: float):
        """Update client information"""
        self.features = features
        self.trust_score = trust_score
        self.last_updated = time.time()

    def __repr__(self) -> str:
        """String representation"""
        head_str = " (HEAD)" if self.is_cluster_head else ""
        return (f"ClientInfo(id = {self.client_id}, cluster={self.cluster_id}, "
                f"trust={self.trust_score:.3f}{head_str})")
    
class Cluster:
    """
    A cluster of similar clients

    Attributes:
        cluster_id: Unique identifier for this cluster
        member_ids: Set of client IDs in this cluster
        cluster_head_id: ID of the cluster head (highest trust)
        centroid: Average feature vector of all members
        last_updated: Timestamp of last update
    """
    def __init__(self, cluster_id: int):
        """Initialize cluster"""
        self.cluster_id = cluster_id
        self.member_ids: Set[str] = set()
        self.cluster_head_id: Optional[str] = None
        self.centroid: Optional[torch.Tensor] = None
        self.last_updated = time.time()

    def add_member(self, client_id: str):
        """Add client to cluster"""
        self.member_ids.add(client_id)
        self.last_updated = time.time()

    def remove_member(self, client_id: str):
        """Remove client from cluster"""
        if self.cluster_head_id == client_id:
            self.cluster_head_id = None
        self.last_updated = time.time()
    
    def size(self) -> int:
        """Get number of members"""
        return len(self.member_ids)
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"Cluster(id={self.cluster_id}, size={self.size()}, "
                f"head={self.cluster_head_id})")
    
class ClusterManager:
    """
    Manages hierarchical clustering of federated clients
    - Within cluster: frequent exchange (similar clients)
    - Across clusters: less frequent exchange via cluster heads

    Architecture:
    Clients -> Clusters -> Cluster Heads -> Cross-Cluster exchange
    """
    def __init__(
        self, 
        num_clusters: int = 5, 
        min_cluster_size: int = 2, 
        max_cluster_size: int = 20, 
        recluster_interval: float = 100.0, 
        similarity_threshold: float = 0.5
    ):
        """
        Initialize cluster manager

        Args:
            num_clusters: Target number of clusters
            min_cluster_size: Minimum clients per cluster
            max_cluster_size: Maximum clients per cluster
            recuster_interval: How often to re-cluster(seconds)
            similarity_threshold: Minimum similarity for same cluster
        """
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.recluster_interval = recluster_interval
        self.similarity_threshold = similarity_threshold

        # Client information: {client_id: ClientInfo}
        self.clients: Dict[str, ClientInfo] = {}
        self.clients_lock = threading.RLock()

        # Clusters: {cluster_id: Cluster}
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_lock = threading.RLock()

        # Clustering state
        self.last_clustering_time = 0.0
        self.clustering_needed = False

        # Statistics
        self.stats = {
            "total_clients": 0,
            "total_clusters": 0,
            "avg_cluster_size": 0.0,
            "num_reclustering": 0
        }
        self.stats_lock = threading.RLock()

        print(f"[ClusterManager] Initialized (num_clusters={num_clusters}, "
              f"recluster_interval={recluster_interval}s)")
    
    def register_client(self, client_id: str, features: torch.Tensor, trust_score: float = 0.5):
        """
        Register a new client

        Args:
            client_id: Client's unique identifier
            features: Representative features (embedding)
            trust_score: Initial trust score
        """
        with self.clients_lock:
            # Create client info
            client_info = ClientInfo(client_id, features, trust_score)    # Client information
            self.clients[client_id] = client_info

            # Mark clustering needed
            self.clustering_needed = True

            # Update stats
            with self.stats_lock:
                self.stats["total_clients"] = len(self.clients)
            print(f"[ClusterManager] Registered client {client_id} "
                  f"(total: {len(self.clients)})")
    
    def update_client(self, client_id: str, features: Optional[torch.Tensor] = None, trust_score: Optional[float] = None):
        """
        Update client information

        Args:
            client_id: Client to update
            features: New representative features (optional)
            trust_score: New trust score (optional)
        """
        with self.clients_lock:
            if client_id not in self.clients:
                print(f"[ClusterManager] Unknown client: {client_id}")
                return
            client_info = self.clients[client_id]

            # Update features if provided
            if features is not None:
                client_info.features = features    # Update with new features
                self.clustering_needed = True      # Features changed, need re-cluster
            
            # Update trust score if provided
            if trust_score is not None:
                old_trust = client_info.trust_score      # Store original old trust score
                client_info.trust_score = trust_score    # Update with new trust_score

                # If this client is cluster head and trust decreased significantly
                # Consider selecting new head
                if client_info.is_cluster_head and trust_score < old_trust * 0.8:   # If trust decreased more than 80%
                    cluster_id = client_info.cluster_id
                    if cluster_id is not None:
                        self.select_cluster_head(cluster_id)    # Select the cluster head again
            
            client_info.last_updated = time.time()

    def remove_client(self, client_id: str):
        """
        Remove a client from the system

        Args:
            client_id: Client to remove
        """
        with self.clients_lock, self.cluster_lock:
            if client_id not in self.clients:
                return 
            
            client_info = self.clients[client_id]
            cluster_id = client_info.cluster_id

            # Remove from cluster
            if cluster_id is not None and cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                cluster.remove_member(client_id)

                # If cluster too small, mark for re-clustering
                if cluster.size() < self.min_cluster_size:
                    self.clustering_needed = True
                
                # If was cluster head, select new head
                if client_info.is_cluster_head:
                    self.select_cluster_head(cluster_id)
            
            # Remove client
            del self.clients[client_id]

            # Update stats
            with self.stats_lock:
                self.stats["total_clients"] = len(self.clients)
            
            print(f"[ClusterManager] Removed client {client_id}")

    def perform_clustering(self):
        """
        Perform clustering of all clients based on feature similarity

        Uses K-Means clustering on representative features
        """
        with self.clients_lock, self.cluster_lock:
            if len(self.clients) < 2:
                print("[ClusterManager] Not enough clients for clustering")
                return
            print(f"[ClusterManager] Performing clustering ({len(self.clients)} clients)...")

            # Gather all client features
            client_ids = list(self.clients.keys())    # List of all client_ids that are registered
            features_list = []

            for client_id in client_ids:
                client_info = self.clients[client_id]
                # Convert to numpy if tensor
                if isinstance(client_info.features, torch.Tensor):
                    features = client_info.features.cpu().numpy()
                else:
                    features = np.array(client_info.features)
                features_list.append(features)   # Append in the features list
            
            # Stack into matrix: (num_clients, feature_dim)
            features_matrix = np.vstack(features_list)

            # Determine actual number of clusters
            actual_num_clusters = min(self.num_clusters, len(self.clients))
            actual_num_clusters = max(1, actual_num_clusters)

            # Perform K-Means clustering
            if actual_num_clusters == 1:
                # All in one cluster
                labels = np.zeros(len(client_ids), dtype = int)
            else:
                kmeans = KMeans(
                    n_clusters = actual_num_clusters,
                    random_state = 42,
                    n_init = 10
                )
                labels = kmeans.fit_predict(features_matrix)
            
            # Clear existing clusters
            self.clusters.clear()

            # Create new clusters
            for i in range(actual_num_clusters):
                cluster = Cluster(cluster_id = i)
                self.clusters[i] = cluster

            # Assign clients to clusters
            for client_id, label in zip(client_ids, labels):
                cluster_id = int(label)
                client_info = self.clients[client_id]
                client_info.cluster_id = cluster_id     # Assign cluster id to client 
                client_info.is_cluster_head = False     # Reset head status within the cluster

                cluster = self.clusters[cluster_id]
                cluster.add_member(client_id)
            
            # Compute cluster centroids
            for cluster_id, cluster in self.clusters.items():
                self.compute_cluster_centroid(cluster_id) 
            
            # Select cluster heads (highest trust in each cluster)
            for cluster_id in self.clusters.keys():
                self.select_cluester_head(cluster_id)
            
            # Update state
            self.last_cluster_time = time.time()
            self.clustering_needed = False

            # Update statistics
            with self.stats_lock:
                self.stats["total_clusters"] = len(self.clusters)
                self.stats["num_reclustering"] += 1
                if len(self.clusters) > 0:
                    total_size = sum(c.size() for c in self.clusters.values())
                    self.stats["avg_cluster_size"] = total_size / len(self.clusters)
            
            # Print results
            print(f"[ClusterManager] Clustering complete:")
            print(f"Clusters: {len(self.clusters)}")
            for cluster_id, cluster in self.clusters.items():
                print(f"Cluster {cluster_id}: {cluster.size()} members, "
                      f"head={cluster.cluster_head_id}")
            
    def compute_cluster_centroid(self, cluster_id: int):
        """
        Compute average feature vector (centroid) for cluster

        Args:
            cluster_id: Cluster to compute centroid for
        """
        if cluster_id not in self.clusters:
            return
        
        cluster = self.clusters[cluster_id]
        if len(cluster.member_ids) == 0:
            cluster.centroid = None
            return
        
        # Gather features from all members
        features_list = []
        for client_id in cluster.member_ids:
            if client_id in self.clients:
                client_info = self.clients[client_id]
                features = client_info.features
                if isinstance(features, torch.Tensor):
                    features = features.cpu()
                features_list.append(features)
        
        if len(features_list) == 0:
            cluster.centroid = None
            return
        
        # Compute mean
        if isinstance(features_list[0], torch.Tensor):
            cluster.centroid = torch.stack(features_list).mean(dim = 0)
        else:
            cluster.centroid = np.mean(features_list, axis = 0)



