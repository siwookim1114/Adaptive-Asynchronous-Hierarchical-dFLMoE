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
    
