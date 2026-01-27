"""
Client Node - Main Orchestration for Federated Learning

Integrates all components (cache, transport, clustering, routing)
into a complete federated learning client with adaptive asynchronous updates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from typing import Dict, List, Optional, Tuple

# Load components
from infra.peer_cache import PeerCache, ExpertPackage
from infra.transport import TCPTransport, Message
from infra.cluster import ClusterManager

class SparseRouter(nn.Module):
    """
    Trust-weighted sparse router for expert selection

    Selects top-K experts based on multi-factor scoring:
    - Trust Score (Reliability)
    - Similarity (Feature Distance)
    - Staleness (Freshness)
    """
    def __init__(
        self, 
        input_dim: int,
        num_classes: int,
        trust_weight: float = 0.4,
        similarity_weight: float = 0.4,
        staleness_weight: float = 0.2
    ):
        """
        Args:
            input_dim: Feature dimension
            num_classes: Number of output classes
            trust_weight: Weight for trust score in selection
            similarity_weight: Weight for feature similarity
            staleness_weight: Weight for staleness (negative)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Scorings
        self.trust_weight = trust_weight
        self.similarity_weight = similarity_weight
        self.staleness_weight = staleness_weight

        print(f"[SparseRouter] Initialized (trust={trust_weight}, "
              f"sim={similarity_weight}, stale={staleness_weight})")
        
    def compute_score(self, query_features: torch.Tensor, expert_package: ExpertPackage) -> float:
        """
        Compute selection score for an expert

        Args:
            query_features: Query features (current input embedding)
            expert_package: Expert package to score
        
        Returns:
            Selection score
        """
        # Trust score (0 ~ 1)
        trust = expert_package.trust_score

        # Similarity score (cosine similarity: -1 to 1, normalized to 0 to 1)
        expert_features = expert_package.representative_features

        # Ensure both on same device
        if query_features.device != expert_features.device:
            expert_features = expert_features.to(query_features.device)   # Aligning
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            query_features.unsqueeze(0),
            expert_features.unsqueeze(0)
        ).item()
        similarity = (similarity + 1) / 2     # Normalization to [0, 1]

        # Staleness score (0 to 1, inverted: fresh = high score)
        staleness = expert_package.get_staleness()
        staleness_score = max(0, 1 - staleness)   # Invert: low staleness = high score

        # Combined scores
        score = (self.trust_weight * trust + self.similarity_weight * similarity + self.staleness_weight * staleness_score)
        return score
    
    def select_top_k(
        self,
        query_features: torch.Tensor,
        expert_packages: List[ExpertPackage],
        k: int = 3
    ) -> List[ExpertPackage]:
        """
        Select top-K experts based on multi-factor scoring

        Args:
            query_features: Query features for similarity computation
            expert_packages: Avaialble expert packages
            k: Number of experts to select

        Returns:
            List of top-K expert packages
        """
        if len(expert_packages) == 0:
            return []
        
        # Compute scores for all experts
        scored_experts = []
        for package in expert_packages:
            score = self.compute_score(query_features, package)
            scored_experts.append((score, package))
        
        # Sort by score (descending)
        scored_experts.sort(key = lambda x: x[0], reverse = True)

        # Return top-K
        top_k = min(k, len(scored_experts))
        return [package for score, package in scored_experts[:top_k]]
    
    def forward(self, x: torch.Tensor, expert_packages: List[ExpertPackage], k: int = 3) -> torch.Tensor:
        """
        Forward pass with expert routing

        Args:
            x: Input features (batch_size, input_dim)
            expert_packages: Available expert packages
            k: Number of experts to use
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize output
        outputs = torch.zeros(batch_size, self.num_classes).to(device)

        if len(expert_packages) == 0:
            return outputs
        
        # For each sample in batch
        for i in range(batch_size):
            query_features = x[i]

            # Select top-K experts
            selected_experts = self.select_top_k(query_features, expert_packages, k)

            if len(selected_experts) == 0:
                continue
        
            # Aggregate predictions from selected experts
            expert_outputs = []
            for package in selected_experts:
                # Load expert head
                expert_head = nn.Linear(self.input_dim, self.num_classes)
                expert_head.load_state_dict(package.head_state_dict) # Load expert's head parameters dict
                expert_head = expert_head.to(device)
                expert_head.eval()

                # Get Prediction
                with torch.no_grad():
                    output = expert_head(query_features.unsqueeze(0))   # Forward pass to head
                expert_outputs.append(output)
            
            # Average Predictions
            outputs[i] = torch.stack(expert_outputs).mean(dim = 0)
        return outputs

class ClientNode:
    """
    Complete federated learning client node

    Integrates:
    - Peer Cache: Expert Storage
    - TCPTransport: P2P communication
    - ClusterManager: Hierarchical organization
    - SparseRouter: Trust-weighted expert selection

    - Adaptive asynchronous updates
    - Hierarchical communication
    - Trust score computation
    - Staleness tracking
    """
    def __init__(
        self,
        client_id: str,
        cluster_manager: ClusterManager,
        host: str = "0.0.0.0",
        port: int = 0,
        feature_dim: int = 512,
        num_classes: int = 10,
        max_cache_size: int = 50,
        staleness_lambda: float = 0.001,
        top_k_experts: int = 3
    ):
        """
        Initialize federated client node

        Args:
            client_id: Unique identifier for this client
            cluster_manager: Shared cluster manager instance
            host: Host for transport server
            port: Port for transport server (0 = auto)
            feature_dim: Dimension of feature embeddings
            num_classes: Number of output classes
            max_cache_size: Maximum experts in cache
            staleness_lambda: Decay rate for staleness
            top_k_experts: Number of experts to select for routing
        """
        self.client_id = client_id
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.top_k_experts = top_k_experts

        # Componenets
        self.cache = PeerCache(
            max_cache_size = max_cache_size,
            staleness_lambda = staleness_lambda
        )

        self.transport = TCPTransport(
            client_id = client_id,
            host = host,
            port = port
        )

        self.cluster_manager = cluster_manager

        self.router = SparseRouter(
            input_dim = feature_dim,
            num_classes = num_classes
        )

        # Model components (to be set by user)
        self.body_encoder = None    # Shared backbone
        self.head = None    # Local expert head

        # Training state
        self.trust_score = 0.5                                     # Initial trust
        self.validation_accuracy = 0.0
        self.num_samples = 0
        self.representative_features = torch.randn(feature_dim)    # Initial

        # Communication settings
        self.cluster_exchange_interval = 1    # Every round
        self.cross_cluster_exchange_interval = 5      # Every 5 rounds
        self.current_round = 0

        # Statistics
        self.stats = {
            "rounds_completed": 0,
            "experts_sent": 0,
            "experts_received": 0,
            "experts_used": 0,
            "trust_updates": 0
        }

        self.stats_lock = threading.Lock()

        # Setup transport handler
        self.transport.register_handler("expert_package", self._handle_expert_package)

        # Register with cluster manager
        self.cluster_manager.register_client(
            client_id = self.client_id,
            features = self.representative_features,
            trust_score = self.trust_score
        )

        print(f"[ClientNode:{client_id}] Initialized on {host}:{self.transport.port}")

    def set_model(self, body_encoder: nn.Module, head: nn.Module):
        """
        Set model components

        Args:
            body_encoder: Shared backbone network
            head: Local expert head (classifier)
        """
        self.body_encoder = body_encoder
        self.head = head
        print(f"[ClientNode:{self.client_id}] Model set")

    def _handle_expert_package(self, message: Message):
        """
        Handle incoming expert package

        Args:
            message: Message containing expert package
        """
        package = message.payload
        
        # Add to cache
        success = self.cache.add(package)

        if success:
            with self.stats_lock:
                self.stats["experts_received"] += 1
            
            print(f"[ClientNode:{self.client_id}] Cached expert from "
                  f"{package.client_id} (trust={package.trust_score:.3f}, "
                  f"cache_size={self.cache.size()})")
        
        else:
            print(f"[ClientNode:{self.client_id}]Failed to cache expert from "
                  f"{package.client_id}")
            
    def compute_representative_features(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Compute representative features (average embeddings from data)

        Args:
            data_loader: Training data loader
        
        Returns:
            Representative features (feature_dim,)
        """
        if self.body_encoder is None:
            print(f"[ClientNode:{self.client_id}] Body encoder not set")
            return torch.randn(self.feature_dim)

        self.body_encoder.eval()
        features_list = []

        with torch.no_grad():
            for x, _ in data_loader:
                # Get device from model
                device = next(self.body_encoder.parameters()).device
                x = x.to(device)

                # Extract features
                features = self.body_encoder(x)
                features_list.append(features)
            
        # Compute mean
        if len(features_list) > 0:
            all_features = torch.cat(features_list, dim = 0)
            representative = all_features.mean(dim = 0)
            self.representative_features = representative
            return representative 
        else:
            return self.representative_features
    
    def compute_trust_score(self, validation_accuracy: float, num_samples: float):
        """
        Compute trust score based on validation accuracy and data size

        Args:
            validation_accuracy: Validation accuracy (0 to 1)
            num_samples: Number of training samples
        """
        # Base trust on accuracy
        accuracy_component = validation_accuracy

        # Data size component(more data = more trust)
        # Normalize by 10000 samples
        data_component = min(1.0, num_samples / 10000.0)

        # Combined trust (weighted_average)
        self.trust_score = 0.7 * accuracy_component + 0.3 * data_component
        self.trust_score = max(0.0, min(1.0, self.trust_score))  # Clamp to [0, 1]

        self.validation_accuracy = validation_accuracy
        self.num_samples = num_samples

        with self.stats_lock:
            self.stats["trust_updates"] += 1
        
        print(f"[ClientNode:{self.client_id}] Trust updated: {self.trust_score:.3f} "
              f"(acc={validation_accuracy:.3f}, samples={num_samples})")
    
    def create_expert_package(self) -> ExpertPackage:
        """
        Create expert package for sharing

        Returns:
            ExpertPackage with current expert head
        """
        if self.head is None:
            raise ValueError("Head not set")
        
        package = ExpertPackage(
            client_id = self.client_id,
            head_state_dict = self.head.state_dict(),
            timestamp = time.time(),
            trust_score = self.trust_score,
            validation_accuracy = self.validation_accuracy,
            representative_features = self.representative_features,
            num_samples = self.num_samples
        )

        return package
    
    def share_expert(self):
        """
        Share expert package with peers (hierarchical)

        Shares with:
        - Cluster peers (every cluster_exchange_interval_rounds)
        - Cluster heads (every cross_cluster_exchange_interval rounds, if head)
        """
        # Create expert package
        package = self.create_expert_package()

        # Get communication targets
        targets = self.cluster_manager.get_communication_targets(self.client_id) # Get the targets for each client_id respective Nodes
        
        experts_sent = 0

        # Within cluster (frequent)
        if self.current_round % self.cluster_exchange_interval == 0:
            cluster_peers = targets["cluster_peers"]
            if len(cluster_peers) > 0:
                num_sent = self.transport.broadcast(
                    cluster_peers,
                    "expert_package",
                    package
                )
                experts_sent += num_sent
                print(f"[ClientNode:{self.client_id}] Shared with "
                      f"{num_sent}/{len(cluster_peers)} cluster peers")
        
        # Across clusters (less frequent, if cluster head)
        if self.current_round % self.cross_cluster_exchange_interval == 0:
            if self.cluster_manager.is_cluster_head(self.client_id):
                cluster_heads = targets["cluster_heads"]
                if len(cluster_heads) > 0:
                    num_sent = self.transport.broadcast(
                        cluster_heads,
                        "expert_package",
                        package
                    )
                    experts_sent += num_sent
                    print(f"[ClientNode:{self.client_id}] Shared with "
                          f"{num_sent}/{len(cluster_heads)} cluster heads "
                          f"(CROSS-CLUSTER)")
        
        with self.stats_lock:
            self.stats["experts_sent"] += experts_sent
        
    


        
    





