"""
Router: Expert selection with trust-weighted scoring

Implements the complete scoring formula:
Score_{ij} = T_{ij} * S_{ij} * e^(-lambda * delta_t_{ij})

Where:
- T_{ij} : Trust score (validation accuracy from head metadata)
- S_{ij}: Feature similarity (Cosine Distance)
- e^(-lambda * delta_t_{ij}) : Staleness decay factor

Integrates with Head's metadata system for complete expert evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import time
from head import Head

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config


class ExpertMetadata:
    """
    Metadata for each expert: M_j = {T, S, delta_t}

    This stores information received from peer heaads via their metadata.
    """
    def __init__(self, expert_id: int):
        self.expert_id = expert_id
        self.trust_score = 1.0   # T_{ij} -> From head's validation_accuracy
        self.last_update_time = time.time()   # For computing delta_t
        self.feature_embedding = None     # For computing similarity S_{ij}
        self.num_samples = 0    # From head metadata
        self.validation_accuracy = 0.0    # From head metadata

    def update_from_head_package(self, head_package: Dict):
        """
        Update metadata from a head package

        Args:
            head_package: Complete package from Head.get_head_package()
        """
        if "metadata" in head_package:
            metadata = head_package["metadata"]

            # Update trus score from validation accuracy
            if "validation_accuracy" in metadata:
                self.validation_accuracy = metadata["validation_accuracy"]
                self.trust_score = metadata["validation_accuracy"]
            elif "trust_score" in metadata:
                self.trust_score = metadata["trust_score"]

            # Update timestamp
            if "timestamp" in metadata:
                self.last_update_time = metadata["timestamp"]
            else:
                self.last_update_time = time.time()
            
            # Update other metadata
            if "num_samples" in metadata:
                self.num_samples = metadata["num_samples"]
    
    def update_timestamp(self):
        """Update timestamp when expert sends new head"""
        self.last_update_time = time.time()
    
    def get_staleness(self):
        """Compute delta_t: time since last update (in seconds)"""
        return time.time() - self.last_update_time
    
    def update_trust(self, performance: float, decay: float = 0.95):
        """
        Update trust score based on validation accuracy
        Trust score is updated using an Exponential Moving Average (EMA) 
        - Gradually update trust based on new performance evidence while remembering past performance
        - Gives balance between stability and adaptability -> D
        """
        self.trust_score = decay * self.trust_score + (1 - decay) * performance
        self.trust_score = max(0.1, min(2.0, self.trust_score))     # Clamping it to [0.1, 2.0]

class Router(nn.Module):
    """
    Router for expert selection with complete trust-weighted scoring

    Implements: Score_{ij} = T_{ij} * S_{ij} * e^(-lambda * delta_t{ij})
    """

    def __init__(
        self, 
        feature_dim: int = 512, 
        hidden_dim: int = 256,
        num_experts: int = 10,
        temperature: float = 1.0,
        top_k: int = 3,
        staleness_lambda: float = 0.001,
        similarity_type: str = "cosine"
    ):
        """
        Args:
            feature_dim: Input feature dimension (from body encoder)
            hidden_dim: Hidden layer dimension for scoring network
            num_experts: Number of available experts(clients)
            temperature: Temperature for softmax (higher = more uniform)
            top_k: Number of top experts to select
            staleness_lambda: Lambda parameter for e^(-lambda * delta_t)
            similarity_type: Type of similarity ("cosine" or "kl")
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = min(top_k, num_experts)
        self.staleness_lambda = staleness_lambda
        self.similarity_type = similarity_type

        # Expert metadata storage
        self.expert_metadata: Dict[int, ExpertMetadata] = {
            i : ExpertMetadata(i) for i in range(num_experts)    # {expert_id: Metadata(expert_id)}
        }

        # Neural network for learning additional routing patterns
        self.use_learned_routing = True
        if self.use_learned_routing:
            self.scoring_network = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace = True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_experts)
            )
        
        # Store reference features for each expert for similarity computation
        ## Stored in buffers (not parameters) -> Saved with model state but not trained
        self.register_buffer(
            "expert_features", torch.zeros(num_experts, feature_dim)
        )
        self.register_buffer(
            "expert_features_count", torch.zeros(num_experts)
        )

    def register_expert_head(
        self, 
        expert_id: int,
        head_package: Dict,
        features: Optional[torch.Tensor] = None
    ):
        """
        Register an expert head with its complete metadata

        Should be called when receiving a head package from a peer. 
        Updates: trust score, timestamp, feature embedding

        Args:
            expert_id: Expert ID
            head_package: Complete package from Head.get_head_package()
            features: Optional feature embedding for similarity computation
        """
        if expert_id not in self.expert_metadata:
            self.expert_metadata[expert_id] = ExpertMetadata(expert_id)

        # Update metadata from head package
        self.expert_metadata[expert_id].update_from_head_package(head_package)

        # Update feature embedding if provided
        if features is not None:
            self.update_expert_features(expert_id, features)
    
    def update_expert_features(
        self, 
        expert_id: int,
        features: torch.Tensor
    ):
        """
        Update stored features for an expert (for similarity computation)

        Args:
            expert_id: Expert ID
            features: Feature tensor (feature_dim,) or (B, feature_dim)
        """
        if features.dim() == 2:
            features = features.mean(dim = 0)   # Average if batch
        
        # Exponential if moving average
        alpha = 0.1
        if self.expert_features_count[expert_id] == 0:
            self.expert_features[expert_id] = features
        else:
            self.expert_features[expert_id] = (
                (1 - alpha) * self.expert_features[expert_id] + alpha * features
            )
        
        self.expert_features_count[expert_id] += 1
    
    def update_expert_timestamp(self, expert_id: int):
        """
        Update timestamp when expert sends new head

        This resets delta_t -> 0, so staleness factor -> 1.0
        """
        if expert_id in self.expert_metadata:
            self.expert_metadata[expert_id].update_timestamp()
    
    def update_trust(
        self,
        expert_id: int,
        performance: float, 
        decay: float = 0.95
    ):
        """
        Update trust score based on validation accuracy

        Args:
            expert_id: Expert ID
            performance: Validation accuracy (0 - 1)
            decay: EMA decay factor
        """
        if expert_id in self.expert_metadata:
            self.expert_metadata[expert_id].update_trust(performance, decay)
    
    def compute_similarity(
        self, 
        features: torch.Tensor,
        expert_id: int
    ) -> torch.Tensor:
        """
        Compute S_{ij} : Feature similarity between input and expert

        Args: 
            features: Input features (B, feature_dim)
            expert_id: Expert ID
        """
        expert_feat = self.expert_features[expert_id].unsqueeze(0)    # (1, feature_dim)

        if self.expert_features_count[expert_id] == 0:
            # Nofeatures stored yet -> Then return neutral similarity
            return torch.ones(features.size(0), device = features.device)

        if self.similarity_type == "cosine":
            # Cosine similarity: (f1 dot f2) / (||f1|| ||f2||)
            similarity = F.cosine_similarity(features, expert_feat, dim = 1)
            
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1.0) / 2.0

        elif self.similarity_type == "kl":
            # KL divergence (convert to similarity)
            f1 = F.softmax(features, dim = 1)
            f2 = F.softmax(expert_feat, dim = 1)
            kl_div = F.kl_div(f1.log(), f2, reduction = "none").sum(dim = 1)
            # Convert divergence to similarity: S = e^(-kl)
            similarity = torch.exp(-kl_div)
        
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        return similarity
    
    def compute_staleness_factor(self, expert_id: int) -> float:
        """
        Compute e^(-lambda * delta_t{ij}): Staleness decay factor

        Args:
            expert_id: Expert ID
        
        Returns:
            staleness_factor: Decay factor in [0, 1]
        """
        metadata = self.expert_metadata[expert_id]
        delta_t = metadata.get_staleness()

        # e^(-lambda * delta_t)
        staleness_factor = torch.exp(torch.tensor(-self.staleness_lambda * delta_t))
        return staleness_factor.item()

    def get_expert_status(self, expert_id: int) -> Dict:
        """Get current status of an expert"""
        if expert_id not in self.expert_metadata:
            return {}
        metadata = self.expert_metadata[expert_id]
        staleness = self.compute_staleness_factor(expert_id)

        return {
            "expert_id": expert_id,
            "trust_score": metadata.trust_score,
            "validation_accuracy": metadata.validation_accuracy,
            "staleness_factor": staleness,
            "delta_t": metadata.get_staleness(),
            "num_samples": metadata.num_samples,
            "has_features": self.expert_features_count[expert_id].item() > 0
        }
    
    def get_all_expert_status(self) -> List[Dict]:
        """Get status of all experts"""
        return [self.get_expert_status(i) for i in range(self.num_experts)]
    
    def reset_expert(self, expert_id: int):
        """Reset an expert's metadata (e.g., when peer disconnects)"""
        if expert_id in self.expert_metadata:
            self.expert_metadata[expert_id] = ExpertMetadata(expert_id)
            self.expert_features_count[expert_id] = 0
    
    def forward(
        self, 
        features: torch.Tensor, 
        available_experts: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: Compute expert selection scores

        Implements: Score_{ij} = T_{ij} * S_{ij} * e^(-lambda * delta_t{ij})

        Args:
            features: Input features (B, feature_dim)
            available_experts: List of available expert IDs (None = all)
        
        Returns:
            expert_weights: Softmax weights for experts (B, num_experts)
            expert_indices: Top-k expert indices (B, top_k)
            scoring_info: Dictionary with detailed scoring breakdown
        """
        batch_size = features.size(0)
        device = features.device

        if available_experts is None:
            available_experts = list(range(self.num_experts))
        
        # Initialize score matrix and components
        scores = torch.zeros(batch_size, self.num_experts, device = device)
        trust_scores = torch.zeros(self.num_experts, device = device)
        staleness_scores = torch.zeros(self.num_experts, device = device)
        similarity_scores = torch.zeros(batch_size, self.num_experts, device = device)

        # Computing scores for each expert
        for expert_id in available_experts:
            metadata = self.expert_metadata[expert_id]

            # T_{ij}: Trust Score (from validation accuracy)
            T_ij = metadata.trust_score
            trust_scores[expert_id] = T_ij

            # S_{ij}: Feature similarity
            S_ij = self.compute_similarity(features, expert_id)    # (B, )  
            similarity_scores[:, expert_id] = S_ij

            # e^(-lambda * delta_t{ij}): Staleness Decay
            staleness = self.compute_staleness_factor(expert_id)
            staleness_scores[expert_id] = staleness

            # Integrated: Score_{ij} = T_{ij} * S_{ij} * e^(-lambda * delta_t)
            expert_score = T_ij * S_ij * staleness    # (B, )
            scores[:, expert_id] = expert_score
        
        # Optional: Add learning routing component
        if self.use_learned_routing:
            learned_scores = self.scoring_network(features)
            learned_scores = torch.sigmoid(learned_scores)

            # Combine the weighted sum of formula-based and learned scores
            scores = 0.7 * scores + 0.3 * learned_scores

        # Applying temperature scaling
        scaled_scores = scores / self.temperature
    
        # Masking unavailable experts
        if len(available_experts) < self.num_experts:
            mask = torch.ones(self.num_experts, dtype = torch.bool, device = device)
            mask[available_experts] = False
            scaled_scores = scaled_scores.masked_fill(
                mask.unsqueeze(0).expand(batch_size, -1),
                float("-inf")
            )
        # Compute softmax weights
        expert_weights = torch.softmax(scaled_scores, dim = 1)

        # Select top-k experts
        top_k_weights, expert_indices = torch.topk(
            expert_weights,
            k = min(self.top_k, len(available_experts)),
            dim = 1,
            sorted = True
        )

        # Detailed scoring information for debugging/analysis
        scoring_info = {
            "trust_scores" : trust_scores,
            "similarity_scores": similarity_scores,
            "staleness_scores": staleness_scores,
            "raw_scores": scores,
            "available_experts": available_experts
        }
        
        return expert_weights, expert_indices, scoring_info

def create_router(config, device: torch.device) -> Router:
    """
    Factory function to create router from config

    Args:
        config: Configuration object with router settings
        device: torch.device to place model on

    Returns:
        Router module
    """
    router = Router(
        feature_dim = config.model.router.input_dim,
        hidden_dim = config.model.router.hidden_dim,
        num_experts = config.system.num_clients,
        temperature = config.model.router.temperature,
        top_k = config.network.top_k_experts,
        staleness_lambda = getattr(config.network, "staleness_lambda", 0.001),
        similarity_type = "cosine"
    )

    torch.backends.cudnn.benchmark = True
    return router.to(device)


# Testing Functions

def test_router():
    """Test router functionality"""
    print("\n" + "="*70)
    print("TESTING ROUTER (COMPLETE INTEGRATION WITH HEAD)")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test 1: Router Creation and Buffer Verification
    print("TEST 1: Router Creation and Buffer Verification")
    print("-" * 70)
    
    router = Router(
        feature_dim=512,
        hidden_dim=256,
        num_experts=10,
        temperature=1.0,
        top_k=3,
        staleness_lambda=0.001
    ).to(device)
    
    num_params = sum(p.numel() for p in router.parameters())
    print(f"✓ Router created")
    print(f"  Parameters: {num_params:,}")
    print(f"  Staleness lambda: {router.staleness_lambda}")
    
    # Verify buffers exist
    print(f"\nVerifying buffers:")
    print(f"  expert_features shape: {router.expert_features.shape}")
    print(f"  expert_features_count shape: {router.expert_features_count.shape}")
    assert router.expert_features.shape == (10, 512), "expert_features shape wrong!"
    assert router.expert_features_count.shape == (10,), "expert_features_count shape wrong!"
    print(f"  Buffers correctly initialized!")
    print()
    
    # Test 2: Integration with Head Package
    print("TEST 2: Integration with Head Package")
    print("-" * 70)
    
    try:        
        # Create mock head with metadata
        head = Head(input_dim=512, output_dim=10).to(device)
        head.update_metadata(
            validation_accuracy=0.87,
            num_samples=5000,
            expert_id=0
        )

        # Get complete package
        head_package = head.get_head_package(expert_id=0)
        
        print("Head package created")
        print(f"  Validation accuracy: {head_package['metadata']['validation_accuracy']:.2f}")
        print(f"  Trust score: {head_package['metadata']['trust_score']:.2f}")
        
        # Register with router
        dummy_features = torch.randn(512).to(device)
        router.register_expert_head(
            expert_id=0,
            head_package=head_package,
            features=dummy_features
        )
        
        print("\n Head registered with router")
        status = router.get_expert_status(0)
        print(f"  Router trust score: {status['trust_score']:.2f}")
        print(f"  Router validation acc: {status['validation_accuracy']:.2f}")
        print(f"  Staleness factor: {status['staleness_factor']:.4f}")
        print(f"  Has features: {status['has_features']}")
        
        # Verify expert_features was updated
        assert router.expert_features_count[0] > 0, "Features not stored!"
        print(f"  Feature count: {router.expert_features_count[0].item()}")
        
        assert abs(status['trust_score'] - 0.87) < 0.01, "Trust score mismatch!"
        print()
    
    except ImportError:
        print("head.py not found, using mock data")
        
        # Create mock head package
        mock_package = {
            'weights': {},
            'metadata': {
                'timestamp': time.time(),
                'trust_score': 0.87,
                'validation_accuracy': 0.87,
                'num_samples': 5000,
                'expert_id': 0
            }
        }
        
        dummy_features = torch.randn(512).to(device)
        router.register_expert_head(0, mock_package, dummy_features)
        status = router.get_expert_status(0)
        print(f"✓ Mock head registered")
        print(f"  Trust score: {status['trust_score']:.2f}")
        print(f"  Has features: {status['has_features']}")
        print()
    
    # Test 3: Complete Scoring Formula
    print("TEST 3: Complete Scoring Formula (T * S * e^(-λΔt))")
    print("-" * 70)
    
    batch_size = 4
    test_features = torch.randn(batch_size, 512).to(device)
    
    # Set up multiple experts with different metadata
    for i in range(5):
        expert_feat = torch.randn(512).to(device)
        router.update_expert_features(i, expert_feat)
        
        # Different trust scores
        if i == 0:
            router.update_trust(i, 0.9)  # High trust
        elif i == 2:
            router.update_trust(i, 0.5)  # Medium trust
    
    # Verify features were stored
    print("Expert features stored:")
    for i in range(5):
        count = router.expert_features_count[i].item()
        print(f"  Expert {i}: {count} updates")
    
    with torch.no_grad():
        expert_weights, expert_indices, scoring_info = router(test_features)
    
    print(f"\n  Forward pass with complete formula")
    print(f"\n  Expert Status (first 3):")
    for i in range(3):
        status = router.get_expert_status(i)
        print(f"    Expert {i}:")
        print(f"      Trust (T): {status['trust_score']:.4f}")
        print(f"      Staleness (e^(-λΔt)): {status['staleness_factor']:.4f}")
        print(f"      Δt: {status['delta_t']:.2f}s")
        print(f"      Has features: {status['has_features']}")
    
    print(f"\n  Sample 0 - Top-{router.top_k} selected:")
    for i in range(router.top_k):
        expert_id = expert_indices[0, i].item()
        weight = expert_weights[0, expert_id].item()
        status = router.get_expert_status(expert_id)
        print(f"    Expert {expert_id}: weight={weight:.4f}, trust={status['trust_score']:.2f}")
    print()
    
    # Test 4: Staleness Decay Over Time
    print("TEST 4: Staleness Decay (Offline Peer)")
    print("-" * 70)
    
    print("Scenario: Expert 5 goes offline")
    
    # Fresh update
    router.update_expert_timestamp(5)
    status_fresh = router.get_expert_status(5)
    print(f"  Fresh: staleness = {status_fresh['staleness_factor']:.6f}")
    
    # Wait
    print("\n  Waiting 2 seconds...")
    time.sleep(2)
    
    status_stale = router.get_expert_status(5)
    print(f"  After 2s: Δt = {status_stale['delta_t']:.2f}s")
    print(f"  After 2s: staleness = {status_stale['staleness_factor']:.6f}")
    print(f"  Decay: {(1 - status_stale['staleness_factor']) * 100:.2f}%")
    
    assert status_stale['staleness_factor'] < status_fresh['staleness_factor']
    print("\n Staleness decay working")
    print()
    
    # Test 5: Expert Returns
    print("TEST 5: Expert Returns (Fresh Update)")
    print("-" * 70)
    
    print("Scenario: Expert 5 reconnects")
    router.update_expert_timestamp(5)
    
    status_restored = router.get_expert_status(5)
    print(f"  After update: Δt = {status_restored['delta_t']:.6f}s")
    print(f"  After update: staleness = {status_restored['staleness_factor']:.6f}")
    
    assert status_restored['staleness_factor'] > 0.99
    print("\n Staleness restored")
    print()
    
    # Test 6: Feature Similarity
    print("TEST 6: Feature Similarity (S_ij)")
    print("-" * 70)
    
    test_feat = torch.randn(1, 512).to(device)
    
    # Similar features
    similar = test_feat[0] + 0.1 * torch.randn(512).to(device)
    router.update_expert_features(6, similar)
    
    # Different features
    different = torch.randn(512).to(device)
    router.update_expert_features(7, different)
    
    sim_6 = router.compute_similarity(test_feat, 6)
    sim_7 = router.compute_similarity(test_feat, 7)
    
    print(f"  Similarity to Expert 6 (similar): {sim_6.item():.4f}")
    print(f"  Similarity to Expert 7 (different): {sim_7.item():.4f}")
    
    # Verify feature counts were updated
    print(f"\n  Feature counts:")
    print(f"    Expert 6: {router.expert_features_count[6].item()}")
    print(f"    Expert 7: {router.expert_features_count[7].item()}")
    
    assert sim_6 > sim_7
    print("\n Similarity computation working")
    print()
    
    # Test 7: Available Experts Masking
    print("TEST 7: Available Experts Masking")
    print("-" * 70)
    
    available = [0, 2, 5, 7]
    
    with torch.no_grad():
        weights, indices, info = router(test_features, available_experts=available)
    
    print(f"Available: {available}")
    print(f"Selected (sample 0):")
    for i in range(router.top_k):
        eid = indices[0, i].item()
        in_avail = "✓" if eid in available else "✗"
        print(f"  Expert {eid} [{in_avail}]")
    
    selected = indices[0].tolist()
    assert all(e in available for e in selected)
    print("\n✓ Masking working")
    print()
    
    # Test 8: create_router with config
    print("TEST 8: create_router() with config")
    print("-" * 70)
    
    try:
        config_path = project_root / 'configs' / 'config.yaml'
        
        if config_path.exists():
            config = load_config(str(config_path))
            router_config = create_router(config, device)
            
            print(f"✓ Router from config")
            print(f"  Num experts: {router_config.num_experts}")
            print(f"  Top-k: {router_config.top_k}")
            
            # Verify buffers in config router
            print(f"  Buffers exist: {hasattr(router_config, 'expert_features')}")
            print(f"  Buffer shape: {router_config.expert_features.shape}")
            print()
        else:
            print(" Config not found")
            print()
    
    except Exception as e:
        print(f" Error: {e}")
        print()
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_router()



