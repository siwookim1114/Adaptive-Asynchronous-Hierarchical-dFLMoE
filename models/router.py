"""
1. Explicit scoring: Score_{ij} = T_{ij} * S_{ij} * e^(-lambda * delta_t_{ij})
2. Learnable gating: sigmoid(proj(features) · expert_emb_j / √d) * base_score
3. Feature Space Transforms: FST_{i -> j}
4. MoE Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import time
import math
from models.fst import FeatureSpaceTransform

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
class ExpertMetadata:
    """
    Metadata for each expert: M_j = {T, S, delta_t}

    This stores information received from peer heads via their metadata.
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

            # Update trust score from validation accuracy
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
    - Explicit scoring formula
    - Learnable gating network
    - FST integration
    - MoE aggregation
    """

    def __init__(
        self, 
        feature_dim: int = 512, 
        hidden_dim: int = 256,
        num_experts: int = 10,
        num_classes: int = 10,
        temperature: float = 1.0,
        top_k: int = 3,
        staleness_lambda: float = 0.001,
        similarity_type: str = "cosine",
        use_learned_gating: bool = True
    ):
        """
        Args:
            feature_dim: Input feature dimension (from body encoder)
            hidden_dim: Hidden layer dimension for scoring network
            num_experts: Number of available experts(clients)
            num_classes: Number of output classes
            temperature: Temperature for softmax (higher = more uniform) for routing
            top_k: Number of top experts to select
            staleness_lambda: Lambda parameter for e^(-lambda * delta_t)
            similarity_type: Type of similarity ("cosine" or "kl")
            use_learned_gating: Enable learnable gating network
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.temperature = temperature
        self.top_k = min(top_k, num_experts)
        self.staleness_lambda = staleness_lambda
        self.similarity_type = similarity_type
        self.use_learned_gating = use_learned_gating

        # Expert metadata storage
        self.expert_metadata: Dict[int, ExpertMetadata] = {
            i : ExpertMetadata(i) for i in range(num_experts)    # {expert_id: Metadata(expert_id)}
        }

        # Learned gating via scaled dot-product: sigmoid(proj(features) · expert_emb_j / √d)
        if self.use_learned_gating:
            # Project input features into hidden space (computed once per input)
            self.feature_projector = nn.Linear(feature_dim, hidden_dim)

            # Learnable embedding per expert (indexed by expert_id)
            self.expert_embeddings = nn.Embedding(num_experts, hidden_dim)
            # Small init so sigmoid(~0) ≈ 0.5, letting base_score dominate initially
            nn.init.normal_(self.expert_embeddings.weight, mean=0, std=0.01)

            print(f"[Router] Gating: proj({feature_dim}->{hidden_dim}) · expert_emb({num_experts}x{hidden_dim})")
            
        # Feature Space Trnasforms (one per expert)
        self.fst_transforms = nn.ModuleDict()

        # Store reference features for each expert for similarity computation (store each expert's feature spaces)
        ## Stored in buffers (not parameters) -> Saved with model state but not trained
        self.register_buffer(
            "expert_features", torch.zeros(num_experts, feature_dim)
        )
        self.register_buffer(
            "expert_features_count", torch.zeros(num_experts)         # Create buffer to count how many times we've updsated each expert's features
        )
    
    def get_or_create_fst(self, expert_id: int) -> FeatureSpaceTransform:
        """
        Get or create FST_{i->j} for expert j

        FST aligns features from local space to expert's space.
        Created lazily when expert is first registered.

        Args:
            expert_id: Expert ID
        
        Returns:
            FST module for this expert
        """
        expert_key = str(expert_id)    # ModuleDict requires string keys

        if expert_key not in self.fst_transforms:
            # Create new FST (from fst.py)
            fst = FeatureSpaceTransform(self.feature_dim)
            self.fst_transforms[expert_key] = fst    # Create fst and store it
            print(f"[Router] Created FST for expert {expert_id}")
        
        return self.fst_transforms[expert_key]

    def register_expert(
        self,
        expert_id: int,
        trust_score: float,
        validation_accuracy: float,
        features: torch.Tensor,
        num_samples: int = 0,
        timestamp: Optional[float] = None
    ):
        """Register an expert with router"""
        if expert_id not in self.expert_metadata:
            self.expert_metadata[expert_id] = ExpertMetadata(expert_id)
        
        metadata = self.expert_metadata[expert_id]
        metadata.trust_score = trust_score
        metadata.validation_accuracy = validation_accuracy
        metadata.num_samples = num_samples
        metadata.last_update_time = timestamp if timestamp else time.time()

        self.update_expert_features(expert_id, features)
        self.get_or_create_fst(expert_id)
    
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
        features = features.detach().to(self.expert_features.device)

        # Exponential moving average
        alpha = 0.1
        if self.expert_features_count[expert_id] == 0:
            self.expert_features[expert_id] = features
        else:
            self.expert_features[expert_id] = (
                (1 - alpha) * self.expert_features[expert_id] + alpha * features
            )
        
        self.expert_features_count[expert_id] += 1
    
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
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        expert_feat = self.expert_features[expert_id].unsqueeze(0)    # (1, feature_dim)

        if self.expert_features_count[expert_id] == 0:
            # No features stored yet -> Then return neutral similarity (0.5)
            result = torch.ones(features.size(0), device = features.device) * 0.5
            return result

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
        return similarity.squeeze(0) if squeeze_output else similarity
    
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
        return math.exp(-self.staleness_lambda * delta_t)
    
    def compute_base_score(self, trust: float, similarity: float, staleness_factor: float) -> float:
        """
        Computes base score: Score = T * S * staleness_factor

        Args:
            trust: Trust score T_{ij}
            similarity: Feature similarity score S_{ij}
            staleness_factor: Pre-computed decay factor e^(-lambda * delta_t) in [0, 1]

        Returns:
            Base score from the combination of the three factors
        """
        return trust * similarity * staleness_factor
    
    def compute_routing_weight(
        self,
        query_features: torch.Tensor,
        expert_id: int,
        projected_features: Optional[torch.Tensor] = None
    ) -> float:
        """
        Computes the routing weight using metadata and per-expert learned gate

        Gate = sigmoid(proj(features) · expert_emb_j / √hidden_dim)
        Final = gate × base_score

        Args:
            query_features: Input embeddings (Input data)
            expert_id: Expert ID of the expert
            projected_features: Pre-computed feature projection to avoid redundant forward passes

        Returns:
            Computed final score combining metadata base score and learned gate
        """
        metadata = self.expert_metadata[expert_id]
        trust = metadata.trust_score
        similarity = self.compute_similarity(query_features, expert_id)

        if isinstance(similarity, torch.Tensor):
            similarity = similarity.item()

        staleness_factor = self.compute_staleness_factor(expert_id)
        base_score = self.compute_base_score(trust, similarity, staleness_factor)

        if self.use_learned_gating:
            if projected_features is None:
                projected_features = self.feature_projector(query_features.unsqueeze(0)).squeeze(0)

            expert_emb = self.expert_embeddings(
                torch.tensor(expert_id, device=query_features.device)
            )
            affinity = torch.dot(projected_features, expert_emb) / math.sqrt(self.hidden_dim)
            gate_value = torch.sigmoid(affinity).item()
            return gate_value * base_score

        return base_score
    
    def select_top_k_experts(
        self,
        query_features: torch.Tensor,
        available_experts: Optional[List[int]] = None,
        k: Optional[int] = None
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Selects top-K experts from the available experts with computed routing weight for each data input

        Args:
            query_features: Input embeddings (Input data)
            available_experts: Experts that could be used for selection
            k: Number of experts to use

        Returns:
            (selected_expert_ids, normalized_weights)
        """
        if k is None:
            k = self.top_k
        if available_experts is None:
            available_experts = list(range(self.num_experts))

        # Pre-compute feature projection once (shared across all experts)
        # Note: No torch.no_grad() here to allow gradient flow for router learning
        projected_features = None
        if self.use_learned_gating:
            projected_features = self.feature_projector(query_features.unsqueeze(0)).squeeze(0)

        scores = []
        valid_expert_ids = []

        for expert_id in available_experts:
            if self.expert_features_count[expert_id] > 0:
                score = self.compute_routing_weight(query_features, expert_id, projected_features=projected_features)
                scores.append(score)
                valid_expert_ids.append(expert_id)

        if len(scores) == 0:
            return [], torch.tensor([])

        scores = torch.tensor(scores, device=query_features.device)
        k = min(k, len(scores))
        top_k_scores, top_k_indices = torch.topk(scores, k)
        normalized_weights = F.softmax(top_k_scores / self.temperature, dim=0)
        selected_ids = [valid_expert_ids[i] for i in top_k_indices]
        return selected_ids, normalized_weights
    
    def forward_moe(
        self,
        features: torch.Tensor,
        expert_heads: Dict[int, nn.Module],
        available_experts: Optional[List[int]] = None,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """MoE Forward Passes with FST integrated"""
        batch_size = features.size(0)
        device = features.device
        outputs = torch.zeros(batch_size, self.num_classes, device=device)

        if len(expert_heads) == 0:
            return outputs

        # Filter to only experts that have heads available (prevents weight loss from skipping)
        if available_experts is None:
            valid_experts = list(expert_heads.keys())
        else:
            valid_experts = [eid for eid in available_experts if eid in expert_heads]

        if len(valid_experts) == 0:
            return outputs

        for i in range(batch_size):
            query_features = features[i]
            selected_ids, weights = self.select_top_k_experts(query_features, valid_experts, k)

            if len(selected_ids) == 0:
                continue

            sample_output = torch.zeros(self.num_classes, device=device)

            for expert_id, weight in zip(selected_ids, weights):
                fst = self.get_or_create_fst(expert_id)
                fst = fst.to(device)
                aligned_features = fst(query_features.unsqueeze(0))

                expert_head = expert_heads[expert_id]
                expert_head.eval()

                # Note: No torch.no_grad() to allow gradient flow through FST
                # Expert head params won't be updated since they're not in any optimizer
                expert_logits = expert_head(aligned_features).squeeze(0)

                sample_output += weight * expert_logits
            outputs[i] = sample_output
        return outputs
    
    def get_expert_status(self, expert_id: int) -> Dict:
        if expert_id not in self.expert_metadata:
            return {"expert_id": expert_id, "registered": False}
        
        metadata = self.expert_metadata[expert_id]
        staleness_factor = self.compute_staleness_factor(expert_id)

        return {
            "expert_id": expert_id,
            "registered": True,
            "trust_score": metadata.trust_score,
            "validation_accuracy": metadata.validation_accuracy,
            "staleness_factor": staleness_factor,
            "delta_t": metadata.get_staleness(),
            "num_samples": metadata.num_samples,
            "has_features": self.expert_features_count[expert_id].item() > 0,
            "has_fst": str(expert_id) in self.fst_transforms
        }

    def get_all_expert_status(self) -> List[Dict]:
        return [self.get_expert_status(i) for i in range(self.num_experts) if self.expert_features_count[i] > 0]
    
    def reset_expert(self, expert_id: int):
        if expert_id in self.expert_metadata:
            self.expert_metadata[expert_id] = ExpertMetadata(expert_id)
            self.expert_features_count[expert_id] = 0
            expert_key = str(expert_id)
            if expert_key in self.fst_transforms:
                del self.fst_transforms[expert_key]
            if self.use_learned_gating and expert_id < self.num_experts:
                nn.init.normal_(self.expert_embeddings.weight[expert_id], mean=0, std=0.01)
