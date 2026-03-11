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
        self.trust_score = 0.5   # Neutral init (not optimistic 1.0) for faster EMA convergence
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
        self.trust_score = max(0.1, min(1.0, self.trust_score))     # Clamping to [0.1, 1.0]

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
        staleness_lambda: float = 0.005,
        staleness_floor: float = 0.1,
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
            staleness_floor: Minimum staleness factor to prevent complete expert suppression
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
        self.staleness_floor = staleness_floor
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
            # Moderate init so experts are distinguishable from round 1
            nn.init.normal_(self.expert_embeddings.weight, mean=0, std=0.1)

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
        metadata.update_trust(trust_score)  # EMA: smooth evolution instead of direct overwrite
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
            
            # Clamp to [0, 1] without compressing range
            # (sim+1)/2 would map 0.6-0.9 to 0.8-0.95 (15% spread)
            # clamp keeps 0.6-0.9 as 0.6-0.9 (30% spread, 2x more discriminative)
            similarity = torch.clamp(similarity, min=0.0)

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
        Compute max(floor, e^(-lambda * delta_t{ij})): Staleness decay factor

        The floor prevents complete suppression of stale experts during training,
        ensuring the router can still learn from cross-cluster experts even when
        they haven't been refreshed recently. Fresh experts still score ~8x higher
        than floor-clamped ones (e.g., 0.86 vs 0.1), preserving the preference
        for fresher experts.

        Args:
            expert_id: Expert ID

        Returns:
            staleness_factor: Decay factor in [floor, 1]
        """
        metadata = self.expert_metadata[expert_id]
        delta_t = metadata.get_staleness()
        return max(self.staleness_floor, math.exp(-self.staleness_lambda * delta_t))
    
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
    
    def compute_routing_weight_tensor(
        self,
        query_features: torch.Tensor,
        expert_id: int,
        projected_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the routing weight as a DIFFERENTIABLE tensor

        Gate = sigmoid(proj(features) · expert_emb_j / √hidden_dim)
        Final = gate × base_score

        Args:
            query_features: Input embeddings (feature_dim,) or (B, feature_dim)
            expert_id: Expert ID of the expert
            projected_features: Pre-computed feature projection (hidden_dim,) or (B, hidden_dim)

        Returns:
            Tensor routing weight (scalar or (B,)) - keeps gradient graph intact
        """
        device = query_features.device
        metadata = self.expert_metadata[expert_id]

        # Trust and staleness as tensors (constants, no grad needed)
        trust = torch.tensor(metadata.trust_score, device=device, dtype=torch.float32)
        staleness_factor = torch.tensor(self.compute_staleness_factor(expert_id), device=device, dtype=torch.float32)

        # Similarity as tensor (keeps any gradients from features)
        similarity = self.compute_similarity(query_features, expert_id)
        if not isinstance(similarity, torch.Tensor):
            similarity = torch.tensor(similarity, device=device, dtype=torch.float32)

        # Base score: T * S * staleness (tensor multiplication)
        base_score = trust * similarity * staleness_factor

        if self.use_learned_gating:
            # Ensure we have projected features
            is_batch = query_features.dim() == 2
            if projected_features is None:
                if is_batch:
                    projected_features = self.feature_projector(query_features)  # (B, hidden_dim)
                else:
                    projected_features = self.feature_projector(query_features.unsqueeze(0)).squeeze(0)  # (hidden_dim,)

            # Get expert embedding
            expert_idx = torch.tensor(expert_id, device=device)
            expert_emb = self.expert_embeddings(expert_idx)  # (hidden_dim,)

            # Compute affinity (scaled dot product)
            if is_batch:
                # Batched: (B, hidden_dim) @ (hidden_dim,) -> (B,)
                affinity = (projected_features @ expert_emb) / math.sqrt(self.hidden_dim)
            else:
                # Single: (hidden_dim,) dot (hidden_dim,) -> scalar
                affinity = torch.dot(projected_features, expert_emb) / math.sqrt(self.hidden_dim)

            # Gate value (differentiable!)
            gate_value = torch.sigmoid(affinity)

            # Final score: gate * base_score (fully differentiable)
            return gate_value * base_score

        return base_score
    
    def _compute_scores_batched(
        self,
        features: torch.Tensor,
        valid_experts: List[int]
    ) -> torch.Tensor:
        """Compute routing scores for all samples x all experts at once.

        Returns a (B, num_valid) tensor of differentiable routing scores.
        The feature projection is computed once and shared across all experts.

        Args:
            features: (B, feature_dim)
            valid_experts: List of expert IDs

        Returns:
            (B, num_valid) tensor of routing scores
        """
        device = features.device

        # Project features once (shared across all experts)
        projected = None
        if self.use_learned_gating:
            projected = self.feature_projector(features)  # (B, hidden_dim)

        score_list = []
        for eid in valid_experts:
            metadata = self.expert_metadata[eid]
            trust = metadata.trust_score
            staleness = self.compute_staleness_factor(eid)

            # Similarity: (B,)
            similarity = self.compute_similarity(features, eid)
            if not isinstance(similarity, torch.Tensor):
                similarity = torch.tensor(similarity, device=device, dtype=torch.float32)

            base_score = trust * similarity * staleness  # (B,)

            if self.use_learned_gating and projected is not None:
                expert_idx = torch.tensor(eid, device=device)
                expert_emb = self.expert_embeddings(expert_idx)  # (hidden_dim,)
                affinity = (projected @ expert_emb) / math.sqrt(self.hidden_dim)  # (B,)
                gate = torch.sigmoid(affinity)  # (B,)
                score_list.append(gate * base_score)
            else:
                score_list.append(base_score)

        return torch.stack(score_list, dim=1)  # (B, num_valid)

    def select_top_k_experts(
        self,
        query_features: torch.Tensor,
        available_experts: Optional[List[int]] = None,
        k: Optional[int] = None
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Selects top-K experts with DIFFERENTIABLE routing weights

        This method keeps all computations as tensors to enable gradient flow
        through the router's feature_projector and expert_embeddings.

        Args:
            query_features: Input embeddings (feature_dim,) - single sample
            available_experts: Experts that could be used for selection
            k: Number of experts to use

        Returns:
            (selected_expert_ids, normalized_weights) where weights are differentiable
        """
        if k is None:
            k = self.top_k
        if available_experts is None:
            available_experts = list(range(self.num_experts))

        device = query_features.device

        # Pre-compute feature projection once (shared across all experts)
        # This is the key differentiable computation for router learning
        projected_features = None
        if self.use_learned_gating:
            projected_features = self.feature_projector(query_features.unsqueeze(0)).squeeze(0)

        # Compute scores as tensors (maintain gradient graph)
        score_tensors = []
        valid_expert_ids = []

        for expert_id in available_experts:
            if self.expert_features_count[expert_id] > 0:
                # Returns a tensor, preserving gradients!
                score = self.compute_routing_weight_tensor(
                    query_features, expert_id, projected_features=projected_features
                )
                score_tensors.append(score)
                valid_expert_ids.append(expert_id)

        if len(score_tensors) == 0:
            return [], torch.tensor([], device=device)

        # Stack scores into a single tensor (keeps gradient graph!)
        scores = torch.stack(score_tensors)  # (num_valid_experts,)

        k = min(k, len(scores))

        # Top-k selection (indices are non-differentiable, but scores are)
        top_k_scores, top_k_indices = torch.topk(scores, k)

        # Normalize weights with softmax (differentiable!)
        normalized_weights = F.softmax(top_k_scores / self.temperature, dim=0)

        # Get expert IDs (non-differentiable, but that's fine - routing decision)
        selected_ids = [valid_expert_ids[idx.item()] for idx in top_k_indices]

        return selected_ids, normalized_weights
    
    def forward_moe(
        self,
        features: torch.Tensor,
        expert_heads: Dict[int, nn.Module],
        available_experts: Optional[List[int]] = None,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """Batched MoE forward pass with FST integration.

        Computes all expert outputs in batched mode (one batched forward per
        expert) instead of per-sample loops.  This is roughly
        (batch_size * K / num_experts) times faster.

        Gradient flow is preserved:
        - Router params (feature_projector, expert_embeddings) get gradients
          through the gating scores and softmax weights.
        - FST params get gradients through the expert output computation.
        - Top-K selection is non-differentiable (discrete), but the selected
          scores and outputs maintain their gradient graphs.
        """
        batch_size = features.size(0)
        device = features.device

        if len(expert_heads) == 0:
            return torch.zeros(batch_size, self.num_classes, device=device)

        # Filter to experts with available heads AND registered features
        if available_experts is None:
            valid_experts = [eid for eid in expert_heads.keys()
                            if self.expert_features_count[eid] > 0]
        else:
            valid_experts = [eid for eid in available_experts
                            if eid in expert_heads and self.expert_features_count[eid] > 0]

        if len(valid_experts) == 0:
            return torch.zeros(batch_size, self.num_classes, device=device)

        effective_k = min(k or self.top_k, len(valid_experts))

        # Step 1: Compute ALL expert outputs for the entire batch (batched per expert)
        expert_output_list = []
        for eid in valid_experts:
            fst = self.get_or_create_fst(eid).to(device)
            aligned = fst(features)                    # (B, feature_dim)
            logits = expert_heads[eid](aligned)        # (B, num_classes)
            expert_output_list.append(logits)
        # (B, num_valid, num_classes)
        expert_outputs = torch.stack(expert_output_list, dim=1)

        # Step 2: Compute routing scores for all samples x all experts
        scores = self._compute_scores_batched(features, valid_experts)  # (B, num_valid)

        # Step 3: Top-K selection and softmax normalization
        top_scores, top_indices = torch.topk(scores, effective_k, dim=1)  # (B, K)
        top_weights = F.softmax(top_scores / self.temperature, dim=1)     # (B, K)

        # Step 4: Gather top-K expert outputs and weighted combine
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, self.num_classes)  # (B, K, C)
        selected = torch.gather(expert_outputs, 1, idx_expanded)                  # (B, K, C)

        # Weighted sum: (B, K, C) * (B, K, 1) -> sum dim=1 -> (B, C)
        return (selected * top_weights.unsqueeze(-1)).sum(dim=1)
    
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
                nn.init.normal_(self.expert_embeddings.weight[expert_id], mean=0, std=0.1)
