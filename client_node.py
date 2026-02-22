"""
Client Node: Federated Learning with Dynamic MoE Routing

Complete federated learning client with:
- Local head included in MoE expert pool (router decides all weights)
- Hybrid loss for gradient routing: L_full = α*L_local + (1-α)*L_MoE
  - α controls gradient balance only (L_local trains body, L_moe trains router)
  - Prediction is purely MoE output (fully dynamic per-sample routing)
- Router integration (register_expert on receive + local self-registration)
- Hierarchical communication
- Trust computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from typing import Dict, List, Optional, Tuple

from infra.peer_cache import PeerCache, ExpertPackage
from infra.transport import TCPTransport, Message
from infra.cluster import ClusterManager
from models.router import Router
from models.head import Head


class ClientNode:
    """FINAL Federated Learning Client Node"""
    
    def __init__(
        self,
        client_id: str,
        cluster_manager: ClusterManager,
        host: str = '0.0.0.0',
        port: int = 0,
        feature_dim: int = 512,
        num_classes: int = 10,
        num_experts: int = 10,
        max_cache_size: int = 50,
        staleness_lambda: float = 0.001,
        top_k_experts: int = 3,
        alpha: float = 0.5,
        warmup_rounds: int = 10,
        local_epochs: int = 5,
        learning_rate_head: float = 0.001,
        learning_rate_body: float = 0.001,
        learning_rate_router: float = 0.001,
        weight_decay: float = 1e-4,
        lr_decay: float = 0.98
    ):
        self.client_id = client_id
        self.local_expert_id = int(client_id.split('_')[-1]) if '_' in client_id else int(client_id)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.alpha = alpha
        
        self.lr_head = learning_rate_head
        self.lr_body = learning_rate_body
        self.lr_router = learning_rate_router
        self.warmup_rounds = warmup_rounds
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.dropout = 0.0  # Set by main.py via head creation; stored for eval
        
        # Components
        self.cache = PeerCache(max_cache_size=max_cache_size, staleness_decay=staleness_lambda)
        self.transport = TCPTransport(client_id=client_id, host=host, port=port)
        self.cluster_manager = cluster_manager
        self.router = Router(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k_experts,
            staleness_lambda=staleness_lambda,
            use_learned_gating=True
        )
        
        # Model components
        self.body_encoder = None
        self.head = None
        
        # Optimizers
        self.optimizer_head = None
        self.optimizer_body = None
        self.optimizer_router = None
        
        # Training state
        self.trust_score = 0.5
        self.validation_accuracy = 0.0
        self.num_samples = 0
        self.representative_features = torch.randn(feature_dim)
        
        # Communication settings
        self.cluster_exchange_interval = 1
        self.cross_cluster_exchange_interval = 5
        self.current_round = 0
        
        # Track which FST keys have been added to optimizer_router
        # (FSTs created after optimizer init must be explicitly added)
        self._registered_fst_keys = set()

        # Statistics
        self._stats = {
            'rounds_completed': 0,
            'experts_sent': 0,
            'experts_received': 0,
            'experts_registered': 0,
            'experts_relayed': 0,
            'experts_used': 0,
            'trust_updates': 0,
            'loss_local': [],
            'loss_moe': [],
            'loss_total': []
        }
        self._stats_lock = threading.Lock()
        
        # Setup
        self.transport.register_handler('expert_package', self._handle_expert_package)
        self.cluster_manager.register_client(
            client_id=self.client_id,
            features=self.representative_features,
            trust_score=self.trust_score
        )
    
    def _ensure_fsts_in_optimizer(self):
        """Add any newly created FST parameters to the router optimizer.

        FSTs are created lazily by router.get_or_create_fst() when experts
        register. Since the optimizer is created before any FSTs exist,
        new FST parameters must be explicitly added via add_param_group.
        Without this, FSTs stay at identity init and never learn.
        """
        if self.optimizer_router is None:
            return
        for key, fst in self.router.fst_transforms.items():
            if key not in self._registered_fst_keys:
                self.optimizer_router.add_param_group({
                    'params': list(fst.parameters()),
                    'lr': self.lr_router,
                    'weight_decay': self.weight_decay
                })
                self._registered_fst_keys.add(key)

    def set_model(self, body_encoder: nn.Module, head: nn.Module):
        self.body_encoder = body_encoder
        self.head = head
        # Store dropout from head for use when creating temp heads in evaluation
        if hasattr(head, 'layers') and len(head.layers) > 2:
            dropout_layer = head.layers[2]
            if isinstance(dropout_layer, nn.Dropout):
                self.dropout = dropout_layer.p

        # Move router to same device as body encoder
        device = next(body_encoder.parameters()).device
        self.router = self.router.to(device)

        self.optimizer_head = optim.Adam(self.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)
        self.optimizer_body = optim.Adam(self.body_encoder.parameters(), lr=self.lr_body, weight_decay=self.weight_decay)
        self.optimizer_router = optim.Adam(self.router.parameters(), lr=self.lr_router, weight_decay=self.weight_decay)

        # Register local head as expert so router can score it
        self._register_local_expert()
        # Add the local expert's FST (just created) to the optimizer
        self._ensure_fsts_in_optimizer()

    def _register_local_expert(self):
        """Register this client's own head as an expert in the router.

        This allows the router to score the local head alongside remote experts
        using the same Trust x Similarity x Staleness x Gate mechanism.
        """
        self.router.register_expert(
            expert_id=self.local_expert_id,
            trust_score=self.trust_score,
            validation_accuracy=self.validation_accuracy,
            features=self.representative_features,
            num_samples=self.num_samples,
            timestamp=time.time()
        )
    
    def _relay_if_head(self, package):
        """Relay cross-cluster expert packages to cluster members.

        Completes the hierarchical top-down dissemination:
          Bottom-up:  members share with peers → head receives (intra-cluster)
          Head-to-head: heads exchange cross-cluster (every N rounds)
          Top-down:   head relays cross-cluster experts to members  ← THIS

        Only cluster heads relay, and only for experts originating from
        a different cluster. Members never relay, so no infinite loops.
        """
        if not self.cluster_manager.is_cluster_head(self.client_id):
            return

        expert_origin = package.client_id
        expert_cluster = self.cluster_manager.get_cluster_id(expert_origin)
        my_cluster = self.cluster_manager.get_cluster_id(self.client_id)

        if expert_cluster is None or my_cluster is None:
            return
        if expert_cluster == my_cluster:
            return  # Same cluster — members already have this via intra-cluster

        peers = self.cluster_manager.get_cluster_peers(self.client_id)
        if peers:
            num_relayed = self.transport.broadcast(peers, 'expert_package', package)
            with self._stats_lock:
                self._stats['experts_relayed'] += num_relayed

    def _handle_expert_package(self, message: Message):
        """CRITICAL: Registers expert with router, relays if cluster head"""
        package = message.payload
        success = self.cache.add(package)

        if success:
            try:
                expert_id = int(package.client_id.split('_')[-1]) if '_' in package.client_id else int(package.client_id)

                self.router.register_expert(
                    expert_id=expert_id,
                    trust_score=package.trust_score,
                    validation_accuracy=package.validation_accuracy,
                    features=package.representative_features,
                    num_samples=package.num_samples,
                    timestamp=package.timestamp
                )
                # NOTE: FST params are added to optimizer at the start of
                # _train_epoch_corrected(), NOT here. This handler runs in the
                # transport thread — calling optimizer methods here would race
                # with the training thread.

                with self._stats_lock:
                    self._stats['experts_registered'] += 1
            except Exception as e:
                print(f"[{self.client_id}] Registration failed: {e}")

            with self._stats_lock:
                self._stats['experts_received'] += 1

            # Head relay: forward cross-cluster experts to cluster members
            self._relay_if_head(package)
    
    def compute_representative_features(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        if self.body_encoder is None:
            return torch.randn(self.feature_dim)
        
        self.body_encoder.eval()
        features_list = []
        
        with torch.no_grad():
            for x, _ in data_loader:
                device = next(self.body_encoder.parameters()).device
                x = x.to(device)
                features = self.body_encoder(x)
                features_list.append(features.cpu())
        
        if len(features_list) > 0:
            all_features = torch.cat(features_list, dim=0)
            representative = all_features.mean(dim=0)
            self.representative_features = representative
            return representative
        
        return self.representative_features
    
    def compute_trust_score(self, validation_accuracy: float, num_samples: int):
        accuracy_component = validation_accuracy
        data_component = min(1.0, num_samples / 10000.0)
        self.trust_score = 0.7 * accuracy_component + 0.3 * data_component
        self.trust_score = max(0.0, min(1.0, self.trust_score))
        self.validation_accuracy = validation_accuracy
        self.num_samples = num_samples
        
        with self._stats_lock:
            self._stats['trust_updates'] += 1
    
    def _get_current_alpha(self) -> float:
        """Adaptive alpha with warm-up schedule.

        Linearly decays from 1.0 (pure local) to target alpha over
        warmup_rounds. This lets body/head stabilize before MoE
        contributes significant gradient signal.
        """
        if self.warmup_rounds <= 0 or self.current_round >= self.warmup_rounds:
            return self.alpha
        progress = self.current_round / self.warmup_rounds
        return 1.0 - (1.0 - self.alpha) * progress

    def create_expert_package(self) -> ExpertPackage:
        if self.head is None:
            raise ValueError("Head not set")
        
        return ExpertPackage(
            client_id=self.client_id,
            head_state_dict=self.head.state_dict(),
            timestamp=time.time(),
            trust_score=self.trust_score,
            validation_accuracy=self.validation_accuracy,
            representative_features=self.representative_features.clone(),
            num_samples=self.num_samples
        )
    
    def share_expert(self):
        package = self.create_expert_package()
        targets = self.cluster_manager.get_communication_targets(self.client_id)
        experts_sent = 0
        
        if self.current_round % self.cluster_exchange_interval == 0:
            cluster_peers = targets['cluster_peers']
            if len(cluster_peers) > 0:
                num_sent = self.transport.broadcast(cluster_peers, 'expert_package', package)
                experts_sent += num_sent
        
        if self.current_round % self.cross_cluster_exchange_interval == 0:
            if self.cluster_manager.is_cluster_head(self.client_id):
                cluster_heads = targets['cluster_heads']
                if len(cluster_heads) > 0:
                    num_sent = self.transport.broadcast(cluster_heads, 'expert_package', package)
                    experts_sent += num_sent
        
        with self._stats_lock:
            self._stats['experts_sent'] += experts_sent
    
    def train_round(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_experts: bool = True
    ) -> Dict:
        if self.body_encoder is None or self.head is None:
            raise ValueError("Model not set")
        
        self.current_round += 1

        # Multiple local epochs per round for better body/head convergence.
        # Each epoch creates a fresh frozen copy of the head for the MoE pool.
        for _epoch in range(self.local_epochs):
            train_metrics = self._train_epoch_corrected(train_loader, criterion, device, use_experts)

        val_loss, val_acc = self._validate(val_loader, criterion, device, use_experts)
        
        self.compute_trust_score(val_acc, len(train_loader.dataset))
        self.compute_representative_features(train_loader)
        
        self.cluster_manager.update_client(
            client_id=self.client_id,
            features=self.representative_features,
            trust_score=self.trust_score
        )

        # Update local expert registration with latest trust/features
        self._register_local_expert()

        # Note: Re-clustering is now handled at the orchestration level (main.py)
        # to avoid excessive re-clustering (previously triggered per-client per-round)

        self.share_expert()

        # LR decay: reduce learning rates each round to stabilize later training
        if self.lr_decay < 1.0:
            for optimizer in [self.optimizer_head, self.optimizer_body, self.optimizer_router]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.lr_decay

        with self._stats_lock:
            self._stats['rounds_completed'] += 1

        return {
            'round': self.current_round,
            'train_loss': train_metrics['loss_total'],
            'train_loss_local': train_metrics['loss_local'],
            'train_loss_moe': train_metrics['loss_moe'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'trust_score': self.trust_score,
            'cache_size': self.cache.size(),
            'experts_used': train_metrics['experts_used'],
            'current_alpha': self._get_current_alpha()
        }
    
    def _train_epoch_corrected(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_experts: bool
    ) -> Dict:
        """Training with hybrid loss and gradient routing"""
        # Ensure any FSTs created since last epoch are in the optimizer
        self._ensure_fsts_in_optimizer()

        self.body_encoder.train()
        self.head.train()
        self.router.train()
        
        total_loss_local = 0.0
        total_loss_moe = 0.0
        total_loss_full = 0.0
        correct = 0
        total = 0
        experts_used_count = 0
        
        # Pre-build expert heads once per epoch (they don't change within an epoch)
        expert_heads = {}
        if use_experts and self.cache.size() > 0:
            expert_packages = self.cache.get_available_experts(exclude_id=self.client_id)
            for pkg in expert_packages:
                eid = int(pkg.client_id.split('_')[-1]) if '_' in pkg.client_id else int(pkg.client_id)
                head_temp = Head(input_dim=self.feature_dim, output_dim=self.num_classes, dropout=self.dropout)
                head_temp.load_state_dict(pkg.head_state_dict)
                head_temp = head_temp.to(device)
                head_temp.eval()
                expert_heads[eid] = head_temp

        # Include a FROZEN COPY of local head in MoE expert pool.
        # Using the live self.head would cause L_moe gradients to conflict
        # with L_local gradients on the same head, degrading training.
        # The copy is treated identically to remote expert heads.
        if use_experts:
            local_head_copy = Head(input_dim=self.feature_dim, output_dim=self.num_classes, dropout=self.dropout)
            local_head_copy.load_state_dict(self.head.state_dict())
            local_head_copy = local_head_copy.to(device)
            local_head_copy.eval()
            expert_heads[self.local_expert_id] = local_head_copy

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            features = self.body_encoder(x)
            local_logits = self.head(features)
            loss_local = criterion(local_logits, y)

            if use_experts and len(expert_heads) > 0:
                # Detach features for MoE path: prevents expert predictions from
                # corrupting body encoder via bad gradients from cross-class experts.
                moe_logits = self.router.forward_moe(features.detach(), expert_heads, k=self.top_k_experts)
                loss_moe = criterion(moe_logits, y)

                # Adaptive α with warm-up: starts high (≈1.0) for stable
                # body/head training, decays to target as features mature.
                # L_local trains body+head, L_moe trains router+FST.
                current_alpha = self._get_current_alpha()
                loss_full = current_alpha * loss_local + (1 - current_alpha) * loss_moe

                self.optimizer_head.zero_grad()
                self.optimizer_body.zero_grad()
                self.optimizer_router.zero_grad()

                loss_full.backward()

                # Gradient clipping on router: preserves gradient DIRECTION
                # (router learns which experts are relatively better) while
                # bounding MAGNITUDE (prevents instability from high MoE loss).
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)

                self.optimizer_head.step()
                self.optimizer_body.step()
                self.optimizer_router.step()

                experts_used_count += len(expert_heads)
                # Prediction is purely MoE — router decides all weights dynamically
                predictions = moe_logits
            else:
                loss_moe = torch.tensor(0.0)
                loss_full = loss_local
                self.optimizer_head.zero_grad()
                self.optimizer_body.zero_grad()
                loss_full.backward()
                self.optimizer_head.step()
                self.optimizer_body.step()
                predictions = local_logits
            
            total_loss_local += loss_local.item()
            total_loss_moe += loss_moe.item() if isinstance(loss_moe, torch.Tensor) else 0.0
            total_loss_full += loss_full.item()
            
            _, predicted = predictions.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        avg_loss_local = total_loss_local / len(data_loader)
        avg_loss_moe = total_loss_moe / len(data_loader)
        avg_loss_full = total_loss_full / len(data_loader)
        accuracy = correct / total
        
        with self._stats_lock:
            self._stats['loss_local'].append(avg_loss_local)
            self._stats['loss_moe'].append(avg_loss_moe)
            self._stats['loss_total'].append(avg_loss_full)
            self._stats['experts_used'] += experts_used_count
        
        return {
            'loss_local': avg_loss_local,
            'loss_moe': avg_loss_moe,
            'loss_total': avg_loss_full,
            'accuracy': accuracy,
            'experts_used': experts_used_count
        }
    
    def _validate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        use_experts: bool
    ) -> Tuple[float, float]:
        self.body_encoder.eval()
        self.head.eval()
        self.router.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Pre-build expert heads once for validation
        val_expert_heads = {}
        if use_experts and self.cache.size() > 0:
            expert_packages = self.cache.get_available_experts(exclude_id=self.client_id)
            for pkg in expert_packages:
                eid = int(pkg.client_id.split('_')[-1]) if '_' in pkg.client_id else int(pkg.client_id)
                head_temp = Head(input_dim=self.feature_dim, output_dim=self.num_classes, dropout=self.dropout)
                head_temp.load_state_dict(pkg.head_state_dict)
                head_temp = head_temp.to(device)
                head_temp.eval()
                val_expert_heads[eid] = head_temp

        # Include local head in expert pool (already in eval mode from above)
        if use_experts:
            val_expert_heads[self.local_expert_id] = self.head

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                features = self.body_encoder(x)

                if use_experts and len(val_expert_heads) > 0:
                    local_logits = self.head(features)
                    moe_logits = self.router.forward_moe(features, val_expert_heads, k=self.top_k_experts)

                    # Hybrid loss for monitoring (comparable to training loss)
                    loss_local = criterion(local_logits, y)
                    loss_moe = criterion(moe_logits, y)
                    current_alpha = self._get_current_alpha()
                    loss = current_alpha * loss_local + (1 - current_alpha) * loss_moe

                    # Accuracy based on MoE output (what inference uses)
                    predictions = moe_logits
                else:
                    predictions = self.head(features)
                    loss = criterion(predictions, y)

                total_loss += loss.item()
                _, predicted = predictions.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        return total_loss / len(data_loader), correct / total
    
    def get_statistics(self) -> Dict:
        """
        Get client statistics

        Returns:
            Dictionary with client statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()

        stats['cache'] = {
            'size': self.cache.size(),
            'stats': self.cache.get_statistics()
        }
        stats['transport'] = self.transport.get_statistics()
        stats['router'] = {
            'num_experts_registered': len([
                eid for eid in range(self.num_experts)
                if self.router.expert_features_count[eid] > 0
            ])
        }

        return stats

    def shutdown(self):
        self.transport.shutdown()
        self.cluster_manager.remove_client(self.client_id)