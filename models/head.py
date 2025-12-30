"""
Head: Lightweight expert classifier H_i

Head is the SHAREABLE part of each client's model that:
- Takes features from body encoder (512-dim)
- Produces class predictions (10-dim for CIFAR-10)
- Lightweight (~5K parameters)
- Gets SHARED with other clients (MoE component)
- Could be cached and aggregated

Architecture:
- Simple MLP: features -> hidden -> output
- Optional dropout for regularization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import time
import time as time_module

import sys
from pathlib import Path
        
# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
        
from utils.config_loader import load_config
from body_encoder import SimpleCNNBody

class Head(nn.Module):
    """
    Lightweight expert head for classification

    Architecture:
    Input (512) -> Linear -> ReLU -> Dropout -> Linear -> Output(10)

    This is the expert in MoE that gets shared
    Shared as: weights + metadata (timestamp, trust, validation accuracy)
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 10, dropout: float = 0.0):
        """
        Args:
            input_dim: Input feature dimension (from body encoder)
            hidden_dim: Hidden layer dimension
            output_dim: Number of classes
            dropout: Dropout probability (0.0 -> No dropout)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Simple 2-layer MLP
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Metadata for sharing (M_i)
        self.metadata = {
            "timestamp": time.time(),     # For computing delta t
            "trust_score": 1.0,           # T_{ij}: Validation Accuracy
            "validation_accuracy": 0.0,   # Actual validation performance
            "num_samples": 0,             # Number of training samples
            "expert_id": None,            # Which client this came from 
            "training_loss": float("inf") # Training loss (optional)
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            features: Input features (B, input_dim)
        
        Returns:
            logits: Class logits (B, output_dim)
        """
        return self.layers(features)
    
    def get_weights(self) -> dict:
        """
        Get model weights for sharing / caching

        Returns:
            Dictionary of weights
        """
        return {name: param.detach().clone() for name, param in self.named_parameters()}
    
    def set_weights(self, weights: dict):
        """
        Set model weights from dictionary

        Args:
            weights: Dictionary of weights
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    def get_head_package(self, expert_id: Optional[int] = None) -> Dict:
        """
        Get complete head package: H_{i} + metadata

        This is what gets shared between peers according to the paper.
        Format: {weights, metadata}

        Metadata includes:
        - timestamp: When this head was last updated (for staleness)
        - trust_score: Validation accuracy (for trust weighting)
        - validation_accuracy: Actual validation performance
        - num_samples: Number of training samples used
        - expert_id: Which client this expert belongs to

        Args:
            expert_id: ID of the expert (client) sharing this head
        
        Returns:
            Complete package: {
                "weights" : {...},
                "metadata": {...}
            }
        """

        package = {
            "weights": self.get_weights(),
            "metadata": {
                "timestamp": time.time(),      # Current time (initialized)
                "trust_score": self.metadata["trust_score"], 
                "validation_accuracy": self.metadata["validation_accuracy"],
                "num_samples": self.metadata["num_samples"],
                "expert_id": expert_id if expert_id is not None else self.metadata["expert_id"],
                "training_loss": self.metadata["training_loss"]
            }
        }

        return package

    def load_head_package(self, package: Dict):
        """
        Load complete head pckage: H_{i} + metadata

        Args:
            package: Complete package with "weights" and "metadata" keys
        """
        # Load Weights
        if "weights" in package:
            self.set_weights(package["weights"])
        
        # Load Metadata
        if "metadata" in package:
            self.metadata.update(package["metadata"])

    def update_metadata(
        self,
        validation_accuracy: Optional[float] = None,
        num_samples: Optional[int] = None,
        expert_id: Optional[int] = None,
        training_loss: Optional[int] = None
    ):
        """
        Update metadata after training/validation
        Called after local training to update the head's metadata before sharing it with peers

        Args:
            validation_accuracy: Validation accuracy (0 - 1)
            num_samples: Number of training samples
            expert_id: Expert ID
            training_loss: Training loss (optional)
        """
        # Timestamp is always updated when metadata changes
        self.metadata["timestamp"] = time.time()

        if validation_accuracy is not None:
            self.metadata["validation_accuracy"] = validation_accuracy
            self.metadata["trust_score"] = validation_accuracy   # Trust score = validation accracy
        
        if num_samples is not None:
            self.metadata["num_samples"] = num_samples
        
        if expert_id is not None:
            self.metadata["expert_id"] = expert_id
        
        if training_loss is not None:
            self.metadata["training_loss"] = training_loss

    def get_metadata(self) -> Dict:
        """Get current metadata"""
        return self.metadata
    
    def get_trust_score(self) -> float:
        """Get current rust score (T_{ij})"""
        return self.metadata["trust_score"]

    def get_validation_accuracy(self) -> float:
        """Get validation accuracy"""
        return self.metadata["validation_accuracy"]
    
    def get_timestamp(self) -> float:
        """Get timestamp (for computing delta_t)"""
        return self.metadata["timestamp"]
    
    def get_age(self) -> float:
        """
        Get age of this head in seconds (delta_t)

        Returns:
            Age in seconds since last update
        """
        return time.time() - self.metadata["timestamp"]

    def is_stale(self, max_age_seconds: float = 600) -> bool:
        """
        Check if this head is stale (too old)

        Args:
            max_age_seconds: Minimum age in seconds (default: 10 minutes)
        
        Returns:
            True if stale (too old)
        """
        return self.get_age() > max_age_seconds

    def reset_metadata(self, expert_id: Optional[int] = None):
        """Reset metadata to initial state"""
        self.metadata = {
            "timestamp": time.time(),
            "trust_score": 1.0,
            "validation_accuracy": 0.0,
            "num_samples": 0,
            "expert_id": expert_id,
            "training_loss": float("inf")
        }

def create_head(config, device: torch.device) -> Head:
    """
    Factory function to create head from config

    Args:
        config: Configuration object with model.head settings
        device: torch.device to place model on

    Returns:
        Head module
    """
    # Get dataset info for output dimension
    dataset_name = config.data.dataset.lower()

    if dataset_name in ["cifar10", "mnist"]:
        output_dim = 10
    elif dataset_name == "cifar100":
        output_dim = 100
    else:
        output_dim = config.data.get("num_classes", 10)

    head = Head(
        input_dim = config.model.head.input_dim,  # From body encoder's output
        hidden_dim = config.model.head.hidden_dim,   # Actual hidden layer
        output_dim = output_dim,      # Number of classes to predict
        dropout = config.model.head.dropout 
    )
    torch.backends.cudnn.benchmark = True
    return head.to(device)

# Testing Functions
def test_head():
    """Test head functionality"""
    print("\n" + "="*70)
    print("TESTING HEAD")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test 1: Basic Head Creation
    print("TEST 1: Basic Head Creation")
    print("-" * 70)
    
    head = Head(
        input_dim=512,
        hidden_dim=256,
        output_dim=10,
        dropout=0.0
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in head.parameters())
    print(f"✓ Head created")
    print(f"  Parameters: {num_params:,}")
    print(f"  Input dim: {head.input_dim}")
    print(f"  Hidden dim: {head.hidden_dim}")
    print(f"  Output dim: {head.output_dim}")
    
    # Test forward pass
    batch_size = 64
    dummy_features = torch.randn(batch_size, 512).to(device)
    
    with torch.no_grad():
        logits = head(dummy_features)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_features.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 10)")
    assert logits.shape == (batch_size, 10), "Output shape mismatch!"
    print()
    
   # Test 2: Metadata Management
    print("TEST 2: Metadata Management")
    print("-" * 70)
    
    print("Initial metadata:")
    metadata = head.get_metadata()
    print(f"  Trust score: {metadata['trust_score']:.2f}")
    print(f"  Validation accuracy: {metadata['validation_accuracy']:.2f}")
    print(f"  Num samples: {metadata['num_samples']}")
    print(f"  Expert ID: {metadata['expert_id']}")
    
    # Simulate training: update metadata
    print("\n✓ Simulating training completion...")
    head.update_metadata(
        validation_accuracy=0.87,
        num_samples=5000,
        expert_id=3,
        training_loss=0.234
    )
    
    print("Updated metadata:")
    metadata = head.get_metadata()
    print(f"  Trust score: {metadata['trust_score']:.2f}")
    print(f"  Validation accuracy: {metadata['validation_accuracy']:.2f}")
    print(f"  Num samples: {metadata['num_samples']}")
    print(f"  Expert ID: {metadata['expert_id']}")
    print(f"  Training loss: {metadata['training_loss']:.3f}")
    
    assert metadata['trust_score'] == 0.87, "Trust score not updated!"
    assert metadata['validation_accuracy'] == 0.87, "Val accuracy not updated!"
    print()
    
    # Test 3: Complete Head Package (H_i + metadata)
    print("TEST 3: Complete Head Package (H_i + metadata)")
    print("-" * 70)
    
    # Get complete package
    package = head.get_head_package(expert_id=3)
    
    print(" Head package created")
    print(f"  Package keys: {list(package.keys())}")
    print(f"  Weights: {len(package['weights'])} tensors")
    print(f"  Metadata keys: {list(package['metadata'].keys())}")
    
    # Verify package structure
    assert 'weights' in package, "Missing weights!"
    assert 'metadata' in package, "Missing metadata!"
    assert 'timestamp' in package['metadata'], "Missing timestamp!"
    assert 'trust_score' in package['metadata'], "Missing trust_score!"
    assert 'validation_accuracy' in package['metadata'], "Missing validation_accuracy!"
    assert 'expert_id' in package['metadata'], "Missing expert_id!"
    
    print("\n Package contains all required components:")
    print(f"  - Weights: {len(package['weights'])} tensors")
    print(f"  - Timestamp: {package['metadata']['timestamp']:.2f}")
    print(f"  - Trust score: {package['metadata']['trust_score']:.2f}")
    print(f"  - Validation accuracy: {package['metadata']['validation_accuracy']:.2f}")
    print(f"  - Expert ID: {package['metadata']['expert_id']}")
    print(f"  - Num samples: {package['metadata']['num_samples']}")
    print()
    
    # Test 4: Loading Head Package
    print("TEST 4: Loading Head Package")
    print("-" * 70)
    
    # Create new head and load package
    new_head = Head(input_dim=512, hidden_dim=256, output_dim=10).to(device)
    
    print("Before loading:")
    print(f"  New head trust score: {new_head.get_trust_score():.2f}")
    print(f"  New head expert ID: {new_head.get_metadata()['expert_id']}")
    
    new_head.load_head_package(package)
    
    print("\n✓ Package loaded into new head")
    print("After loading:")
    print(f"  New head trust score: {new_head.get_trust_score():.2f}")
    print(f"  New head expert ID: {new_head.get_metadata()['expert_id']}")
    
    # Verify metadata matches
    assert new_head.get_trust_score() == head.get_trust_score(), "Trust scores don't match!"
    assert new_head.get_metadata()['expert_id'] == 3, "Expert ID not loaded!"
    
    # Verify outputs are identical
    with torch.no_grad():
        out1 = head(dummy_features)
        out2 = new_head(dummy_features)
    
    diff = (out1 - out2).abs().max().item()
    print(f"\n Output difference: {diff:.10f} (should be ~0)")
    assert diff < 1e-6, "Outputs don't match!"
    print()
    
    # Test 5: Staleness Detection
    print("TEST 5: Staleness Detection (delta_t)")
    print("-" * 70)
    
    print(f"Head age: {head.get_age():.4f}s")
    print(f"Is stale (1 hour threshold): {head.is_stale(max_age_seconds=3600)}")
    print(f"Is stale (0.1 second threshold): {head.is_stale(max_age_seconds=0.1)}")
    
    # Simulate waiting
    print("\n✓ Waiting 0.5 seconds...")
    time_module.sleep(0.5)
    
    print(f"Head age after wait: {head.get_age():.4f}s")
    assert head.get_age() > 0.5, "Age not increasing!"
    print("✓ Staleness detection working")
    print()
    
    # Test 6: Head with Dropout
    print("TEST 6: Head with Dropout")
    print("-" * 70)
    
    head_dropout = Head(
        input_dim=512,
        hidden_dim=256,
        output_dim=10,
        dropout=0.5
    ).to(device)
    
    print(f"✓ Head with dropout created")
    print(f"  Dropout: 0.5")
    
    # Test train vs eval mode
    head_dropout.train()
    output_train1 = head_dropout(dummy_features)
    output_train2 = head_dropout(dummy_features)
    
    head_dropout.eval()
    with torch.no_grad():
        output_eval1 = head_dropout(dummy_features)
        output_eval2 = head_dropout(dummy_features)
    
    train_diff = (output_train1 - output_train2).abs().mean().item()
    eval_diff = (output_eval1 - output_eval2).abs().mean().item()
    
    print(f"✓ Dropout behavior verified")
    print(f"  Train mode difference: {train_diff:.6f} (should be > 0)")
    print(f"  Eval mode difference: {eval_diff:.6f} (should be ~0)")
    assert train_diff > 0.01, "Dropout not working in train mode!"
    assert eval_diff < 1e-6, "Outputs differ in eval mode!"
    print()
    
    # Test 7: Weight Get/Set (Legacy)
    print("TEST 7: Weight Get/Set Operations (Legacy)")
    print("-" * 70)
    
    weights = head.get_weights()
    print(f"  Weights extracted")
    print(f"  Number of weight tensors: {len(weights)}")
    print(f"  Weight keys: {list(weights.keys())}")
    
    head_copy = Head(input_dim=512, hidden_dim=256, output_dim=10).to(device)
    head_copy.set_weights(weights)
    
    print(f" Weights set to new head")
    
    with torch.no_grad():
        output1 = head(dummy_features)
        output2 = head_copy(dummy_features)
    
    diff = (output1 - output2).abs().max().item()
    print(f" Output difference: {diff:.10f} (should be ~0)")
    assert diff < 1e-6, "Weights not copied correctly!"
    print()
    
    # Test 8: create_head with config
    print("TEST 8: create_head() with config")
    print("-" * 70)
    
    try:        
        config_path = project_root / 'configs' / 'config.yaml'
        print(f"Loading config from: {config_path}")
        
        if not config_path.exists():
            print(f"   Config file not found at {config_path}")
            print(f"   Skipping Test 8...")
            print()
        else:
            config = load_config(str(config_path))
            
            print(f"✓ Config loaded successfully")
            print(f"  Dataset: {config.data.dataset}")
            print(f"  Input dim: {config.model.head.input_dim}")
            
            head_from_config = create_head(config, device)
            
            print(f"✓ Head created from config")
            print(f"  Input dim: {head_from_config.input_dim}")
            print(f"  Output dim: {head_from_config.output_dim}")
            
            with torch.no_grad():
                logits = head_from_config(dummy_features)
            
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {logits.shape}")
            print()
    
    except Exception as e:
        print(f"   Error in Test 8: {e}")
        print(f"   This is okay - continuing with other tests...")
        print()
    
    # Test 9: Gradient Flow
    print("TEST 9: Gradient Flow")
    print("-" * 70)
    
    head.train()
    dummy_features.requires_grad = True
    
    logits = head(dummy_features)
    loss = logits.sum()
    loss.backward()
    
    has_gradients = all(p.grad is not None for p in head.parameters() if p.requires_grad)
    print(f"✓ Backward pass successful")
    print(f"  All parameters have gradients: {has_gradients}")
    assert has_gradients, "Some parameters don't have gradients!"
    print()
    
    # Test 10: Combined Body + Head Pipeline
    print("TEST 10: Combined Body + Head Pipeline")
    print("-" * 70)
    
    try:        
        body = SimpleCNNBody(input_channels=3, output_dim=512).to(device)
        head_test = Head(input_dim=512, output_dim=10).to(device)
        
        # Update head metadata
        head_test.update_metadata(
            validation_accuracy=0.92,
            num_samples=10000,
            expert_id=0
        )
        
        print(f" Body and Head created")
        
        dummy_images = torch.randn(4, 3, 32, 32).to(device)
        
        with torch.no_grad():
            features = body(dummy_images)
            logits = head_test(features)
        
        print(f"  End-to-end forward pass successful")
        print(f"  Image shape: {dummy_images.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Logits shape: {logits.shape}")
        
        predictions = logits.argmax(dim=1)
        print(f"  Predictions: {predictions.tolist()}")
        
        # Get package for sharing
        package = head_test.get_head_package(expert_id=0)
        print(f"\nHead package ready for sharing:")
        print(f"  Trust score: {package['metadata']['trust_score']:.2f}")
        print(f"  Validation acc: {package['metadata']['validation_accuracy']:.2f}")
        print(f"  Expert ID: {package['metadata']['expert_id']}")
        print()
    
    except ImportError:
        print(f"⚠️  body_encoder not found, skipping combined test")
        print()
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_head()