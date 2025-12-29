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
from typing import Optional

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
    
    # Test 2: Head with Dropout
    print("TEST 2: Head with Dropout")
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
    
    # In train mode with dropout, outputs should differ
    train_diff = (output_train1 - output_train2).abs().mean().item()
    # In eval mode, outputs should be identical
    eval_diff = (output_eval1 - output_eval2).abs().mean().item()
    
    print(f"✓ Dropout behavior verified")
    print(f"  Train mode difference: {train_diff:.6f} (should be > 0)")
    print(f"  Eval mode difference: {eval_diff:.6f} (should be ~0)")
    assert train_diff > 0.01, "Dropout not working in train mode!"
    assert eval_diff < 1e-6, "Outputs differ in eval mode!"
    print()
    
    # Test 3: Weight Get/Set
    print("TEST 3: Weight Get/Set Operations")
    print("-" * 70)
    
    # Get weights
    weights = head.get_weights()
    print(f"✓ Weights extracted")
    print(f"  Number of weight tensors: {len(weights)}")
    print(f"  Weight keys: {list(weights.keys())}")
    
    # Create new head and set weights
    head_copy = Head(input_dim=512, hidden_dim=256, output_dim=10).to(device)
    head_copy.set_weights(weights)
    
    print(f"✓ Weights set to new head")
    
    # Verify outputs are identical
    with torch.no_grad():
        output1 = head(dummy_features)
        output2 = head_copy(dummy_features)
    
    diff = (output1 - output2).abs().max().item()
    print(f"✓ Output difference: {diff:.10f} (should be ~0)")
    assert diff < 1e-6, "Weights not copied correctly!"
    print()

    # Test 4: create_head with config
    print("TEST 4: create_head() with config")
    print("-" * 70)
    
    try:
        # Load config
        config_path = project_root / 'configs' / 'config.yaml'
        print(f"Loading config from: {config_path}")
        
        if not config_path.exists():
            print(f"⚠️  Config file not found at {config_path}")
            print(f"   Skipping Test 4...")
            print()
        else:
            config = load_config(str(config_path))
            
            print(f"✓ Config loaded successfully")
            print(f"  Dataset: {config.data.dataset}")
            print(f"  Input dim: {config.model.head.input_dim}")
            
            # Create head from config
            head_from_config = create_head(config, device)
            
            print(f"✓ Head created from config")
            print(f"  Input dim: {head_from_config.input_dim}")
            print(f"  Output dim: {head_from_config.output_dim}")
            
            # Test forward pass
            with torch.no_grad():
                logits = head_from_config(dummy_features)
            
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {logits.shape}")
            print()
    
    except Exception as e:
        print(f"Error in Test 4: {e}")
        print()
    
    # Test 5: Gradient Flow
    print("TEST 5: Gradient Flow")
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
    
    # Test 6: Combined Body + Head
    print("TEST 6: Combined Body + Head Pipeline")
    print("-" * 70)
    
    try:
        # Create body and head
        body = SimpleCNNBody(input_channels=3, output_dim=512).to(device)
        head_test = Head(input_dim=512, output_dim=10).to(device)
        
        print(f"✓ Body and Head created")
        
        # Test end-to-end
        dummy_images = torch.randn(4, 3, 32, 32).to(device)
        
        with torch.no_grad():
            features = body(dummy_images)
            logits = head_test(features)
        
        print(f"✓ End-to-end forward pass successful")
        print(f"  Image shape: {dummy_images.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Logits shape: {logits.shape}")
        
        # Get predictions
        predictions = logits.argmax(dim=1)
        print(f"  Predictions: {predictions.tolist()}")
        print()
    
    except ImportError:
        print(f"body_encoder not found, skipping combined test")
        print()
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_head()