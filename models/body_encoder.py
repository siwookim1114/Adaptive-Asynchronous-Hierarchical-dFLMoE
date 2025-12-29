"""
Body Encoder: Private feature extractor B_i

Body encoder is the Private part of each client's model that:
- Extracts features from raw input (images)
- Never shared with other clients
- Stays on the client's device (privacy-preserving)
- Can be heterogeneous (different architectures per client)
"""

import torch
import torch.nn as nn
import torchvision
import sys
from pathlib import Path

# Add parent directory to path to import config_loader
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config

class BodyEncoder(nn.Module):
    """
    Abstract base clas for body encoders
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim    # Store output_dimnesion value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

class SimpleCNNBody(BodyEncoder):
    """
    Simple CNN body encoder for CIFAR-10/100

    Architecture:
    - 3 conv blocks (64, 128, 256 channels)
    - Each block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
    - Global average pooling + FC layer

    Input: (B, 3, 32, 32) RGB images
    Output: (B, output_dim) feature vectors
    """
    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        super().__init__(output_dim)    
        self.conv_layers = nn.Sequential(
            # Block 1: 32 x 32 -> 16 x 16
            nn.Conv2d(input_channels, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16 x 16 -> 8 x 8
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),

            # Blcok 3: 8 x 8 -> 4 x 4
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, output_dim),   # 512
            nn.ReLU(inplace = True)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, 3, 32, 32)
        
        Returns:
            features: Feature tensor (B, output_dim)
        """
        features = self.projection(self.conv_layers(x))
        return features
    
class ResNetBody(BodyEncoder):
    """
    Resnet-18 based body encoder

    Uses pretrained Resnet-18 and removes the final FC layer.

    Input: (B, 3, 32, 32) RGB images
    Output: (B, output_dim) feature vectors
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 512, pretrained: bool = False):
        super().__init__(output_dim)

        # Load Resnet-18
        resnet = torchvision.models.resnet18(pretrained = pretrained)

        # Modify first conv if input_channels != 3
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(input_channels, 64, 7, stride = 2, padding = 3, bias = False)

        # Feature exxtract before the final fully connected layer
        ## Unpack Resnet model and only remove the final classifier head layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet-18 outputs 512-dim features
        resnet_output_dim = 512

        # Add projection layer if output_dim is different from 512
        if output_dim != resnet_output_dim:
            self.projection = nn.Linear(resnet_output_dim, output_dim)
        else:
            self.projection= None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, 3, 32 32)

        Returns:
            features: Feature tensor (B, output_dim)
        """
        features = self.features(x)
        features = features.flatten(1)

        if self.projection is not None:
            features = self.projection(features)
        
        return features

def create_body_encoder(config, device: torch.device) -> BodyEncoder:
    """
    Factory function to create body encoder from config

    Args:
        config: Configuration object with model.body settings
        device: torch.device to place model on

    Returns:
        Body encoder module
    """
    body_type = config.model.body.type.lower()
    if body_type == "simple_cnn":
        encoder = SimpleCNNBody(
            input_channels = config.model.body.input_channels,
            output_dim = config.model.head.input_dim
        )
    elif body_type == "resnet18":
        encoder = ResNetBody(
            input_channels = config.model.body.input_channels,
            output_dim = config.model.head.input_dim,
            pretrained = False
        )
    else:
        raise ValueError(f"Unknown body type: {body_type}. Supported: simple_cnn, resnet18")
    
    torch.backends.cudnn.benchmark = True
    return encoder.to(device)

def test_body_encoder():
    """Test body encoder functionality"""
    print("\n" + "="*70)
    print("TESTING BODY ENCODER")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Test 1: Simple CNN Body
    print("TEST 1: SimpleCNNBody")
    print("-" * 70)

    simple_body = SimpleCNNBody(input_channels = 3, output_dim = 512).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in simple_body.parameters())
    print(f"Simple CNN Body Created")
    print(f"Parameters: {num_params:,}")

    # Test Forward pass
    batch_size = 64
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    with torch.no_grad():
        features = simple_body(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({batch_size}, 512)")
    assert features.shape == (batch_size, 512), "Output shape mismatch!"
    print()

    # Test 2: ResNetBody
    print("TEST 2: ResNetBody")
    print("-" * 70)
    
    resnet_body = ResNetBody(input_channels=3, output_dim=512, pretrained=False).to(device)
    
    num_params = sum(p.numel() for p in resnet_body.parameters())
    print(f"✓ ResNetBody created")
    print(f"  Parameters: {num_params:,}")
    
    with torch.no_grad():
        features = resnet_body(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Expected: ({batch_size}, 512)")
    assert features.shape == (batch_size, 512), "Output shape mismatch!"
    print()

    # Test 3: create_body_encoder with config.yaml
    print("TEST 3: create_body_encoder() with actual config")
    print("-" * 70)

    config_path = project_root / "configs" / "config.yaml"
    print(f"Loading config from: {config_path}")

    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        print(f"Skipping Test 3...")
        print()
    else:
        config = load_config(str(config_path))
        print(f"Config loaded successfully")
        print(f"Body type: {config.model.body.type}")
        print(f"Input channels: {config.model.body.input_channels}")
        print(f"Head hidden dim: {config.model.head.input_dim}")

        # Create body encoder from config
        body = create_body_encoder(config, device)

        print(f"Body encoder created from config")
        print(f"Output dimension: {body.output_dim}")

        # Testing forward pass
        with torch.no_grad():
            features = body(dummy_input)
        
        print(f"Forward pass successful")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Expected: ({batch_size}, {config.model.head.input_dim})")
        assert features.shape == (batch_size, config.model.head.input_dim), "Output shape mismatch!"
        print()
    
    # Test 4: Graident flow
    print(f"TEST 4: Gradient Flow")
    print("-" * 70)
    
    simple_body.train()
    dummy_input.requires_grad = True

    features = simple_body(dummy_input)
    loss = features.sum()
    loss.backward()

    has_gradients = all(p.grad is not None for p in simple_body.parameters() if p.requires_grad)
    print(f"Backward pass sucessfully")
    print(f"All parameters have gradients: {has_gradients}")
    assert has_gradients, "Some parameters don't have gradients!"
    print()

    print("=" * 70)
    print("All Tests Passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_body_encoder()

    
    
    
