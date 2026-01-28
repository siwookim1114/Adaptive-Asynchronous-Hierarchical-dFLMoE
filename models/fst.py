import torch
import torch.nn as nn

class FeatureSpaceTransform(nn.Module):
    """
    Feature Space Trnasform (FST_{i -> j})

    Aligns features from client i's space to client j's space.
    For heterogeneous federated learning with non-IID data
    """
    def __init__(self, feature_dim: int):
        """
        Initialize FST

        Initialize the weight as an identity matrix and bias as zeros so that:
        1. If two clients have similar data, FST stays near-identity (no transformation needed)
        2. If they have different data, FST gradually learns to transform
        3. It starts safe and adapts as needed

        Args:
            feature_dim: Dimension of features
        """
        super().__init__()
        # Learnable linear transformation
        ## Matrix multiplication that can rotate/scale/shift the features into a different space.
        self.transform = nn.Linear(feature_dim, feature_dim)
        # Initialize to near-identity
        nn.init.eye_(self.transform.weight)    
        nn.init.zeros_(self.transform.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Transform features

        Applies the linear transformation. Input shape (batch, 512) -> Output shape (batch, 512)

        Args:
            features: Input features (batch_size, feature_dim)

        Returns:
            Transformed features (batch_size, feature_dim)
        """
        return self.transform(features)
    
