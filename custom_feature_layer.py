import torch
import torch.nn as nn

class CustomFeatureProjector(nn.Module):
    def __init__(self, custom_feature_dim: int, target_embedding_dim: int):
        super().__init__()
        self.custom_feature_dim = custom_feature_dim
        self.target_embedding_dim = target_embedding_dim

        # A simple linear layer to project the custom feature
        self.projection_layer = nn.Linear(custom_feature_dim, target_embedding_dim)

        # Optional: Add an activation function if desired
        # self.activation = nn.ReLU() # or nn.Tanh(), etc.

        # Optional: Add LayerNorm if you find it helps stabilize training
        # self.layer_norm = nn.LayerNorm(target_embedding_dim)

    def forward(self, custom_feature_vector: torch.Tensor) -> torch.Tensor:
        """
        Projects the custom feature vector to the target embedding dimension.

        Args:
            custom_feature_vector (torch.Tensor):
                The raw custom feature.
                Shape: (batch_size, custom_feature_dim) if sequence-level feature.
                Shape: (batch_size, seq_len, custom_feature_dim) if token-level feature.

        Returns:
            torch.Tensor: The projected custom feature.
                          Shape: (batch_size, target_embedding_dim) for sequence-level.
                          Shape: (batch_size, seq_len, target_embedding_dim) for token-level.
        """
        projected_feature = self.projection_layer(custom_feature_vector)

        # if hasattr(self, 'activation'):
        #     projected_feature = self.activation(projected_feature)
        #
        # if hasattr(self, 'layer_norm'):
        #     projected_feature = self.layer_norm(projected_feature)

        return projected_feature
