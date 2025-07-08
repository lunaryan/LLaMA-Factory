import torch
import torch.nn as nn

class CustomFeatureProjector(nn.Module):
    def __init__(self, custom_feature_dim: int, target_embedding_dim: int):
        super().__init__()
        self.custom_feature_dim = custom_feature_dim
        self.target_embedding_dim = target_embedding_dim
        self.projection_layer = nn.Linear(custom_feature_dim, target_embedding_dim)
        # Optional: self.activation = nn.ReLU()
        # Optional: self.layer_norm = nn.LayerNorm(target_embedding_dim)

    def forward(self, custom_feature_vector: torch.Tensor) -> torch.Tensor:
        projected_feature = self.projection_layer(custom_feature_vector)
        # if hasattr(self, 'activation'):
        #     projected_feature = self.activation(projected_feature)
        # if hasattr(self, 'layer_norm'):
        #     projected_feature = self.layer_norm(projected_feature)
        return projected_feature
