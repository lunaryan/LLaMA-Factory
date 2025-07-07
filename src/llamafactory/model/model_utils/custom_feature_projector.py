import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class CustomFeatureProjector(nn.Module):
    def __init__(
        self,
        custom_feature_dim: int,
        model_hidden_dim: int,
        projector_hidden_act: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.custom_feature_dim = custom_feature_dim
        self.model_hidden_dim = model_hidden_dim
        self.use_layernorm = use_layernorm

        self.projection_layer = nn.Linear(custom_feature_dim, model_hidden_dim, bias=True)

        if projector_hidden_act is not None and projector_hidden_act != "linear":
            self.act = ACT2FN[projector_hidden_act]
        else:
            self.act = nn.Identity()

        if self.use_layernorm:
            self.ln = nn.LayerNorm(model_hidden_dim, bias=True)
        else:
            self.ln = nn.Identity()

    def forward(self, custom_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            custom_features (torch.Tensor): Tensor of shape [batch_size, custom_feature_dim]
        Returns:
            torch.Tensor: Tensor of shape [batch_size, model_hidden_dim]
        """
        if custom_features.shape[-1] != self.custom_feature_dim:
            raise ValueError(
                f"Input feature dimension {custom_features.shape[-1]} "
                f"does not match expected custom_feature_dim {self.custom_feature_dim}"
            )

        projected_features = self.projection_layer(custom_features)
        projected_features = self.act(projected_features)
        projected_features = self.ln(projected_features)

        return projected_features

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"custom_feature_dim={self.custom_feature_dim}, "
            f"model_hidden_dim={self.model_hidden_dim}, "
            f"activation={self.act.__class__.__name__ if not isinstance(self.act, nn.Identity) else 'linear'}, "
            f"layernorm={self.use_layernorm})"
        )
