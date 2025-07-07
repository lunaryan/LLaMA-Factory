import pytest
import torch
import torch.nn as nn

from llamafactory.model.model_utils.custom_feature_projector import CustomFeatureProjector

class TestCustomFeatureProjector:
    @pytest.fixture
    def projector_config_basic(self):
        return {
            "custom_feature_dim": 128,
            "model_hidden_dim": 768,
            "projector_hidden_act": "gelu",
            "use_layernorm": True,
        }

    @pytest.fixture
    def projector_config_no_ln(self):
        return {
            "custom_feature_dim": 128,
            "model_hidden_dim": 768,
            "projector_hidden_act": "relu",
            "use_layernorm": False,
        }

    @pytest.fixture
    def projector_config_linear_act(self):
        return {
            "custom_feature_dim": 128,
            "model_hidden_dim": 768,
            "projector_hidden_act": "linear", # or None
            "use_layernorm": True,
        }

    def test_projector_creation_and_attributes(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic)
        assert projector.custom_feature_dim == projector_config_basic["custom_feature_dim"]
        assert projector.model_hidden_dim == projector_config_basic["model_hidden_dim"]
        assert isinstance(projector.projection_layer, nn.Linear)
        assert projector.projection_layer.in_features == projector_config_basic["custom_feature_dim"]
        assert projector.projection_layer.out_features == projector_config_basic["model_hidden_dim"]
        assert projector.use_layernorm
        assert isinstance(projector.ln, nn.LayerNorm)
        assert not isinstance(projector.act, nn.Identity) # gelu

    def test_projector_creation_no_layernorm(self, projector_config_no_ln):
        projector = CustomFeatureProjector(**projector_config_no_ln)
        assert not projector.use_layernorm
        assert isinstance(projector.ln, nn.Identity)
        assert not isinstance(projector.act, nn.Identity) # relu

    def test_projector_creation_linear_activation(self, projector_config_linear_act):
        projector = CustomFeatureProjector(**projector_config_linear_act)
        assert isinstance(projector.act, nn.Identity)

    def test_forward_pass_shape_dtype(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic)
        batch_size = 4
        custom_features = torch.randn(batch_size, projector_config_basic["custom_feature_dim"], dtype=torch.float32)

        output = projector(custom_features)

        assert output.shape == (batch_size, projector_config_basic["model_hidden_dim"])
        assert output.dtype == torch.float32

    def test_forward_pass_different_dtype(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic).to(dtype=torch.bfloat16)
        batch_size = 4
        custom_features = torch.randn(batch_size, projector_config_basic["custom_feature_dim"], dtype=torch.bfloat16)

        output = projector(custom_features)

        assert output.shape == (batch_size, projector_config_basic["model_hidden_dim"])
        assert output.dtype == torch.bfloat16


    def test_forward_pass_incorrect_input_dim(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic)
        batch_size = 4
        wrong_dim = projector_config_basic["custom_feature_dim"] + 1
        custom_features = torch.randn(batch_size, wrong_dim)

        with pytest.raises(ValueError, match="Input feature dimension"):
            projector(custom_features)

    def test_parameters_have_gradients(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic)
        for param in projector.parameters():
            assert param.requires_grad

        for param in projector.parameters():
            param.requires_grad = False
        for param in projector.parameters():
            assert not param.requires_grad

    def test_repr_output(self, projector_config_basic):
        projector = CustomFeatureProjector(**projector_config_basic)
        representation = repr(projector)
        assert "CustomFeatureProjector" in representation
        assert f"custom_feature_dim={projector_config_basic['custom_feature_dim']}" in representation
        assert f"model_hidden_dim={projector_config_basic['model_hidden_dim']}" in representation
        assert "activation=GELUActivation" in representation
        assert "layernorm=True" in representation

    def test_repr_output_no_ln_relu_act(self, projector_config_no_ln):
        projector = CustomFeatureProjector(**projector_config_no_ln)
        representation = repr(projector)
        assert "activation=ReLU" in representation
        assert "layernorm=False" in representation

    def test_repr_output_linear_act(self, projector_config_linear_act):
        projector = CustomFeatureProjector(**projector_config_linear_act)
        representation = repr(projector)
        assert "activation=linear" in representation
        assert "layernorm=True" in representation
