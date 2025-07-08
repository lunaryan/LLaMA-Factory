from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
# Import CausalLMOutputWithPast if it's the specific output type, or a more generic one
from transformers.modeling_outputs import ModelOutput # Using a more generic base
try:
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    # Fallback or define a compatible structure if CausalLMOutputWithPast is not available
    # For recent transformers versions, it should be.
    class CausalLMOutputWithPast(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

from .custom_feature_layer import CustomFeatureProjector


class Qwen2VLCustomForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)

        custom_feature_dim = getattr(config, 'custom_feature_dim', None)
        if custom_feature_dim is None:
            # Try to get it from text_config as a fallback if not on main config
            # This depends on how we structure the config.json in the custom model dir
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'custom_feature_dim'):
                custom_feature_dim = config.text_config.custom_feature_dim
            else:
                raise ValueError(
                    "`custom_feature_dim` must be specified in the model config "
                    "for Qwen2VLCustomForConditionalGeneration."
                )

        # Ensure text_config exists, which it should for Qwen2VLConfig
        if not hasattr(config, 'text_config'):
            raise ValueError("Qwen2VLConfig must have a text_config attribute.")

        text_hidden_size = config.text_config.hidden_size

        self.feature_projector = CustomFeatureProjector(
            custom_feature_dim=custom_feature_dim,
            target_embedding_dim=text_hidden_size
        )
        self.is_custom_feature_sequence_level = getattr(config, 'is_custom_feature_sequence_level', True)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        custom_feature: Optional[torch.Tensor] = None, # Our new input
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is not None and custom_feature is not None:
            raise ValueError("Cannot provide both inputs_embeds and custom_feature. If using custom_feature, input_ids must be provided.")
        if input_ids is None and inputs_embeds is None:
             raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if custom_feature is not None and input_ids is None:
            raise ValueError("input_ids must be provided when custom_feature is used.")

        final_inputs_embeds = inputs_embeds
        input_ids_for_super = input_ids # Default to passing input_ids

        if custom_feature is not None:
            token_embeddings = self.model.language_model.embed_tokens(input_ids)
            projected_custom_feature = self.feature_projector(custom_feature)

            if self.is_custom_feature_sequence_level:
                if projected_custom_feature.ndim == 2 and token_embeddings.ndim == 3:
                    # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
                    projected_custom_feature_expanded = projected_custom_feature.unsqueeze(1)
                else:
                    # This case implies custom_feature might already be (batch_size, 1, hidden_size)
                    # or there's a dimension mismatch. For safety, let's assume it needs unsqueezing
                    # if it's 2D and token_embeddings is 3D.
                    projected_custom_feature_expanded = projected_custom_feature
                    if projected_custom_feature.ndim != token_embeddings.ndim:
                         projected_custom_feature_expanded = projected_custom_feature.unsqueeze(1)


                # Ensure broadcasting is possible or expand explicitly
                if token_embeddings.shape[1] > 1 and projected_custom_feature_expanded.shape[1] == 1:
                     projected_custom_feature_expanded = projected_custom_feature_expanded.expand(-1, token_embeddings.shape[1], -1)

                combined_embeddings = token_embeddings + projected_custom_feature_expanded

            else: # Token-level
                combined_embeddings = token_embeddings + projected_custom_feature

            final_inputs_embeds = combined_embeddings
            input_ids_for_super = None # Crucial: set to None when using inputs_embeds for superclass

        # All other arguments are passed through to the superclass's forward method.
        # LLaMA Factory's data collator should provide `pixel_values`, `image_grid_thw` etc.,
        # if they are part of the processed dataset. The Qwen2VLForConditionalGeneration.forward
        # (and its internal Qwen2VLModel.forward) will handle these multimodal inputs using `final_inputs_embeds`.

        return super().forward(
            input_ids=input_ids_for_super,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=final_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs
        )

    # Regarding prepare_inputs_for_generation:
    # If using model.generate(), this method in the custom class would also need to be
    # overridden to correctly handle `custom_feature` across generation steps.
    # This involves:
    # 1. Accepting `custom_feature` as a kwarg.
    # 2. When `inputs_embeds` is not yet in `model_inputs` (first step), generate it using
    #    `input_ids` and `custom_feature` similar to the forward pass.
    # 3. If `inputs_embeds` is already present (subsequent steps, usually only contains the last token's embed),
    #    this logic becomes more complex. Typically, only the embeddings for the new tokens are needed.
    #    The `custom_feature` (if sequence-level) would have already been "absorbed" into the
    #    past_key_values implicitly. If it's token-level, it would need to be sliced for the new tokens.
    # For simply fine-tuning the projector layer via Trainer.train(), overriding `forward` is the primary concern.
    # We can add a basic prepare_inputs_for_generation later if generation is explicitly required.
