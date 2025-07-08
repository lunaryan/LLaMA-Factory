import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
from typing import Optional # Import Optional

# Assuming custom_feature_layer.py is in the same directory or accessible in PYTHONPATH
from custom_feature_layer import CustomFeatureProjector

class QwenWithCustomFeature(nn.Module):
    def __init__(self, qwen_model_name_or_path: str, custom_feature_dim: int,
                 is_custom_feature_sequence_level: bool = True):
        super().__init__()
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_model_name_or_path)
        self.config = self.qwen_model.config # Convenience
        self.text_hidden_size = self.config.text_config.hidden_size

        self.feature_projector = CustomFeatureProjector(
            custom_feature_dim=custom_feature_dim,
            target_embedding_dim=self.text_hidden_size
        )
        self.is_custom_feature_sequence_level = is_custom_feature_sequence_level

    def get_input_embeddings(self):
        return self.qwen_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.qwen_model.get_output_embeddings()

    def forward(self, input_ids: torch.LongTensor,
                custom_feature: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[list[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs
                ):

        token_embeddings = self.qwen_model.model.language_model.embed_tokens(input_ids)

        projected_custom_feature = self.feature_projector(custom_feature)

        if self.is_custom_feature_sequence_level:
            projected_custom_feature_expanded = projected_custom_feature.unsqueeze(1)
            combined_embeddings = token_embeddings + projected_custom_feature_expanded
        else:
            combined_embeddings = token_embeddings + projected_custom_feature

        return self.qwen_model(
            input_ids=None,
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, custom_feature, past_key_values=None, **kwargs):
        # Note: This is a simplified placeholder. Real implementation for generation
        # with custom features requires careful handling of inputs across generation steps,
        # especially with beam search and multimedia inputs.
        # For training the projector layer, this simplified version might be sufficient if generate is not called.

        model_kwargs = self.qwen_model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)

        current_input_ids = model_kwargs['input_ids']
        # Handle the case where 'input_ids' might not be present if 'inputs_embeds' is already being used by the base model
        if current_input_ids is None and 'inputs_embeds' in model_kwargs:
             # This scenario is complex: if inputs_embeds is already set, we'd be overwriting it.
             # Or, we'd need to add our feature to the existing inputs_embeds.
             # For now, assume input_ids is primary for the first step or when not using past_key_values.
             # This part needs robust design if generation is a primary use case for the wrapper.
             pass # Needs careful thought if this path is taken.

        if current_input_ids is not None:
            token_embeddings = self.qwen_model.model.language_model.embed_tokens(current_input_ids)
            projected_custom_feature = self.feature_projector(custom_feature)

            if self.is_custom_feature_sequence_level:
                if custom_feature.shape[0] != token_embeddings.shape[0]:
                    num_beams = token_embeddings.shape[0] // custom_feature.shape[0]
                    projected_custom_feature = projected_custom_feature.repeat_interleave(num_beams, dim=0)
                projected_custom_feature_expanded = projected_custom_feature.unsqueeze(1)
                combined_embeddings = token_embeddings + projected_custom_feature_expanded
            else:
                if custom_feature.shape[0] != token_embeddings.shape[0] and custom_feature.ndim > 1 and token_embeddings.ndim > 1 and custom_feature.shape[1] == token_embeddings.shape[1]:
                    num_beams = token_embeddings.shape[0] // custom_feature.shape[0]
                    projected_custom_feature = projected_custom_feature.repeat_interleave(num_beams, dim=0)
                combined_embeddings = token_embeddings + projected_custom_feature

            model_kwargs['inputs_embeds'] = combined_embeddings
            model_kwargs['input_ids'] = None

        model_kwargs['custom_feature'] = custom_feature # Pass it along

        # Retain Qwen2VL's specific logic for multimedia and position_ids from its prepare_inputs_for_generation
        # This is usually handled within the call to self.qwen_model.prepare_inputs_for_generation
        # and then we modify/add to model_kwargs.

        return model_kwargs

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.qwen_model, name):
                return getattr(self.qwen_model, name)
            raise
