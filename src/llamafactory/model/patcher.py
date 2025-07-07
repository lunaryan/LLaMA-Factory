# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model
from .model_utils.custom_feature_projector import CustomFeatureProjector


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    if "InternVLChatModel" in getattr(config, "architectures", []):
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    # Custom Feature Projector Integration
    if model_args.custom_feature_dim and model_args.custom_feature_dim > 0:
        model_hidden_dim = getattr(model.config, "hidden_size", None)
        if model_hidden_dim is None and hasattr(model.config, "text_config"): # Common for VLMs
            model_hidden_dim = getattr(model.config.text_config, "hidden_size", None)

        if model_hidden_dim is None:
            if hasattr(model, "config") and hasattr(model.config, "hidden_size"): # Fallback
                 model_hidden_dim = model.config.hidden_size
            else:
                raise ValueError("Could not determine model hidden dimension for CustomFeatureProjector.")

        target_device = model.device if hasattr(model, "device") and model.device is not None else \
                        next(model.parameters()).device if len(list(model.parameters())) > 0 else \
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure compute_dtype is a torch.dtype object
        if isinstance(model_args.compute_dtype, str): # Should have been converted by patch_config
            model_args.compute_dtype = getattr(torch, model_args.compute_dtype)

        target_dtype = model_args.compute_dtype or (next(model.parameters()).dtype if len(list(model.parameters())) > 0 else torch.float32)

        model.custom_feature_projector = CustomFeatureProjector(
            custom_feature_dim=model_args.custom_feature_dim,
            model_hidden_dim=model_hidden_dim,
            projector_hidden_act=model_args.custom_projector_hidden_act,
            use_layernorm=model_args.custom_projector_use_layernorm,
        ).to(device=target_device, dtype=target_dtype)

        logger.info_rank0(
            f"Initialized CustomFeatureProjector ({model.custom_feature_projector}) "
            f"on device {target_device} with dtype {target_dtype}."
        )

        if is_trainable: # This ensures projector is trainable if overall training is enabled
            for param in model.custom_feature_projector.parameters():
                param.requires_grad = True
            logger.info_rank0("Set CustomFeatureProjector parameters to trainable.")

        # Patching language model's forward pass to incorporate the custom feature
        lm_sub_component = None
        if hasattr(model, "language_model"):
            lm_sub_component = model.language_model
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model.model, "layers"):
            if not isinstance(model.model, PeftModel):
                lm_sub_component = model.model

        if lm_sub_component:
            _original_lm_forward = lm_sub_component.forward

            # Use `model` from the outer scope of `patch_model` to access `custom_feature_projector`
            # and the temporary attribute `_temp_custom_feature_tensor_for_lm_patch`
            # `self_lang_model` is the instance of lm_sub_component
            def patched_language_model_forward(self_lang_model, *patch_args, **patch_kwargs):
                inputs_embeds = patch_kwargs.get("inputs_embeds")
                custom_feature_tensor = getattr(model, "_temp_custom_feature_tensor_for_lm_patch", None)

                if inputs_embeds is not None and custom_feature_tensor is not None and hasattr(model, "custom_feature_projector"):
                    projected_custom_feature = model.custom_feature_projector(custom_feature_tensor)
                    projected_custom_feature = projected_custom_feature.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
                    patch_kwargs["inputs_embeds"] = inputs_embeds + projected_custom_feature.unsqueeze(1)
                elif custom_feature_tensor is not None:
                    logger.warning_rank0(
                        "Custom feature tensor provided to language model, but could not apply it."
                    )
                return _original_lm_forward(*patch_args, **patch_kwargs)

            lm_sub_component.forward = MethodType(patched_language_model_forward, lm_sub_component)
            logger.info_rank0(f"Patched forward of LM sub-component ({lm_sub_component.__class__.__name__}) for custom features.")

            _original_main_model_forward = model.forward
            # `self_main_model` is the instance of the main model (`model`)
            def patched_main_model_forward(self_main_model, *patch_args, **patch_kwargs):
                custom_feature_tensor_local = patch_kwargs.pop("custom_feature_tensor", None)

                if custom_feature_tensor_local is not None:
                    # Use self_main_model to set/del attribute, which is `model` in this context
                    setattr(self_main_model, "_temp_custom_feature_tensor_for_lm_patch", custom_feature_tensor_local)

                output = _original_main_model_forward(*patch_args, **patch_kwargs)

                if hasattr(self_main_model, "_temp_custom_feature_tensor_for_lm_patch"):
                    delattr(self_main_model, "_temp_custom_feature_tensor_for_lm_patch")

                return output

            model.forward = MethodType(patched_main_model_forward, model)
            logger.info_rank0("Patched main model's forward to pass custom_feature_tensor to LM patch.")
        else:
            logger.warning_rank0("Language model sub-component not found. Cannot patch for custom features.")

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
