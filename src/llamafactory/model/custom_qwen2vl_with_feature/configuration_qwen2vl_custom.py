from transformers import Qwen2VLConfig

class Qwen2VLCustomConfig(Qwen2VLConfig):
    model_type = "qwen2vl_custom" # Needs to be unique if we register a new AutoModel class

    def __init__(self, custom_feature_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.custom_feature_dim = custom_feature_dim
        # Add any other custom config parameters here if needed
        # For example, is_custom_feature_sequence_level, though this might be better as a model arg
        # rather than a config saved with the model if it's a training-time choice.
        # For now, let's assume custom_feature_dim is the main one needed in the config.

    # If LLaMAFactory's model loader directly uses AutoModelForVision2Seq.from_config,
    # and our custom model is also named Qwen2VLForConditionalGenerationCustom,
    # we might not need to change model_type here if we rely on architecture override in config.json.
    # However, giving it a unique model_type is cleaner if we were to register it formally.
    # For using with trust_remote_code, the architecture name in config.json is key.
    # Let's revert model_type for now to qwen2_vl to ensure LLaMA Factory's existing
    # visual component logic for qwen2_vl is picked up easily.
    # We will rely on the "architectures" field in the config.json of our custom model directory.
    # model_type = "qwen2_vl" # Keep original model_type for LLaMA Factory's internal dispatching for visual parts

# Re-evaluation: For custom code loading with `trust_remote_code=True`,
# the `model_type` in the *config file on disk* should match what `AutoConfig` expects
# for the *base model* (Qwen/Qwen2.5-VL-7B-Instruct's config has "qwen2_vl").
# The custom architecture is specified in the `architectures` list in that config.json.
# So, this custom config class is mostly for us to have a place to store custom_feature_dim
# if we were to load the model using `Qwen2VLForConditionalGenerationCustom.from_pretrained(..., config=our_custom_config_object)`.
# LLaMA Factory's `load_config` will load the `config.json` from the hub or local path.
# We will create a config.json in our custom model directory that includes our custom architecture.

# Let's simplify. The custom_feature_dim will be passed as an argument when creating the model,
# not necessarily stored in the config unless we are creating a totally new pretrainable model type.
# For fine-tuning an existing one with a wrapper, we can pass it to __init__.

# Revised configuration_qwen2vl_custom.py:
# We might not even need a custom config *class* if we pass parameters directly to the model __init__.
# The `config.json` in the custom model folder will be key.
# For now, let's assume we will add `custom_feature_dim` to the `config.json` that LLaMA Factory loads.
# And our custom model's __init__ will look for it there.

# Simpler approach: The custom model's __init__ will receive custom_feature_dim.
# The main config loaded by LLaMA Factory will be the standard Qwen2VLConfig.
# Our wrapper will just use it. So, this file might not be strictly necessary if
# custom_feature_dim is a direct init arg to our wrapper model, not part of the HF config object.

# Let's assume custom_feature_dim will be part of the main config.json loaded by LLaMA Factory.
# This means when LLaMA Factory calls AutoConfig.from_pretrained for our custom model folder,
# the config.json in that folder should have this field.
# So, the config class should reflect this.

# This class definition is more for if we were to *programmatically* create a config
# for our custom model. When LLaMA Factory loads a model, it loads the `config.json`
# from the `model_name_or_path` directly.

# The main purpose of this file, if used with `trust_remote_code`, is to be
# the place where the custom architecture class (defined in modeling_qwen2vl_custom.py)
# is associated with the model_type. But that's usually done by having this code
# available and setting `architectures` in `config.json`.

# For now, let's keep it minimal, mainly to signal this is a custom config.
# The actual `custom_feature_dim` will be added to the `config.json` file manually later.
# And the model's __init__ will expect it.
# config.json will have: "model_type": "qwen2_vl" (to keep LLaMA Factory happy for base model logic)
# and "architectures": ["Qwen2VLCustomForConditionalGeneration"]

class Qwen2VLCustomConfig(Qwen2VLConfig):
    # No actual changes needed here if custom_feature_dim is read from the loaded config object
    # by the custom model. We just need this class to exist if referenced by model_type.
    # Let's assume the actual Qwen2VLConfig loaded by LLaMA Factory will be augmented
    # (in our custom model's directory) with `custom_feature_dim`.
    pass
