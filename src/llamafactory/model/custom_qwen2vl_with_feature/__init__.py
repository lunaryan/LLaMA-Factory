# This file makes Python treat the directory custom_qwen2vl_with_feature as a package.
# It can also be used to control what is imported when `from .custom_qwen2vl_with_feature import *` is used.

# For Hugging Face `trust_remote_code=True` to work, it primarily relies on finding
# the custom code files (modeling_*.py, configuration_*.py) in the specified path.
# The `config.json`'s `auto_map` or `architectures` field then points to the classes within these files.

# We need to make sure LLaMA Factory can find our custom classes.
# If LLaMA Factory's model loader uses `AutoConfig.from_pretrained` and then
# `AutoModelFor[Task].from_pretrained` with `trust_remote_code=True`,
# having these files in the directory specified by `model_name_or_path`
# and a correct `config.json` is the main requirement.

# Explicitly importing them here can help if other parts of LLaMA Factory
# try to import directly from this module path, though it's often not strictly necessary
# for the `trust_remote_code` mechanism itself.

from .configuration_qwen2vl_custom import Qwen2VLCustomConfig
from .modeling_qwen2vl_custom import Qwen2VLCustomForConditionalGeneration
from .custom_feature_layer import CustomFeatureProjector

__all__ = ["Qwen2VLCustomConfig", "Qwen2VLCustomForConditionalGeneration", "CustomFeatureProjector"]
