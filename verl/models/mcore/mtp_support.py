# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Capability-based support helpers for Megatron-Core native HybridModel MTP."""

from functools import lru_cache
from inspect import signature


@lru_cache(maxsize=1)
def has_native_mtp_support() -> bool:
    """Return whether the installed MCore exposes the native MTP API we need."""

    try:
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.multi_token_prediction import process_mtp_loss

        config_fields = getattr(TransformerConfig, "__dataclass_fields__", {})
        has_detach_heads = hasattr(TransformerConfig, "mtp_detach_heads") or "mtp_detach_heads" in config_fields
        return has_detach_heads and "input_ids" in signature(process_mtp_loss).parameters
    except (ImportError, TypeError, ValueError):
        return False


def configure_native_hybrid_mtp(provider, mtp_config, transformer_overrides: dict) -> bool:
    """Configure native HybridModel MTP before a Bridge provider is finalized."""

    is_hybrid_provider = bool(getattr(provider, "is_hybrid_model", False)) or any(
        provider_type.__name__ == "HybridModelProvider" for provider_type in type(provider).__mro__
    )
    if not is_hybrid_provider:
        return False
    if not mtp_config.enable:
        # Hybrid providers can infer a physical repeated MTP block from the
        # retained pattern even when mtp_num_layers is None. Keep these
        # provider-specific flags away from strict GPT providers.
        transformer_overrides.update(
            {
                "mtp_hybrid_override_pattern": None,
                "mtp_use_repeated_layer": False,
                "keep_mtp_spec_in_bf16": False,
            }
        )
        return False
    if not mtp_config.enable_train:
        raise ValueError("Native HybridModel requires model.mtp.enable_train=True when model.mtp.enable=True.")
    if not has_native_mtp_support():
        raise RuntimeError(
            "The installed Megatron-Core lacks the native HybridModel MTP API required by this model. "
            "Use the dependency snapshot documented for Nemotron 3.5 Lightning."
        )
    transformer_overrides["mtp_detach_heads"] = bool(mtp_config.detach_encoder)
    return True


def is_native_hybrid_model(model) -> bool:
    """Return whether ``model`` is an MCore native HybridModel instance."""

    if not has_native_mtp_support():
        return False
    try:
        from megatron.core.models.hybrid.hybrid_model import HybridModel
    except ImportError:
        return False
    return isinstance(model, HybridModel)
