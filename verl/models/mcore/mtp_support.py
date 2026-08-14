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
"""Compatibility helpers for Megatron-Core's native HybridModel MTP path."""

from packaging import version

_MIN_MEGATRON_BRIDGE_VERSION = "0.6.0"
_MIN_MEGATRON_CORE_VERSION = "0.19.0"


def require_native_mtp_versions() -> None:
    """Require the Megatron versions that expose native HybridModel MTP."""
    import megatron.bridge
    import megatron.core

    bridge_version = version.parse(megatron.bridge.__version__)
    mcore_version = version.parse(megatron.core.__version__)
    if bridge_version < version.parse(_MIN_MEGATRON_BRIDGE_VERSION) or mcore_version < version.parse(
        _MIN_MEGATRON_CORE_VERSION
    ):
        raise RuntimeError(
            "Native HybridModel MTP requires "
            f"Megatron-Bridge>={_MIN_MEGATRON_BRIDGE_VERSION} and "
            f"Megatron-Core>={_MIN_MEGATRON_CORE_VERSION}; found "
            f"Megatron-Bridge=={megatron.bridge.__version__} and Megatron-Core=={megatron.core.__version__}."
        )


def configure_native_hybrid_mtp(provider, mtp_config, transformer_overrides: dict) -> bool:
    """Validate and configure native HybridModel MTP before model construction."""
    is_hybrid_provider = bool(getattr(provider, "is_hybrid_model", False)) or any(
        provider_type.__name__ == "HybridModelProvider" for provider_type in type(provider).__mro__
    )
    if not mtp_config.enable or not is_hybrid_provider:
        return False

    if not mtp_config.enable_train:
        raise ValueError(
            "HybridModel does not support model.mtp.enable=True with model.mtp.enable_train=False "
            "in this Megatron-Core version. Disable MTP entirely or enable MTP training."
        )
    require_native_mtp_versions()

    transformer_overrides["mtp_detach_heads"] = bool(mtp_config.detach_encoder)
    return True


def is_native_hybrid_model(model) -> bool:
    """Return whether ``model`` is a Megatron-Core HybridModel."""
    try:
        from megatron.core.models.hybrid.hybrid_model import HybridModel
    except ImportError:
        return False
    return isinstance(model, HybridModel)
