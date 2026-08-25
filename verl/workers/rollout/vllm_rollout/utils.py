# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


def get_max_position_embeddings(hf_config: Any) -> int:
    """Read the text context limit from flat and nested HF configs."""
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        max_len = getattr(text_config, "max_position_embeddings", None)
    if max_len is None:
        raise ValueError("max_position_embeddings not found in the Hugging Face config")
    return int(max_len)


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a config field from dataclass, OmegaConf, or mapping forms."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def get_mtp_config(rollout_config: Any, model_config: Any) -> Any:
    """Return MTP config from rollout config, with a model-config fallback.

    The fallback keeps the v0.7 rollout compatible with the newer shared model
    MTP config without requiring the larger rollout-config refactor.
    """
    mtp_config = get_config_value(rollout_config, "mtp")
    if mtp_config is None:
        mtp_config = get_config_value(model_config, "mtp")
    return mtp_config


def is_mtp_rollout_enabled(rollout_config: Any, model_config: Any) -> bool:
    """Return whether the rollout should create and synchronize an MTP drafter."""
    mtp_config = get_mtp_config(rollout_config, model_config)
    return bool(
        mtp_config is not None
        and get_config_value(mtp_config, "enable", False)
        and get_config_value(mtp_config, "enable_rollout", False)
    )


def build_mtp_speculative_config(
    method: str, num_speculative_tokens: int, engine_speculative_config: Any = None
) -> dict[str, Any]:
    """Build vLLM's MTP speculative config and apply engine overrides."""
    if engine_speculative_config is None:
        engine_speculative_config = {}
    if isinstance(engine_speculative_config, str):
        engine_speculative_config = json.loads(engine_speculative_config)
    if not isinstance(engine_speculative_config, Mapping):
        raise TypeError("rollout.engine_kwargs.vllm.speculative_config must be a mapping")

    return {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        **{key: value for key, value in engine_speculative_config.items() if value is not None},
    }


def create_vllm_worker_wrapper(worker_wrapper_cls: type, vllm_config: Any) -> Any:
    """Instantiate WorkerWrapperBase across the pre- and post-0.13 APIs."""
    try:
        return worker_wrapper_cls(vllm_config=vllm_config)
    except TypeError:
        return worker_wrapper_cls()


def execute_vllm_worker_method(
    worker_wrapper: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fallback: Callable | None = None,
) -> Any:
    """Use a wrapper's legacy dispatcher or vLLM V1's run_method."""
    execute_method = getattr(worker_wrapper, "execute_method", None)
    if execute_method is not None:
        return execute_method(method, *args, **kwargs)
    if fallback is None:
        from vllm.v1.serial_utils import run_method

        fallback = run_method
    return fallback(worker_wrapper, method, args, kwargs)


def get_vllm_models_for_weight_sync(model_runner: Any) -> list[Any]:
    """Return the target model and, for MTP only, its colocated draft model."""
    models = [model_runner.model]
    speculative_config = getattr(model_runner.vllm_config, "speculative_config", None)
    drafter = getattr(model_runner, "drafter", None)
    if (
        speculative_config is not None
        and getattr(speculative_config, "method", None) == "mtp"
        and drafter is not None
        and getattr(drafter, "model", None) is not None
    ):
        models.append(drafter.model)
    return models


def get_vllm_models_and_configs_for_weight_sync(model_runner: Any) -> list[tuple[Any, Any]]:
    """Pair every dynamically updated model with its vLLM model config."""
    models_and_configs = [(model_runner.model, model_runner.vllm_config.model_config)]
    if len(get_vllm_models_for_weight_sync(model_runner)) == 1:
        return models_and_configs

    speculative_config = model_runner.vllm_config.speculative_config
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        raise RuntimeError("vLLM created an MTP drafter without a draft_model_config")
    models_and_configs.append((model_runner.drafter.model, draft_model_config))
    return models_and_configs


def iter_weight_buckets(weights: Iterable[tuple[str, Any]], bucket_megabytes: int) -> Iterator[list[tuple[str, Any]]]:
    """Bound the extra memory needed to replay a one-shot stream to two models."""
    if bucket_megabytes <= 0:
        raise ValueError("update_weights_bucket_megabytes must be positive")

    bucket_limit = bucket_megabytes * 1024 * 1024
    bucket: list[tuple[str, Any]] = []
    bucket_bytes = 0
    for name, tensor in weights:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if bucket and bucket_bytes + tensor_bytes > bucket_limit:
            yield bucket
            bucket = []
            bucket_bytes = 0
        bucket.append((name, tensor))
        bucket_bytes += tensor_bytes
    if bucket:
        yield bucket


def load_weights_into_vllm_models(models: list[Any], weights: Iterable[tuple[str, Any]], bucket_megabytes: int) -> None:
    """Load a one-shot actor stream into the target and optional MTP draft."""
    if len(models) == 1:
        models[0].load_weights(weights)
        return
    for bucket in iter_weight_buckets(weights, bucket_megabytes):
        for model in models:
            model.load_weights(bucket)


def process_vllm_models_after_weight_sync(model_runner: Any, process_weights: Callable | None = None) -> None:
    """Run vLLM's post-load transform once per updated model."""
    if process_weights is None:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        process_weights = process_weights_after_loading
    for model, model_config in get_vllm_models_and_configs_for_weight_sync(model_runner):
        process_weights(model, model_config, model_runner.device)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")
