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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Import this stdlib-only helper module directly. Importing its package also
# imports the vLLM/torch rollout implementation, which is intentionally absent
# from CPU-only test images.
_UTILS_PATH = Path(__file__).resolve().parents[4] / "verl" / "workers" / "rollout" / "vllm_rollout" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("verl_vllm_weight_sync_helpers", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


class _FakeTensor:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


class _FakeModel:
    def __init__(self):
        self.loads = []

    def load_weights(self, weights):
        self.loads.append([name for name, _ in weights])


def test_max_position_embeddings_supports_flat_and_nested_configs():
    assert utils.get_max_position_embeddings(SimpleNamespace(max_position_embeddings=8192)) == 8192
    nested = SimpleNamespace(max_position_embeddings=None, text_config=SimpleNamespace(max_position_embeddings=4096))
    assert utils.get_max_position_embeddings(nested) == 4096

    with pytest.raises(ValueError, match="max_position_embeddings"):
        utils.get_max_position_embeddings(SimpleNamespace())


def test_mtp_config_uses_rollout_then_model_fallback():
    model_mtp = SimpleNamespace(enable=True, enable_rollout=True)
    model_config = SimpleNamespace(mtp=model_mtp)

    assert utils.get_mtp_config(SimpleNamespace(), model_config) is model_mtp
    assert utils.is_mtp_rollout_enabled(SimpleNamespace(), model_config)

    rollout_mtp = SimpleNamespace(enable=True, enable_rollout=False)
    rollout_config = SimpleNamespace(mtp=rollout_mtp)
    assert utils.get_mtp_config(rollout_config, model_config) is rollout_mtp
    assert not utils.is_mtp_rollout_enabled(rollout_config, model_config)

    mapping_config = {"mtp": {"enable": True, "enable_rollout": True}}
    assert utils.is_mtp_rollout_enabled(mapping_config, {})
    assert utils.get_config_value(mapping_config["mtp"], "method", "mtp") == "mtp"


def test_mtp_speculative_config_merges_json_engine_overrides():
    result = utils.build_mtp_speculative_config(
        "mtp",
        1,
        '{"num_speculative_tokens": 1, "disable_by_batch_size": 16, "ignored": null}',
    )
    assert result == {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "disable_by_batch_size": 16,
    }

    with pytest.raises(TypeError, match="must be a mapping"):
        utils.build_mtp_speculative_config("mtp", 1, ["invalid"])


def test_worker_wrapper_constructor_and_method_dispatch_cover_vllm_027():
    vllm_config = object()

    class LegacyWrapper:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config

        def execute_method(self, method, *args, **kwargs):
            return method, args, kwargs

    legacy = utils.create_vllm_worker_wrapper(LegacyWrapper, vllm_config)
    assert legacy.vllm_config is vllm_config
    assert utils.execute_vllm_worker_method(legacy, "work", (1,), {"value": 2}) == (
        "work",
        (1,),
        {"value": 2},
    )

    class Vllm027Wrapper:
        def __init__(self):
            self.value = 7

    modern = utils.create_vllm_worker_wrapper(Vllm027Wrapper, vllm_config)

    def fake_run_method(obj, method, args, kwargs):
        return obj.value, method, args, kwargs

    assert utils.execute_vllm_worker_method(modern, "work", (1,), {"value": 2}, fallback=fake_run_method) == (
        7,
        "work",
        (1,),
        {"value": 2},
    )


def test_mtp_weight_sync_selects_target_and_exact_v1_drafter_shape():
    target_model = object()
    target_config = object()
    draft_model = object()
    draft_config = object()
    runner = SimpleNamespace(
        model=target_model,
        drafter=SimpleNamespace(model=draft_model),
        vllm_config=SimpleNamespace(
            model_config=target_config,
            speculative_config=SimpleNamespace(method="mtp", draft_model_config=draft_config),
        ),
    )

    assert utils.get_vllm_models_for_weight_sync(runner) == [target_model, draft_model]
    assert utils.get_vllm_models_and_configs_for_weight_sync(runner) == [
        (target_model, target_config),
        (draft_model, draft_config),
    ]

    runner.vllm_config.speculative_config.method = "ngram"
    assert utils.get_vllm_models_for_weight_sync(runner) == [target_model]


def test_mtp_weight_sync_requires_draft_model_config_for_post_load():
    runner = SimpleNamespace(
        model=object(),
        drafter=SimpleNamespace(model=object()),
        vllm_config=SimpleNamespace(
            model_config=object(),
            speculative_config=SimpleNamespace(method="mtp", draft_model_config=None),
        ),
    )

    with pytest.raises(RuntimeError, match="draft_model_config"):
        utils.get_vllm_models_and_configs_for_weight_sync(runner)


def test_weight_buckets_bound_replay_of_one_shot_stream():
    visited = []

    def weights():
        for name, nbytes in (("a", 600_000), ("b", 600_000), ("c", 100_000)):
            visited.append(name)
            yield name, _FakeTensor(nbytes)

    buckets = list(utils.iter_weight_buckets(weights(), bucket_megabytes=1))

    assert [[name for name, _ in bucket] for bucket in buckets] == [["a"], ["b", "c"]]
    assert visited == ["a", "b", "c"]
    with pytest.raises(ValueError, match="must be positive"):
        list(utils.iter_weight_buckets([], bucket_megabytes=0))


def test_one_shot_weight_stream_reaches_target_and_drafter_then_postprocesses_once():
    target = _FakeModel()
    draft = _FakeModel()

    def weights():
        yield "a", _FakeTensor(600_000)
        yield "b", _FakeTensor(600_000)
        yield "c", _FakeTensor(100_000)

    utils.load_weights_into_vllm_models([target, draft], weights(), bucket_megabytes=1)
    assert target.loads == [["a"], ["b", "c"]]
    assert draft.loads == target.loads

    target_config = object()
    draft_config = object()
    runner = SimpleNamespace(
        model=target,
        device="cuda:0",
        drafter=SimpleNamespace(model=draft),
        vllm_config=SimpleNamespace(
            model_config=target_config,
            speculative_config=SimpleNamespace(method="mtp", draft_model_config=draft_config),
        ),
    )
    post_load_calls = []
    utils.process_vllm_models_after_weight_sync(
        runner,
        process_weights=lambda model, model_config, device: post_load_calls.append((model, model_config, device)),
    )
    assert post_load_calls == [
        (target, target_config, "cuda:0"),
        (draft, draft_config, "cuda:0"),
    ]
