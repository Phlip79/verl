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
"""CPU coverage for vLLM MTP configuration and actor weight synchronization."""

import importlib.util
from pathlib import Path

import pytest


def _load_rollout_utils():
    repo_root = Path(__file__).parents[4]
    path = repo_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "utils.py"
    spec = importlib.util.spec_from_file_location("vllm_rollout_mtp_utils_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingModel:
    def __init__(self):
        self.weights = []

    def load_weights(self, weights):
        self.weights.extend(weights)
        return {name for name, _ in self.weights}


def test_build_mtp_speculative_config_preserves_engine_overrides():
    rollout_utils = _load_rollout_utils()

    config = rollout_utils.build_mtp_speculative_config(
        "mtp",
        1,
        '{"num_speculative_tokens": 3, "disable_padded_drafter_batch": false}',
    )

    assert config == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "disable_padded_drafter_batch": False,
    }


def test_build_mtp_speculative_config_uses_model_defaults():
    rollout_utils = _load_rollout_utils()

    assert rollout_utils.build_mtp_speculative_config("mtp", 1) == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }


def test_build_mtp_speculative_config_rejects_non_mapping_override():
    rollout_utils = _load_rollout_utils()

    with pytest.raises(TypeError, match="must be a mapping"):
        rollout_utils.build_mtp_speculative_config("mtp", 1, ["not", "a", "mapping"])


def test_load_mtp_aware_weights_syncs_target_and_drafter():
    rollout_utils = _load_rollout_utils()
    target = _RecordingModel()
    drafter = _RecordingModel()
    weights = iter(
        [
            ("backbone.layers.0.weight", 1),
            ("mtp.layers.0.enorm.weight", 2),
            ("mtp.layers.1.mixer.weight", 3),
            ("lm_head.weight", 4),
        ]
    )

    rollout_utils.load_mtp_aware_weights(target, drafter, weights)

    assert [name for name, _ in target.weights] == [
        "backbone.layers.0.weight",
        "mtp.layers.0.enorm.weight",
        "mtp.layers.1.mixer.weight",
        "lm_head.weight",
    ]
    assert [name for name, _ in drafter.weights] == [
        "mtp.layers.0.enorm.weight",
        "mtp.layers.1.mixer.weight",
    ]


def test_load_mtp_aware_weights_streams_target_without_drafter():
    rollout_utils = _load_rollout_utils()
    target = _RecordingModel()

    target_result, draft_result = rollout_utils.load_mtp_aware_weights(
        target,
        None,
        iter([("backbone.layers.0.weight", 1)]),
    )

    assert target_result == {"backbone.layers.0.weight"}
    assert draft_result is None


def test_load_mtp_aware_weights_requires_mtp_tensors_for_drafter():
    rollout_utils = _load_rollout_utils()

    with pytest.raises(RuntimeError, match="did not export any 'mtp\\.\\*' tensors"):
        rollout_utils.load_mtp_aware_weights(
            _RecordingModel(),
            _RecordingModel(),
            iter([("backbone.layers.0.weight", 1)]),
        )
