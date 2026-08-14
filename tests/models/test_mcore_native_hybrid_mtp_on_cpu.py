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
"""CPU coverage for the native Megatron-Core HybridModel MTP adapter."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_forward_with_stubs(monkeypatch, *, native_hybrid=True):
    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model

    workers_config = types.ModuleType("verl.workers.config")
    workers_config.MtpConfig = object

    mtp_support = types.ModuleType("verl.models.mcore.mtp_support")
    mtp_support.is_native_hybrid_model = lambda model: native_hybrid

    util = types.ModuleType("verl.models.mcore.util")
    util_names = [
        "postprocess_bshd",
        "postprocess_bshd_no_padding",
        "postprocess_packed_seqs",
        "postprocess_thd_no_padding",
        "preprocess_bshd",
        "preprocess_bshd_no_padding",
        "preprocess_packed_seqs",
        "preprocess_thd_no_padding",
    ]
    for name in util_names:
        setattr(util, name, lambda *args, **kwargs: None)

    mcore_package = types.ModuleType("verl.models.mcore")
    mcore_package.__path__ = []
    monkeypatch.setitem(sys.modules, "verl.models.mcore", mcore_package)
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", megatron_utils)
    monkeypatch.setitem(sys.modules, "verl.workers.config", workers_config)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.mtp_support", mtp_support)
    monkeypatch.setitem(sys.modules, "verl.models.mcore.util", util)

    return _load_module(
        "verl.models.mcore.model_forward_native_mtp_test",
        "verl/models/mcore/model_forward.py",
    )


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


@pytest.mark.parametrize(
    ("bridge_version", "mcore_version"),
    [("0.6.0", "0.19.0"), ("0.7.0.dev0", "0.20.0rc1")],
)
def test_native_mtp_accepts_supported_versions(monkeypatch, bridge_version, mcore_version):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    bridge = types.ModuleType("megatron.bridge")
    core.__version__ = mcore_version
    bridge.__version__ = bridge_version
    megatron.core = core
    megatron.bridge = bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    mtp_support = _load_module(
        "mtp_support_version_test",
        "verl/models/mcore/mtp_support.py",
    )

    mtp_support.require_native_mtp_versions()


@pytest.mark.parametrize(
    ("bridge_version", "mcore_version"),
    [("0.5.9", "0.19.0"), ("0.6.0", "0.18.9")],
)
def test_native_mtp_rejects_unsupported_versions(monkeypatch, bridge_version, mcore_version):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    bridge = types.ModuleType("megatron.bridge")
    core.__version__ = mcore_version
    bridge.__version__ = bridge_version
    megatron.core = core
    megatron.bridge = bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    mtp_support = _load_module(
        "mtp_support_old_version_test",
        "verl/models/mcore/mtp_support.py",
    )

    with pytest.raises(RuntimeError, match="Megatron-Bridge>=0.6.0.*Megatron-Core>=0.19.0"):
        mtp_support.require_native_mtp_versions()


@pytest.mark.parametrize("detach_encoder", [False, True])
def test_configure_native_hybrid_mtp_maps_detach_encoder(monkeypatch, detach_encoder):
    mtp_support = _load_module(
        "mtp_support_config_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "require_native_mtp_versions", lambda: None)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=detach_encoder)
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is True
    assert overrides["mtp_detach_heads"] is detach_encoder


@pytest.mark.parametrize(
    "provider",
    [
        type("HybridModelProvider", (), {})(),
        type("MarkedHybridProvider", (), {"is_hybrid_model": True})(),
    ],
    ids=["provider-class-name", "explicit-marker"],
)
def test_configure_native_hybrid_mtp_recognizes_supported_provider_forms(monkeypatch, provider):
    mtp_support = _load_module(
        "mtp_support_provider_detection_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "require_native_mtp_versions", lambda: None)
    mtp_config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=False)
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is True
    assert overrides == {"mtp_detach_heads": False}


def test_configure_native_hybrid_mtp_recognizes_provider_subclass(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_provider_subclass_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "require_native_mtp_versions", lambda: None)

    class HybridModelProvider:
        pass

    class SpecializedHybridProvider(HybridModelProvider):
        pass

    overrides = {}
    configured = mtp_support.configure_native_hybrid_mtp(
        SpecializedHybridProvider(),
        SimpleNamespace(enable=True, enable_train=True, detach_encoder=True),
        overrides,
    )

    assert configured is True
    assert overrides == {"mtp_detach_heads": True}


def test_configure_native_hybrid_mtp_rejects_skip_compute(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_skip_compute_test",
        "verl/models/mcore/mtp_support.py",
    )
    monkeypatch.setattr(mtp_support, "require_native_mtp_versions", lambda: None)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(enable=True, enable_train=False, detach_encoder=True)

    with pytest.raises(ValueError, match="enable_train=False"):
        mtp_support.configure_native_hybrid_mtp(provider, mtp_config, {})


def test_configure_native_hybrid_mtp_requires_supported_versions(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_recent_mcore_test",
        "verl/models/mcore/mtp_support.py",
    )

    def reject_old_versions():
        raise RuntimeError("Megatron-Bridge>=0.6.0 and Megatron-Core>=0.19.0 are required")

    monkeypatch.setattr(mtp_support, "require_native_mtp_versions", reject_old_versions)
    provider = SimpleNamespace(is_hybrid_model=True)
    mtp_config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=True)

    with pytest.raises(RuntimeError, match="Megatron-Bridge>=0.6.0"):
        mtp_support.configure_native_hybrid_mtp(provider, mtp_config, {})


def test_gpt_provider_with_hybrid_pattern_is_not_misclassified(monkeypatch):
    mtp_support = _load_module(
        "mtp_support_gpt_provider_test",
        "verl/models/mcore/mtp_support.py",
    )
    provider = SimpleNamespace(is_hybrid_model=False, hybrid_layer_pattern="M*")
    mtp_config = SimpleNamespace(enable=True, enable_train=False, detach_encoder=True)
    overrides = {}

    configured = mtp_support.configure_native_hybrid_mtp(provider, mtp_config, overrides)

    assert configured is False
    assert overrides == {}


def test_native_hybrid_mtp_uses_input_ids_targets_and_raw_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch)
    packed_seq_params = SimpleNamespace()
    preprocess_calls = []

    def preprocess_thd(value, *, need_roll=False, **kwargs):
        preprocess_calls.append((value, need_roll, kwargs))
        values = value.values() if value.is_nested else value.reshape(-1)
        if need_roll:
            values = torch.roll(values, shifts=-1, dims=0)
        position_ids = torch.arange(values.shape[0], dtype=torch.long)
        return values.unsqueeze(0), packed_seq_params, position_ids.unsqueeze(0)

    model_forward.preprocess_thd_no_padding = preprocess_thd
    model_forward.postprocess_thd_no_padding = lambda output, *args, **kwargs: output

    input_ids = _nested(
        [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ]
    )
    response_loss_mask = _nested(
        [
            torch.tensor([True, False]),
            torch.tensor([True]),
        ]
    )
    temperature = _nested([torch.ones(3), torch.ones(2)])

    class FakeHybridModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.zeros(1, kwargs["input_ids"].shape[1], 8)

    model = FakeHybridModel()
    processor_args = {}

    def logits_processor(logits, **kwargs):
        processor_args.update(kwargs)
        return {"log_probs": torch.zeros_like(kwargs["label"], dtype=torch.float32)}

    output = model_forward.gptmodel_forward_no_padding(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": temperature,
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="thd",
        mtp_enable_train=True,
    )

    assert model.forward_kwargs["labels"] is None
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[False, True, False, False, True]]),
    )
    assert preprocess_calls[0][2]["include_total_tokens"] is True
    assert processor_args["label"].shape == (1, 5)
    assert output["log_probs"].shape == (1, 5)


def test_native_hybrid_mtp_bshd_uses_no_labels_and_unshifted_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch)
    preprocess_calls = []

    def preprocess_bshd(value, *, need_roll=False, **kwargs):
        dense = value.to_padded_tensor(0)
        if need_roll:
            dense = torch.roll(dense, shifts=-1, dims=1)
        attention_mask = torch.arange(dense.shape[1]).unsqueeze(0) < value.offsets().diff().unsqueeze(1)
        position_ids = torch.arange(dense.shape[1], dtype=torch.long).unsqueeze(0).expand_as(dense)
        preprocess_calls.append((value, need_roll, dense.clone()))
        return dense, attention_mask, position_ids

    model_forward.preprocess_bshd_no_padding = preprocess_bshd
    model_forward.postprocess_bshd_no_padding = lambda output, *args, **kwargs: output

    input_ids = _nested(
        [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ]
    )
    response_loss_mask = _nested(
        [
            torch.tensor([True, False]),
            torch.tensor([True]),
        ]
    )

    class FakeHybridModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            batch, sequence = kwargs["input_ids"].shape
            return torch.zeros(batch, sequence, 8)

    model = FakeHybridModel()
    processor_args = {}

    def logits_processor(logits, **kwargs):
        processor_args.update(kwargs)
        return {"log_probs": torch.zeros_like(kwargs["label"], dtype=torch.float32)}

    output = model_forward.gptmodel_forward_no_padding(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": _nested([torch.ones(3), torch.ones(2)]),
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="bshd",
        mtp_enable_train=True,
    )

    assert model.forward_kwargs["labels"] is None
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[False, True, False], [False, True, False]]),
    )
    assert [need_roll for _, need_roll, _ in preprocess_calls] == [False, False, True, False]
    torch.testing.assert_close(
        processor_args["label"],
        torch.tensor([[11, 12, 10], [21, 0, 20]]),
    )
    assert "loss_mask" not in processor_args
    assert output["log_probs"].shape == (2, 3)


def test_legacy_gpt_mtp_keeps_shifted_labels_and_loss_mask(monkeypatch):
    model_forward = _load_model_forward_with_stubs(monkeypatch, native_hybrid=False)
    packed_seq_params = SimpleNamespace()
    preprocess_calls = []

    def preprocess_thd(value, *, need_roll=False, **kwargs):
        preprocess_calls.append(need_roll)
        values = value.values() if value.is_nested else value.reshape(-1)
        if need_roll:
            values = torch.roll(values, shifts=-1, dims=0)
        position_ids = torch.arange(values.shape[0], dtype=torch.long)
        return values.unsqueeze(0), packed_seq_params, position_ids.unsqueeze(0)

    model_forward.preprocess_thd_no_padding = preprocess_thd
    model_forward.postprocess_thd_no_padding = lambda output, *args, **kwargs: output

    input_ids = _nested(
        [
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21], dtype=torch.long),
        ]
    )
    response_loss_mask = _nested(
        [
            torch.tensor([True, False]),
            torch.tensor([True]),
        ]
    )

    class FakeGPTModel:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.forward_kwargs = None

        def __call__(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.zeros(1, kwargs["input_ids"].shape[1], 8)

    model = FakeGPTModel()
    processor_args = {}

    def logits_processor(logits, **kwargs):
        processor_args.update(kwargs)
        return {"log_probs": torch.zeros_like(kwargs["label"]).float()}

    model_forward.gptmodel_forward_no_padding(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": input_ids.clone(),
            "temperature": _nested([torch.ones(3), torch.ones(2)]),
            "loss_mask": response_loss_mask,
            "response_attention_mask": None,
        },
        data_format="thd",
        mtp_enable_train=True,
    )

    torch.testing.assert_close(
        model.forward_kwargs["labels"],
        torch.tensor([[11, 12, 20, 21, 10]]),
    )
    torch.testing.assert_close(
        model.forward_kwargs["loss_mask"],
        torch.tensor([[True, False, False, True, False]]),
    )
    torch.testing.assert_close(
        processor_args["label"],
        torch.tensor([[11, 12, 20, 21, 10]]),
    )
    assert "loss_mask" not in processor_args
    assert preprocess_calls == [False, True, True, True, False]
