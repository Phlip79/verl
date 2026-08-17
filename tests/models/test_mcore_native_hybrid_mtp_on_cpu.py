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

"""CPU-only contract tests for the native HybridModel MTP integration.

These tests deliberately load the small MTP modules with a fake Megatron-Core
surface.  That keeps the behavioural contracts runnable in the normal CPU test
job, while the real model execution remains covered by the GPU integration
recipes.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
MTP_SUPPORT_PATH = REPO_ROOT / "verl/models/mcore/mtp_support.py"
MODEL_FORWARD_PATH = REPO_ROOT / "verl/models/mcore/model_forward.py"
MCORE_UTIL_PATH = REPO_ROOT / "verl/models/mcore/util.py"
MEGATRON_UTILS_PATH = REPO_ROOT / "verl/utils/megatron_utils.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_megatron_mtp_api(monkeypatch, *, accepts_input_ids: bool = True):
    """Install the narrow MCore API inspected by ``has_native_mtp_support``."""

    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    core = types.ModuleType("megatron.core")
    core.__path__ = []
    transformer = types.ModuleType("megatron.core.transformer")
    transformer.__path__ = []
    mtp = types.ModuleType("megatron.core.transformer.multi_token_prediction")

    class TransformerConfig:
        mtp_detach_heads = False

    if accepts_input_ids:

        def process_mtp_loss(input_ids, loss_mask):
            return input_ids, loss_mask

    else:

        def process_mtp_loss(loss_mask):
            return loss_mask

    transformer.TransformerConfig = TransformerConfig
    mtp.process_mtp_loss = process_mtp_loss
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.multi_token_prediction", mtp)


def _load_mtp_support(monkeypatch, *, accepts_input_ids: bool = True):
    _install_megatron_mtp_api(monkeypatch, accepts_input_ids=accepts_input_ids)
    return _load_module("_mtp_support_under_test", MTP_SUPPORT_PATH)


def _install_model_forward_dependencies(monkeypatch, *, is_native_hybrid_model=True):
    """Load model_forward without importing the full distributed stack."""

    package = types.ModuleType("_mcore_forward_test")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "_mcore_forward_test", package)

    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", megatron_utils)

    mtp_support = types.ModuleType("_mcore_forward_test.mtp_support")
    mtp_support.is_native_hybrid_model = lambda model: is_native_hybrid_model
    monkeypatch.setitem(sys.modules, "_mcore_forward_test.mtp_support", mtp_support)

    util = types.ModuleType("_mcore_forward_test.util")
    util.postprocess_bshd = lambda *args, **kwargs: args[0]
    util.postprocess_bshd_no_padding = lambda *args, **kwargs: args[0]
    util.postprocess_packed_seqs = lambda *args, **kwargs: args[0]
    util.postprocess_thd_no_padding = lambda *args, **kwargs: args[0]
    util.preprocess_bshd = lambda *args, **kwargs: args
    util.preprocess_bshd_no_padding = lambda *args, **kwargs: args
    util.preprocess_packed_seqs = lambda *args, **kwargs: args
    util.preprocess_thd_no_padding = lambda *args, **kwargs: args
    monkeypatch.setitem(sys.modules, "_mcore_forward_test.util", util)
    return _load_module("_mcore_forward_test.model_forward", MODEL_FORWARD_PATH), util


def _install_util_dependencies(monkeypatch, *, cp_size=1, cp_rank=0, tp_size=1):
    """Return the source util module with CPU fake parallel-state primitives."""

    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    core = types.ModuleType("megatron.core")
    core.__path__ = []
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank

    packed_seq = types.ModuleType("megatron.core.packed_seq_params")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    packed_seq.PackedSeqParams = PackedSeqParams
    core.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq)

    model = types.ModuleType("verl.utils.model")
    model.CausalLMOutputForPPO = type("CausalLMOutputForPPO", (), {})
    monkeypatch.setitem(sys.modules, "verl.utils.model", model)
    return _load_module("_mcore_util_under_test", MCORE_UTIL_PATH)


def _nested(values):
    return torch.nested.nested_tensor([torch.tensor(value) for value in values], layout=torch.jagged)


def _nested_rows(value):
    offsets = value.offsets().tolist()
    values = value.values()
    return [values[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def test_native_mtp_capability_probe_requires_new_mcore_api(monkeypatch):
    supported = _load_mtp_support(monkeypatch)
    assert supported.has_native_mtp_support()

    unsupported = _load_mtp_support(monkeypatch, accepts_input_ids=False)
    assert not unsupported.has_native_mtp_support()


def test_configure_native_hybrid_provider_enforces_train_and_detach(monkeypatch):
    mtp_support = _load_mtp_support(monkeypatch)

    class HybridModelProvider:
        pass

    provider = HybridModelProvider()
    overrides = {}
    config = SimpleNamespace(enable=True, enable_train=True, detach_encoder=True)
    assert mtp_support.configure_native_hybrid_mtp(provider, config, overrides)
    assert overrides == {"mtp_detach_heads": True}

    with pytest.raises(ValueError, match="enable_train=True"):
        mtp_support.configure_native_hybrid_mtp(
            provider,
            SimpleNamespace(enable=True, enable_train=False, detach_encoder=False),
            {},
        )

    assert not mtp_support.configure_native_hybrid_mtp(object(), config, {"unchanged": True})


def test_response_aware_mtp_mask_preserves_internal_zeroes(monkeypatch):
    model_forward, _ = _install_model_forward_dependencies(monkeypatch)
    # The first sample has a valid response token whose training weight is zero.
    # Its response span must still be three tokens, rather than being inferred
    # from loss_mask.sum().
    response_mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.float)
    response_attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    result = model_forward._build_mtp_loss_mask_nested(response_mask, [5, 4], response_attention_mask)
    assert _nested_rows(result) == [[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]


def test_native_thd_forward_keeps_mtp_mask_unshifted(monkeypatch):
    model_forward, _ = _install_model_forward_dependencies(monkeypatch)
    calls = []

    def preprocess(value, **kwargs):
        calls.append((value, kwargs))
        packed = SimpleNamespace()
        if kwargs.get("return_position_ids"):
            return value, packed, torch.tensor([[0, 1, 2]], dtype=torch.long)
        if kwargs.get("need_roll"):
            return "shifted-label", packed
        return value, packed

    # model_forward imports this helper by name, so replace that bound symbol
    # rather than mutating the source utility module after import.
    model_forward.preprocess_thd_no_padding = preprocess

    class Model:
        pre_process = True
        post_process = True

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.tensor([1.0])

    observed = {}

    def logits_processor(output, **kwargs):
        observed.update(kwargs)
        return {"log_probs": output}

    model = Model()
    input_ids = _nested([[10, 11, 12, 13, 14], [20, 21, 22, 23]])
    label = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 0]])
    loss_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)
    response_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    model_forward.gptmodel_forward_no_padding(
        model,
        input_ids,
        {},
        logits_processor=logits_processor,
        logits_processor_args={
            "label": label,
            "loss_mask": loss_mask,
            "response_attention_mask": response_attention_mask,
        },
        mtp_enable_train=True,
    )

    assert model.kwargs["labels"] is None
    assert model.kwargs["position_ids"].tolist() == [[0, 1, 2]]
    # Initial input packing requests total_tokens.  MCore then derives native
    # MTP labels from that unshifted token stream.
    assert calls[0][1]["include_total_tokens"] is True
    loss_calls = [call for call in calls if call[0].is_nested and call[1].get("need_roll") is False]
    assert len(loss_calls) == 1
    assert _nested_rows(loss_calls[0][0]) == [[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    # verl's regular log-prob loss remains the one-token-shifted label path.
    assert observed["label"] == "shifted-label"
    assert any(call[1].get("need_roll") is True for call in calls)


def test_thd_packing_sets_total_tokens_and_positions(monkeypatch):
    util = _install_util_dependencies(monkeypatch, tp_size=2)
    packed_ids, packed, position_ids = util.preprocess_thd_no_padding(
        _nested([[1, 2, 3], [4, 5]]),
        include_total_tokens=True,
        return_position_ids=True,
    )
    assert packed.total_tokens == 6  # lengths padded to 4 and 2 for TP=2
    assert packed_ids.tolist() == [[1, 2, 3, 0, 4, 5]]
    assert position_ids.tolist() == [[0, 1, 2, 0, 0, 1]]


@pytest.mark.parametrize(
    ("cp_rank", "expected_tokens", "expected_positions"),
    [
        (0, [[1, 2, 0, 0]], [[0, 1, 0, 0]]),
        (1, [[3, 4, 5, 0]], [[2, 3, 4, 0]]),
    ],
)
def test_thd_packing_cp_keeps_global_total_and_local_positions(
    monkeypatch, cp_rank, expected_tokens, expected_positions
):
    util = _install_util_dependencies(monkeypatch, cp_size=2, cp_rank=cp_rank)
    packed_ids, packed, position_ids = util.preprocess_thd_no_padding(
        _nested([[1, 2, 3, 4, 5]]),
        include_total_tokens=True,
        return_position_ids=True,
    )
    assert packed.total_tokens == 8
    assert packed_ids.tolist() == expected_tokens
    assert position_ids.tolist() == expected_positions


@pytest.mark.parametrize(
    ("cp_rank", "expected_tokens"),
    [
        (0, [[7, 0]]),
        (1, [[0, 0]]),
    ],
)
def test_thd_packing_cp_pads_sequences_shorter_than_alignment(monkeypatch, cp_rank, expected_tokens):
    util = _install_util_dependencies(monkeypatch, cp_size=2, cp_rank=cp_rank)
    packed_ids, packed, position_ids = util.preprocess_thd_no_padding(
        _nested([[7]]),
        include_total_tokens=True,
        return_position_ids=True,
    )
    assert packed.total_tokens == 4
    assert packed_ids.tolist() == expected_tokens
    assert position_ids.tolist() == [[0, 0]]


def _load_patch_engine_mtp():
    """Compile the source function alone to avoid importing all MCore utilities."""

    tree = ast.parse(MEGATRON_UTILS_PATH.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "patch_engine_mtp")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "unwrap_model": lambda model: model,
        "logger": SimpleNamespace(info=lambda *args, **kwargs: None),
    }
    exec(compile(module, str(MEGATRON_UTILS_PATH), "exec"), namespace)
    return namespace["patch_engine_mtp"]


def _load_check_mtp_config():
    """Compile the config helpers without importing the distributed stack."""

    tree = ast.parse(MEGATRON_UTILS_PATH.read_text())
    helper_names = {"_get_mtp_num_layers", "_set_mtp_num_layers", "check_mtp_config"}
    nodes = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name in helper_names]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {}
    exec(compile(module, str(MEGATRON_UTILS_PATH), "exec"), namespace)
    return namespace["check_mtp_config"]


def test_disabled_mtp_clears_generic_and_hybrid_provider_overrides(monkeypatch):
    check_mtp_config = _load_check_mtp_config()
    hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        mtp=SimpleNamespace(enable=False, mtp_loss_scaling_factor=0.3),
    )
    engine_config = SimpleNamespace(
        override_transformer_config={
            "mtp_num_layers": 2,
            "mtp_hybrid_override_pattern": "*E",
            "mtp_use_repeated_layer": True,
            "keep_mtp_spec_in_bf16": True,
            "mtp_loss_scaling_factor": 0.3,
        }
    )

    check_mtp_config(model_config, engine_config)

    assert hf_config.num_nextn_predict_layers == 0
    assert engine_config.override_transformer_config == {"mtp_num_layers": None}

    mtp_support = _load_mtp_support(monkeypatch)

    class HybridModelProvider:
        pass

    overrides = dict(engine_config.override_transformer_config)
    assert not mtp_support.configure_native_hybrid_mtp(HybridModelProvider(), SimpleNamespace(enable=False), overrides)
    assert overrides == {
        "mtp_num_layers": None,
        "mtp_hybrid_override_pattern": None,
        "mtp_use_repeated_layer": False,
        "keep_mtp_spec_in_bf16": False,
    }

    strict_gpt_overrides = {"mtp_num_layers": None}
    assert not mtp_support.configure_native_hybrid_mtp(object(), SimpleNamespace(enable=False), strict_gpt_overrides)
    assert strict_gpt_overrides == {"mtp_num_layers": None}


def test_native_patch_bypasses_legacy_and_rejects_other_models(monkeypatch):
    mtp_support = types.ModuleType("verl.models.mcore.mtp_support")
    mtp_support.is_native_hybrid_model = lambda model: model == "hybrid"
    monkeypatch.setitem(sys.modules, "verl.models.mcore.mtp_support", mtp_support)
    patch_engine_mtp = _load_patch_engine_mtp()

    config = SimpleNamespace(mtp=SimpleNamespace(enable_train=True))
    assert patch_engine_mtp(["hybrid"], config) is None

    with pytest.raises(NotImplementedError, match="native Megatron-Core HybridModel"):
        patch_engine_mtp("legacy-gpt", config)

    with pytest.raises(ValueError, match="enable_train=True"):
        patch_engine_mtp("hybrid", SimpleNamespace(mtp=SimpleNamespace(enable_train=False)))
