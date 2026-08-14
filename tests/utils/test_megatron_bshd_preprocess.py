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
"""Regression coverage for Hybrid/Mamba packed-sequence metadata."""

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_mcore_util_with_stubbed_megatron(
    monkeypatch,
    tp_size: int = 1,
    cp_size: int = 1,
    cp_rank: int = 0,
):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    model_module = types.ModuleType("verl.utils.model")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_context_parallel_group = lambda: object()
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.seq_idx = None
            cu_seqlens = kwargs.get("cu_seqlens_q_padded")
            if cu_seqlens is None:
                cu_seqlens = kwargs.get("cu_seqlens_q")
            total_tokens = kwargs.get("total_tokens")
            if isinstance(cu_seqlens, torch.Tensor) and total_tokens is not None:
                cu_seqlens_with_total = torch.cat(
                    [
                        cu_seqlens,
                        torch.tensor([total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device),
                    ]
                )
                seq_lengths = (cu_seqlens_with_total[1:] - cu_seqlens_with_total[:-1]).clamp(min=0)
                self.seq_idx = (
                    torch.repeat_interleave(torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths)
                    .to(torch.int32)
                    .unsqueeze(0)
                )

    packed_seq_params.PackedSeqParams = PackedSeqParams
    model_module.CausalLMOutputForPPO = object

    core.parallel_state = parallel_state
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setitem(sys.modules, "verl.utils.model", model_module)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_native_mtp_test", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_preprocess_thd_no_padding_builds_packed_sequence_indices(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    input_ids = _nested_tensor(
        [
            torch.tensor([11, 12, 13], dtype=torch.long),
            torch.tensor([21, 22], dtype=torch.long),
        ]
    )

    local_ids, packed_seq_params, _ = mcore_util.preprocess_thd_no_padding(
        input_ids,
        include_total_tokens=True,
    )

    assert local_ids.shape == (1, 5)
    assert packed_seq_params.total_tokens == 5
    torch.testing.assert_close(
        packed_seq_params.seq_idx,
        torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
    )


def test_preprocess_thd_no_padding_builds_global_sequence_indices_with_context_parallelism(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2)
    input_ids = _nested_tensor(
        [
            torch.tensor([11, 12, 13, 14], dtype=torch.long),
            torch.tensor([21, 22, 23, 24, 25, 26, 27, 28], dtype=torch.long),
        ]
    )

    local_ids, packed_seq_params, _ = mcore_util.preprocess_thd_no_padding(
        input_ids,
        include_total_tokens=True,
    )

    assert local_ids.shape == (1, 6)
    assert packed_seq_params.total_tokens == 12
    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 4, 12]
    torch.testing.assert_close(
        local_ids,
        torch.tensor([[11, 14, 21, 22, 27, 28]], dtype=torch.long),
    )
    torch.testing.assert_close(
        packed_seq_params.seq_idx,
        torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32),
    )


def test_preprocess_thd_no_padding_uses_valid_positions_for_ragged_context_parallel_chunk(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, cp_size=2, cp_rank=1)
    input_ids = _nested_tensor([torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)])

    local_ids, _, position_ids = mcore_util.preprocess_thd_no_padding(
        input_ids,
        include_total_tokens=True,
    )

    torch.testing.assert_close(local_ids, torch.tensor([[13, 14, 15, 0]], dtype=torch.long))
    torch.testing.assert_close(position_ids, torch.tensor([[2, 3, 4, 0]], dtype=torch.long))


def test_preprocess_thd_no_padding_omits_mamba_metadata_by_default(monkeypatch):
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    input_ids = _nested_tensor([torch.tensor([11, 12, 13], dtype=torch.long)])

    _, packed_seq_params, _ = mcore_util.preprocess_thd_no_padding(input_ids)

    assert getattr(packed_seq_params, "total_tokens", None) is None
    assert packed_seq_params.seq_idx is None
