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

"""Focused CPU contracts for the v0.7 V1 Megatron R3 backport."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Lightweight CPU jobs do not install Megatron-Core. Stub only the import
# surface needed by the router replay modules; a real installation wins.
for _module in (
    "megatron",
    "megatron.core",
    "megatron.core.pipeline_parallel",
    "megatron.core.pipeline_parallel.schedules",
    "megatron.core.pipeline_parallel.utils",
    "megatron.core.tensor_parallel",
    "megatron.core.transformer",
    "megatron.core.transformer.moe",
    "megatron.core.transformer.moe.moe_utils",
    "megatron.core.transformer.moe.router",
    "megatron.core.transformer.moe.token_dispatcher",
    "megatron.core.transformer.transformer_config",
    "megatron.core.transformer.transformer_layer",
    "verl.models.mcore.util",
):
    sys.modules.setdefault(_module, MagicMock())


from verl.utils.megatron import router_replay_utils  # noqa: E402
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction  # noqa: E402


def _nested(rows, *, dtype=torch.long):
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def _rows(value):
    return [part.tolist() for part in value.unbind()]


def test_r3_alignment_and_mask_ignore_only_the_unused_final_row():
    input_ids = _nested([[1, 2, 3, 4, 5], [6, 7, 8]])
    routes = torch.nested.as_nested_tensor(
        [torch.ones(4, 2, 2, dtype=torch.int16), torch.full((3, 2, 2), 2, dtype=torch.int16)],
        layout=torch.jagged,
    )

    aligned = router_replay_utils.align_r3_router_replay_data(routes, input_ids)
    aligned_parts = list(aligned.unbind())
    assert [part.shape[0] for part in aligned_parts] == [5, 3]
    assert aligned_parts[0][-1].eq(0).all()

    replay_mask = router_replay_utils.build_r3_replay_mask(
        input_ids,
        torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
    )
    assert _rows(replay_mask) == [[True, True, True, True, False], [True, True, False]]


def test_r3_alignment_rejects_more_than_one_missing_route_row():
    input_ids = _nested([[1, 2, 3, 4]])
    routes = torch.nested.as_nested_tensor([torch.ones(2, 1, 1, dtype=torch.int16)], layout=torch.jagged)

    with pytest.raises(RuntimeError, match="expected equal lengths or exactly one missing final route"):
        router_replay_utils.align_r3_router_replay_data(routes, input_ids)


def test_replay_mask_keeps_the_final_row_on_megatron_native_routing():
    replay = RouterReplay()
    replay.set_target_indices(
        torch.tensor([[0, 1], [0, 1], [0, 1]]),
        replay_mask=torch.tensor([True, True, False]),
    )
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])

    _, indices = replay.get_replay_topk(
        scores,
        2,
        default_compute_topk=lambda values, topk, **_: torch.topk(values, k=topk, dim=1),
    )

    assert indices[:2].tolist() == [[0, 1], [0, 1]]
    assert indices[2].tolist() == [3, 2]


class _FakeTopKRouter(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.router_replay = RouterReplay()


def test_targets_and_actions_reach_live_decoder_routers_but_not_mtp(monkeypatch):
    monkeypatch.setattr(router_replay_utils, "TopKRouter", _FakeTopKRouter)
    monkeypatch.setattr(router_replay_utils, "device_name", "cpu")
    monkeypatch.setattr(router_replay_utils, "preprocess_packed_seqs", lambda value, *_args, **_kwargs: (value, None))
    monkeypatch.setattr(router_replay_utils, "scatter_to_sequence_parallel_region", lambda value: value)
    RouterReplay.router_instances.clear()
    orphans = [RouterReplay(), RouterReplay()]

    model = torch.nn.Module()
    model.decoder = torch.nn.ModuleList([_FakeTopKRouter(1), _FakeTopKRouter(2)])
    model.mtp = torch.nn.ModuleList([_FakeTopKRouter(1)])
    routes = torch.zeros(1, 3, 2, 1, dtype=torch.int16)
    routes[:, :, 1, :] = 1
    config = SimpleNamespace(num_layers=2, moe_layer_freq=1)

    router_replay_utils.set_router_replay_data(routes, None, config, model=model)
    router_replay_utils.set_model_router_replay_action(model, RouterReplayAction.REPLAY_BACKWARD)

    assert model.decoder[0].router_replay.target_topk_idx.eq(0).all()
    assert model.decoder[1].router_replay.target_topk_idx.eq(1).all()
    assert all(item.target_topk_idx is None for item in orphans)
    assert model.mtp[0].router_replay.target_topk_idx is None
    assert all(
        layer.router_replay.router_replay_action == RouterReplayAction.REPLAY_BACKWARD for layer in model.decoder
    )
    assert model.mtp[0].router_replay.router_replay_action is None

    RouterReplay.router_instances.clear()


def test_full_layer_routes_map_sparse_moe_layers_by_global_layer_number(monkeypatch):
    monkeypatch.setattr(router_replay_utils, "TopKRouter", _FakeTopKRouter)
    monkeypatch.setattr(router_replay_utils, "device_name", "cpu")
    monkeypatch.setattr(router_replay_utils, "preprocess_packed_seqs", lambda value, *_args, **_kwargs: (value, None))
    monkeypatch.setattr(router_replay_utils, "scatter_to_sequence_parallel_region", lambda value: value)
    RouterReplay.router_instances.clear()

    moe_layer_numbers = [index + 1 for index in range(52) if index % 2 == 1][:23]
    moe_layer_freq = [int(index + 1 in moe_layer_numbers) for index in range(52)]
    model = torch.nn.Module()
    model.decoder = torch.nn.ModuleList([_FakeTopKRouter(layer_number) for layer_number in moe_layer_numbers])
    routes = torch.arange(52, dtype=torch.int16).view(1, 1, 52, 1).expand(1, 3, 52, 1).clone()
    config = SimpleNamespace(num_layers=52, moe_layer_freq=moe_layer_freq)

    router_replay_utils.set_router_replay_data(routes, None, config, model=model)

    assert [int(router.router_replay.target_topk_idx[0, 0]) for router in model.decoder] == [
        layer_number - 1 for layer_number in moe_layer_numbers
    ]

    with pytest.raises(RuntimeError, match="route-layer dimension"):
        router_replay_utils.set_router_replay_data(routes[:, :, :51], None, config, model=model)

    RouterReplay.router_instances.clear()
