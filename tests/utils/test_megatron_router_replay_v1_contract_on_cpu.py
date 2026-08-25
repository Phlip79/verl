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

"""No-dependency source contracts for the V1 Megatron R3 backport."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
ENGINE_PATH = REPO_ROOT / "verl/workers/engine/megatron/transformer_impl.py"
ENGINE_CONFIG_PATH = REPO_ROOT / "verl/workers/config/engine.py"
ENGINE_WORKER_PATH = REPO_ROOT / "verl/workers/engine_workers.py"
PADDING_PATH = REPO_ROOT / "verl/workers/utils/padding.py"
LAUNCHER_PATH = REPO_ROOT / "examples/grpo_trainer/run_nemotron_3_5_lightning_30b_a3b_megatron.sh"


def _class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )


def test_engine_config_exposes_v1_router_replay():
    source = ENGINE_CONFIG_PATH.read_text(encoding="utf-8")
    assert "class EngineRouterReplayConfig" in source
    assert "router_replay: EngineRouterReplayConfig" in source


def test_worker_enables_replay_only_for_actor_compute_and_update():
    source = ENGINE_WORKER_PATH.read_text(encoding="utf-8")
    assert "self.config.actor.megatron.router_replay.mode" in source
    assert source.count("@_with_routing_replay_flag(enabled=True)") == 2
    assert source.count("@_with_routing_replay_flag(enabled=False)") == 1


def test_worker_serializes_grad_norm_as_a_host_scalar():
    postprocess = ast.unparse(_class_method(ENGINE_WORKER_PATH, "TrainingWorker", "_postprocess_output"))

    assert "isinstance(grad_norm, torch.Tensor)" in postprocess
    assert "grad_norm.detach().item()" in postprocess


def test_engine_replays_live_decoder_routes_and_always_clears_state():
    forward_backward = ast.unparse(_class_method(ENGINE_PATH, "MegatronEngine", "forward_backward_batch"))
    forward_step = ast.unparse(_class_method(ENGINE_PATH, "MegatronEngineWithLMHead", "forward_step"))

    assert "RouterReplay.set_global_router_replay_action" in forward_backward
    assert "finally:" in forward_backward
    assert "RouterReplay.clear_global_indices()" in forward_backward
    assert "RouterReplay.clear_global_router_replay_action()" in forward_backward
    assert "align_r3_router_replay_data" in forward_step
    assert "build_r3_replay_mask" in forward_step
    assert "model=unwrapped_model" in forward_step
    assert "set_model_router_replay_action" in forward_step
    assert 'override_transformer_config["moe_router_fusion"] = False' in ENGINE_PATH.read_text(encoding="utf-8")
    assert "virtual_pipeline_model_parallel_size is not None" in ENGINE_PATH.read_text(encoding="utf-8")


def test_engine_uses_the_matching_bridge_weight_export_api():
    export_source = ast.unparse(_class_method(ENGINE_PATH, "MegatronEngine", "get_per_tensor_param"))

    assert "if self.vanilla_bridge:" in export_source
    assert "self.bridge.export_weights(self.module)" in export_source
    assert "self.bridge.export_hf_weights(self.module)" in export_source


def test_route_transport_and_launcher_keep_parity_guards():
    padding_source = PADDING_PATH.read_text(encoding="utf-8")
    launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
    replay_utils_source = (REPO_ROOT / "verl/utils/megatron/router_replay_utils.py").read_text(encoding="utf-8")

    assert 'data.get("routed_experts")' in padding_source
    assert "index_first_axis" in padding_source
    assert "routed_experts.to(torch.int16)" in padding_source
    assert "route_layer_count == tf_config.num_layers" in replay_utils_source
    assert "route_layer_count == moe_layer_count" in replay_utils_source
    for expected in (
        "export VLLM_BATCH_INVARIANT=0",
        "MTP_ROLLOUT_SPEC=${MTP_ROLLOUT_SPEC:-1}",
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}",
        "ROLLOUT_N=${ROLLOUT_N:-2}",
        "data.seed=${SEED}",
        "actor_rollout_ref.actor.data_loader_seed=${SEED}",
        "actor_rollout_ref.actor.megatron.seed=${SEED}",
        "actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}",
        "actor_rollout_ref.rollout.enable_rollout_routing_replay=${ROLLOUT_ROUTING_REPLAY_ENABLED}",
        "actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}",
        "actor_rollout_ref.rollout.logprobs_mode=raw_logprobs",
        "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}",
        "+actor_rollout_ref.rollout.repetition_penalty=${ROLLOUT_REPETITION_PENALTY}",
        "override_transformer_config.moe_router_fusion=False",
        "R3 raw-logprob parity requires ROLLOUT_TEMPERATURE=1.0",
    ):
        assert expected in launcher_source
