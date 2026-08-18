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

"""Source-contract tests for the vLLM router-replay version guard.

The async server imports vLLM, Ray, and torch-backed verl modules at import
time. Extracting the narrow guard from its AST keeps this test runnable in the
CPU-only test image while still exercising the guard's actual implementation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging import version

SERVER_PATH = (
    Path(__file__).resolve().parents[4] / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
)


def _is_routing_replay_condition(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "enable_rollout_routing_replay"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    )


def _load_routing_replay_block(installed_version: str):
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))
    server_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServerBase"
    )
    launch_server = next(
        node for node in server_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "launch_server"
    )
    replay_block = next(
        node for node in ast.walk(launch_server) if isinstance(node, ast.If) and _is_routing_replay_condition(node.test)
    )

    function = ast.FunctionDef(
        name="apply_routing_replay_config",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="args")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[replay_block, ast.Return(value=ast.Name(id="args", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "_VLLM_VERSION": version.parse(installed_version),
        "is_mtp_rollout_enabled": lambda config, model_config: config.mtp_rollout_enabled,
        "version": version,
        "vllm": SimpleNamespace(__version__=installed_version),
    }
    exec(compile(module, str(SERVER_PATH), "exec"), namespace)
    return namespace["apply_routing_replay_config"]


def _server_with_routing_replay(enabled: bool, mtp_rollout_enabled: bool = False):
    return SimpleNamespace(
        config=SimpleNamespace(
            enable_rollout_routing_replay=enabled,
            mtp_rollout_enabled=mtp_rollout_enabled,
        ),
        model_config=SimpleNamespace(),
    )


def test_router_replay_rejects_vllm_older_than_022():
    apply_config = _load_routing_replay_block("0.21.1")

    with pytest.raises(RuntimeError, match=r"requires vLLM >= 0\.22\.0 .*installed: 0\.21\.1"):
        apply_config(_server_with_routing_replay(True), {})


def test_router_replay_accepts_vllm_022_and_preserves_disabled_behavior():
    apply_supported_config = _load_routing_replay_block("0.22.0")
    assert apply_supported_config(_server_with_routing_replay(True), {}) == {"enable_return_routed_experts": True}

    apply_old_config = _load_routing_replay_block("0.21.1")
    assert apply_old_config(_server_with_routing_replay(False), {}) == {}


def test_router_replay_with_mtp_speculation_requires_vllm_026():
    apply_old_config = _load_routing_replay_block("0.25.1")
    with pytest.raises(RuntimeError, match=r"MTP speculative rollout with router replay requires vLLM >= 0\.26\.0"):
        apply_old_config(_server_with_routing_replay(True, mtp_rollout_enabled=True), {})

    assert apply_old_config(_server_with_routing_replay(True), {}) == {"enable_return_routed_experts": True}

    apply_supported_config = _load_routing_replay_block("0.26.0")
    assert apply_supported_config(_server_with_routing_replay(True, mtp_rollout_enabled=True), {}) == {
        "enable_return_routed_experts": True
    }
