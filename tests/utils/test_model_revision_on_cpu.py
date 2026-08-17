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

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

MODEL_CONFIG_PATH = Path(__file__).parents[2] / "verl/workers/config/model.py"


def _load_resolve_model_path(copy_to_local):
    tree = ast.parse(MODEL_CONFIG_PATH.read_text())
    config_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HFModelConfig")
    method = next(
        node for node in config_class.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_model_path"
    )
    module = ast.Module(
        body=[ast.ClassDef(name="HFModelConfig", bases=[], keywords=[], body=[method], decorator_list=[])],
        type_ignores=[],
    )
    module = ast.fix_missing_locations(module)
    namespace = {"copy_to_local": copy_to_local, "os": os}
    exec(compile(module, str(MODEL_CONFIG_PATH), "exec"), namespace)
    return namespace["HFModelConfig"]._resolve_model_path


def test_remote_model_revision_resolves_to_immutable_snapshot(monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or "/cache/immutable-snapshot"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    copied = []
    resolve = _load_resolve_model_path(lambda path, use_shm=False: copied.append((path, use_shm)) or path)
    result = resolve(SimpleNamespace(path="nvidia/model", revision="abc123", use_shm=True))

    assert calls == [{"repo_id": "nvidia/model", "revision": "abc123"}]
    assert copied == [("/cache/immutable-snapshot", True)]
    assert result == "/cache/immutable-snapshot"


def test_local_and_hdfs_model_paths_do_not_use_hugging_face(monkeypatch, tmp_path):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    copied = []
    resolve = _load_resolve_model_path(lambda path, use_shm=False: copied.append((path, use_shm)) or path)
    local_path = tmp_path / "model"
    local_path.mkdir()

    assert resolve(SimpleNamespace(path=str(local_path), revision="abc123", use_shm=False)) == str(local_path)
    assert resolve(SimpleNamespace(path="hdfs://models/model", revision="abc123", use_shm=False)) == (
        "hdfs://models/model"
    )
    assert copied == [(str(local_path), False), ("hdfs://models/model", False)]
