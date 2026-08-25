# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only source contracts for V1 Megatron PEFT checkpointing."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).parents[2]
ENGINE_PATH = REPO_ROOT / "verl/workers/engine/megatron/transformer_impl.py"
CHECKPOINT_MANAGER_PATH = REPO_ROOT / "verl/utils/checkpoint/megatron_checkpoint_manager.py"
PEFT_UTILS_PATH = REPO_ROOT / "verl/utils/megatron_peft_utils.py"


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _load_peft_utils():
    spec = importlib.util.spec_from_file_location("_megatron_peft_utils_under_test", PEFT_UTILS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_megatron_modules(*, expert_dp_rank: int, context_rank: int):
    megatron = types.ModuleType("megatron")
    megatron.__path__ = []
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_expert_data_parallel_rank=lambda: expert_dp_rank,
        get_context_parallel_rank=lambda: context_rank,
    )
    return {"megatron": megatron, "megatron.core": core}


def test_v1_engine_forwards_peft_to_checkpoint_manager():
    initialize = _method(ENGINE_PATH, "MegatronEngine", "initialize")
    manager_call = next(
        node
        for node in ast.walk(initialize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MegatronCheckpointManager"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in manager_call.keywords}

    assert keywords["peft_cls"] == "self.peft_cls"


def test_peft_checkpoint_does_not_also_export_full_hf_weights():
    save_checkpoint = _method(CHECKPOINT_MANAGER_PATH, "MegatronCheckpointManager", "save_checkpoint")
    hf_export_guard = next(
        node
        for node in ast.walk(save_checkpoint)
        if isinstance(node, ast.If)
        and "self.use_hf_checkpoint" in ast.unparse(node.test)
        and "self.peft_cls" in ast.unparse(node.test)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"save_weights", "save_hf_weights"}
            for child in ast.walk(node)
        )
    )

    assert ast.unparse(hf_export_guard.test) == (
        "self.use_hf_checkpoint and (self.peft_cls is None or self.should_save_hf_model)"
    )


def test_only_one_dp_cp_replica_writes_each_adapter_shard():
    module = _load_peft_utils()

    for expert_dp_rank, context_rank, expected in (
        (0, 0, True),
        (1, 0, False),
        (0, 1, False),
        (1, 1, False),
    ):
        fake_modules = _fake_megatron_modules(
            expert_dp_rank=expert_dp_rank,
            context_rank=context_rank,
        )
        with patch.dict(sys.modules, fake_modules):
            assert module._is_adapter_checkpoint_writer() is expected


def test_adapter_shard_writers_synchronize_before_save_completes():
    save_checkpoint = _method(CHECKPOINT_MANAGER_PATH, "MegatronCheckpointManager", "save_checkpoint")
    calls = [node for node in ast.walk(save_checkpoint) if isinstance(node, ast.Call)]
    adapter_save = next(
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "save_adapter_checkpoint"
    )
    barrier = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and ast.unparse(node.func) == "torch.distributed.barrier"
        and node.lineno > adapter_save.lineno
    )

    assert barrier.lineno == adapter_save.end_lineno + 1


def test_transformer_config_checkpoint_neutralizes_runtime_process_groups():
    save_checkpoint = _method(CHECKPOINT_MANAGER_PATH, "MegatronCheckpointManager", "save_checkpoint")
    bypass_keys = next(
        node
        for node in ast.walk(save_checkpoint)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "bypass_keys" for target in node.targets)
    )

    assert "_pg_collection" in ast.literal_eval(bypass_keys.value)


def test_transformer_config_checkpoint_serializes_runtime_enums_as_strings():
    save_checkpoint = _method(CHECKPOINT_MANAGER_PATH, "MegatronCheckpointManager", "save_checkpoint")
    json_dump = next(
        node
        for node in ast.walk(save_checkpoint)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "json.dump"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in json_dump.keywords}

    assert keywords["default"] == "str"
