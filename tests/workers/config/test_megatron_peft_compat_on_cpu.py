# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only compatibility tests for Megatron-Bridge PEFT construction."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[3] / "verl/workers/config/megatron_peft.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_megatron_peft_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_peft_modules(monkeypatch, *, with_factory=False):
    for name in ("megatron", "megatron.bridge", "megatron.bridge.peft"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    classes = {}
    for module_name, class_names in {
        "canonical_lora": ("CanonicalLoRA",),
        "dora": ("DoRA",),
        "lora": ("LoRA", "VLMLoRA"),
    }.items():
        peft_module = types.ModuleType(f"megatron.bridge.peft.{module_name}")
        for class_name in class_names:
            peft_cls = type(
                class_name,
                (),
                {"__init__": lambda self, **kwargs: setattr(self, "kwargs", kwargs)},
            )
            setattr(peft_module, class_name, peft_cls)
            classes[class_name] = peft_cls
        monkeypatch.setitem(sys.modules, peft_module.__name__, peft_module)

    if with_factory:
        utils = types.ModuleType("megatron.bridge.peft.utils")
        utils.create_peft = lambda config, dtype=None: (config, dtype)
        monkeypatch.setitem(sys.modules, utils.__name__, utils)
    else:
        monkeypatch.delitem(sys.modules, "megatron.bridge.peft.utils", raising=False)
    return classes


def test_get_peft_cls_prefers_new_bridge_factory(monkeypatch):
    _install_peft_modules(monkeypatch, with_factory=True)
    module = _load_module()
    config = {"rank": 8, "type": "lora"}

    assert module.get_peft_cls(SimpleNamespace(lora=config), object(), object()) == (config, None)


@pytest.mark.parametrize(
    ("lora_type", "class_name", "expected_target"),
    [
        ("lora", "LoRA", "linear_qkv"),
        ("vlm_lora", "VLMLoRA", "linear_qkv"),
        ("canonical_lora", "CanonicalLoRA", "linear_q"),
        ("dora", "DoRA", "linear_qkv"),
    ],
)
def test_get_peft_cls_falls_back_to_legacy_bridge_constructors(monkeypatch, lora_type, class_name, expected_target):
    classes = _install_peft_modules(monkeypatch)
    module = _load_module()

    result = module.get_peft_cls(
        SimpleNamespace(lora={"rank": 4, "type": lora_type}),
        object(),
        object(),
    )

    assert isinstance(result, classes[class_name])
    assert result.kwargs["dim"] == 4
    assert expected_target in result.kwargs["target_modules"]
