# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_TOKENIZER_PATH = Path(__file__).parents[2] / "verl/utils/tokenizer.py"
_SPEC = importlib.util.spec_from_file_location("verl_tokenizer_under_test", _TOKENIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOKENIZER_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOKENIZER_MODULE)
normalize_token_ids = _TOKENIZER_MODULE.normalize_token_ids


class DummyBatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class DummyToList:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


@pytest.mark.parametrize(
    ("tokenized_output", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        (DummyToList([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3], dtype=np.int64), [1, 2, 3]),
        ({"input_ids": [1, 2, 3]}, [1, 2, 3]),
        ({"input_ids": DummyToList([1, 2, 3])}, [1, 2, 3]),
        ({"input_ids": [[1, 2, 3]]}, [1, 2, 3]),
        (DummyBatchEncoding([1, 2, 3]), [1, 2, 3]),
        (DummyBatchEncoding(DummyToList([[1, 2, 3]])), [1, 2, 3]),
        ([np.int64(1), np.int32(2), np.int16(3)], [1, 2, 3]),
    ],
)
def test_normalize_token_ids_valid_outputs(tokenized_output, expected):
    assert normalize_token_ids(tokenized_output) == expected


@pytest.mark.parametrize(
    "tokenized_output",
    [
        "not-token-ids",
        {"attention_mask": [1, 1, 1]},
        [[1, 2], [3, 4]],
        [1, object(), 3],
    ],
)
def test_normalize_token_ids_invalid_outputs(tokenized_output):
    with pytest.raises(TypeError):
        normalize_token_ids(tokenized_output)
