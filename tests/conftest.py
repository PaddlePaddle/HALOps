# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Shared pytest configuration for HALOps operator tests."""

import random

import numpy as np
import pytest
import torch

from tests.utils.device import skip_reason


SEED = 2026


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: fast subset of the mega_mla cases")
    if torch.cuda.is_available():
        torch.set_default_device("cuda")


def pytest_collection_modifyitems(config, items):
    reason = skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _cuda_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
