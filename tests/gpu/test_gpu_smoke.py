"""GPU integration tests (run on the pod with KDEV_GPU_TESTS=1). Each package adds tests/gpu/test_<pkg>.py."""
from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_torch_cuda_available():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
