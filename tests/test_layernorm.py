# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for unsloth's fused LayerNorm Triton kernel.

Lifted from the in-file `test_layernorm` / `testing_suite_layernorm` helpers
that used to live in `unsloth/kernels/layernorm.py`, now pytest-discoverable.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level = True)

if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
    pytest.skip("requires a CUDA GPU to run Triton kernels", allow_module_level = True)


@pytest.mark.parametrize("dim", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [349, 2048, 3341])
@pytest.mark.parametrize("random_state", [3407, 42])
def test_layernorm_matches_torch_reference(dim, dtype, seqlen, random_state):
    """Build torch.nn.LayerNorm with random uniform weight + bias, run
    reference forward+backward, run unsloth's fast_layernorm, and check
    the input gradient matches within tolerance."""
    from torch.nn import LayerNorm

    from unsloth.kernels.layernorm import fast_layernorm

    eps = 1e-5
    bsz = 21

    layernorm = LayerNorm((dim,), eps = eps, device = "cuda", dtype = dtype)
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    torch.nn.init.uniform_(layernorm.bias)

    with torch.autocast(device_type = "cuda", dtype = dtype):
        X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
        XX = X.clone()
        X.requires_grad_(True)
        XX.requires_grad_(True)
        Y = layernorm(X)
        YY = torch.randn(
            (bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True,
        )
        Y.backward(YY)
        correct_grad = X.grad.clone()
        Y2 = fast_layernorm(layernorm, XX)
        Y2.backward(YY)

    assert torch.dist(correct_grad, XX.grad).item() <= 0.1
