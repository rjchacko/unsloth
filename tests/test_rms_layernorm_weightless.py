# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for FlashNorm-style weightless RMSNorm support.

Covers:
- weight=None (truly weightless modules)
- weight=ones (HF convention for FlashNorm-folded checkpoints)
- weight=randn (negative case — must NOT trigger the weightless path)
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level = True)

if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
    pytest.skip("requires a CUDA GPU to run Triton kernels", allow_module_level = True)


def _ref_rmsnorm(x: torch.Tensor, eps: float, weight = None) -> torch.Tensor:
    """Reference RMSNorm in pure PyTorch float32, casts back to x.dtype."""
    xf = x.to(torch.float32)
    inv_rms = (xf.square().mean(-1, keepdim = True) + eps).rsqrt()
    y = xf * inv_rms
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(x.dtype)


@pytest.mark.parametrize("dim", [512, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weightless_none_matches_reference(dim, dtype):
    """When weight=None, fast_rms_layernorm output and gradient match the
    unweighted reference."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    eps = 1e-5
    bsz, seqlen = 4, 256

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    layernorm.weight = None  # truly weightless

    torch.manual_seed(3407)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    X_ref = X.clone().requires_grad_(True)
    X_test = X.clone().requires_grad_(True)
    grad_out = torch.randn_like(X)

    Y_ref = _ref_rmsnorm(X_ref, eps, weight = None)
    Y_ref.backward(grad_out)

    Y_test = fast_rms_layernorm(layernorm, X_test)
    Y_test.backward(grad_out)

    assert torch.amax((Y_test - Y_ref).abs()).item() <= 5e-2
    assert torch.amax((X_test.grad - X_ref.grad).abs()).item() <= 5e-2


@pytest.mark.parametrize("dim", [512, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weightless_ones_matches_reference_and_caches(dim, dtype):
    """When weight=all-ones, fast_rms_layernorm matches the unweighted
    reference and caches `_unsloth_weightless=True` after the first call."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    eps = 1e-5
    bsz, seqlen = 4, 256

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    with torch.no_grad():
        layernorm.weight.fill_(1.0)

    assert getattr(layernorm, "_unsloth_weightless", None) is None  # not yet cached

    torch.manual_seed(42)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    X_ref = X.clone().requires_grad_(True)
    X_test = X.clone().requires_grad_(True)
    grad_out = torch.randn_like(X)

    Y_ref = _ref_rmsnorm(X_ref, eps, weight = None)
    Y_ref.backward(grad_out)

    Y_test = fast_rms_layernorm(layernorm, X_test)
    Y_test.backward(grad_out)

    assert torch.amax((Y_test - Y_ref).abs()).item() <= 5e-2
    assert torch.amax((X_test.grad - X_ref.grad).abs()).item() <= 5e-2
    assert layernorm._unsloth_weightless is True


@pytest.mark.parametrize("dim", [1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_random_weight_does_not_trigger_weightless(dim, dtype):
    """Negative case: a random weight tensor must not be misidentified as
    weightless, and the kernel output must include the scale."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    eps = 1e-5
    bsz, seqlen = 4, 256

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    with torch.no_grad():
        layernorm.weight.copy_(torch.rand(dim, device = "cuda") + 0.5)  # in [0.5, 1.5)

    torch.manual_seed(2024)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    X_ref = X.clone().requires_grad_(True)
    X_test = X.clone().requires_grad_(True)
    grad_out = torch.randn_like(X)

    Y_ref = _ref_rmsnorm(X_ref, eps, weight = layernorm.weight)
    Y_ref.backward(grad_out)

    Y_test = fast_rms_layernorm(layernorm, X_test)
    Y_test.backward(grad_out)

    assert layernorm._unsloth_weightless is False
    assert torch.amax((Y_test - Y_ref).abs()).item() <= 5e-2
    assert torch.amax((X_test.grad - X_ref.grad).abs()).item() <= 5e-2


def test_is_weightless_caches_result():
    """The detection helper must cache its first result on the module."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import _is_weightless

    dim = 256
    layernorm = LlamaRMSNorm((dim,), eps = 1e-5).to("cuda")
    with torch.no_grad():
        layernorm.weight.fill_(1.0)

    assert getattr(layernorm, "_unsloth_weightless", None) is None
    assert _is_weightless(layernorm) is True
    assert layernorm._unsloth_weightless is True

    # Now mutate the underlying tensor — the cache should still report True
    # (we don't auto-invalidate; the contract is "cache on first call").
    with torch.no_grad():
        layernorm.weight.copy_(torch.randn(dim, device = "cuda"))
    assert _is_weightless(layernorm) is True  # cached
