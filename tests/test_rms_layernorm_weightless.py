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
    reference and the weightless detection caches its result."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm, _is_weightless

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
    assert _is_weightless(layernorm) is True


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

    from unsloth.kernels.rms_layernorm import _is_weightless
    assert _is_weightless(layernorm) is False
    assert torch.amax((Y_test - Y_ref).abs()).item() <= 5e-2
    assert torch.amax((X_test.grad - X_ref.grad).abs()).item() <= 5e-2


def test_is_weightless_caches_result_and_invalidates_on_mutation():
    """The detection helper caches its result for repeat calls, but
    invalidates when the weight tensor is mutated in place (e.g. via
    `load_state_dict`'s `copy_`) or reassigned."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import _is_weightless

    dim = 256
    layernorm = LlamaRMSNorm((dim,), eps = 1e-5).to("cuda")
    with torch.no_grad():
        layernorm.weight.fill_(1.0)

    # First call populates the cache.
    assert getattr(layernorm, "_unsloth_weightless", None) is None
    assert _is_weightless(layernorm) is True
    cache_after_first = layernorm._unsloth_weightless
    assert cache_after_first is not None

    # Second call with unchanged weight should hit the cache (same key).
    assert _is_weightless(layernorm) is True
    assert layernorm._unsloth_weightless is cache_after_first  # same tuple

    # In-place mutation (matches what HF's load_state_dict does) must
    # invalidate the cache — otherwise we'd silently skip a real scale.
    with torch.no_grad():
        layernorm.weight.copy_(torch.randn(dim, device = "cuda") + 0.5)
    assert _is_weightless(layernorm) is False

    # Reassigning the parameter to a fresh all-ones tensor must also
    # re-trigger detection (data_ptr changes).
    layernorm.weight = torch.nn.Parameter(torch.ones(dim, device = "cuda"))
    assert _is_weightless(layernorm) is True


def test_load_state_dict_invalidates_weightless_cache():
    """End-to-end regression: a model first seen with ones weights must
    not produce stale (weightless) outputs after load_state_dict swaps in
    a checkpoint with non-identity RMSNorm weights."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    eps = 1e-5
    dim = 1024
    bsz, seqlen = 2, 128
    dtype = torch.float16

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    with torch.no_grad():
        layernorm.weight.fill_(1.0)

    torch.manual_seed(7)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")

    # Prime the cache via a first forward.
    _ = fast_rms_layernorm(layernorm, X.clone())

    # Swap in a non-identity checkpoint, mirroring load_state_dict.
    new_weight = torch.rand(dim, device = "cuda") + 0.5
    layernorm.load_state_dict({"weight": new_weight})

    # Now the kernel must apply the scale; output should match a weighted
    # reference, not the (stale) weightless one.
    Y_test = fast_rms_layernorm(layernorm, X.clone())
    Y_weighted = _ref_rmsnorm(X, eps, weight = new_weight)
    Y_unscaled = _ref_rmsnorm(X, eps, weight = None)

    assert torch.amax((Y_test - Y_weighted).abs()).item() <= 5e-2
    # Sanity: the new weight is meaningfully different from identity, so
    # the weighted and unscaled outputs should diverge by more than the
    # tolerance — otherwise this test wouldn't be catching anything.
    assert torch.amax((Y_weighted - Y_unscaled).abs()).item() > 1e-1
