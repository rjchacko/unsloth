# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-runnable tests for the FlashNorm-folded RMSNorm support paths.

These cover the pure-Python pieces — the `_is_weightless` detection
helper, its cache invalidation, and the dispatch routing in
`fast_rms_layernorm` — so they exercise the same logic codex flagged
without needing a GPU. The actual Triton kernels still require CUDA
and are covered by tests/test_rms_layernorm.py.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level = True)


def _make_layernorm(dim: int = 256):
    """Build a fresh HF LlamaRMSNorm on CPU. Imported here (not at module
    scope) so collection doesn't fail if transformers isn't installed in
    some minimal test environment."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    return LlamaRMSNorm((dim,), eps = 1e-5)


# ---- _is_weightless detection ----------------------------------------------


def test_is_weightless_returns_true_when_weight_is_none():
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    ln.weight = None
    assert _is_weightless(ln) is True


def test_is_weightless_returns_true_when_weight_is_ones():
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)
    assert _is_weightless(ln) is True


def test_is_weightless_returns_false_for_random_weight():
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.copy_(torch.rand(256) + 0.5)
    assert _is_weightless(ln) is False


def test_is_weightless_returns_false_for_zeros_weight():
    """Zeros are not Llama-style identity — only all-ones is. Gemma uses
    zeros for identity via `(w + 1)`, but Gemma has its own kernel and we
    don't currently auto-fold it."""
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.zero_()
    assert _is_weightless(ln) is False


# ---- cache behavior --------------------------------------------------------


def test_cache_hit_reuses_result_without_rescanning():
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)

    assert _is_weightless(ln) is True
    first_cache = ln._unsloth_weightless
    assert _is_weightless(ln) is True
    # Same tuple object on consecutive cache hits.
    assert ln._unsloth_weightless is first_cache


def test_cache_invalidates_on_in_place_copy():
    """The codex bug: in-place mutation must invalidate the cache.

    HF's load_state_dict uses `param.copy_(...)` under the hood, which
    bumps `tensor._version` without changing `data_ptr()`."""
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)
    assert _is_weightless(ln) is True

    with torch.no_grad():
        ln.weight.copy_(torch.rand(256) + 0.5)
    assert _is_weightless(ln) is False


def test_cache_invalidates_on_parameter_reassignment():
    """Direct `.weight = nn.Parameter(...)` swap must invalidate the cache
    (the new parameter has a different storage pointer)."""
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.copy_(torch.rand(256) + 0.5)
    assert _is_weightless(ln) is False

    ln.weight = torch.nn.Parameter(torch.ones(256))
    assert _is_weightless(ln) is True


def test_cache_invalidates_on_load_state_dict():
    """End-to-end regression for the codex scenario, without GPU."""
    from unsloth.kernels.rms_layernorm import _is_weightless

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)
    assert _is_weightless(ln) is True

    ln.load_state_dict({"weight": torch.rand(256) + 0.5})
    assert _is_weightless(ln) is False


# ---- dispatch routing in fast_rms_layernorm --------------------------------


def test_fast_rms_layernorm_passes_none_when_weightless(monkeypatch):
    """The wrapper must pass W=None into Fast_RMS_Layernorm.apply when
    the layer is FlashNorm-folded, so the Triton kernel takes the
    weightless code path."""
    from unsloth.kernels import rms_layernorm as mod

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)

    captured = {}

    class FakeFastRMSLayernorm:
        @staticmethod
        def apply(X, W, eps, gemma):
            captured["W"] = W
            captured["eps"] = eps
            captured["gemma"] = gemma
            return X

    monkeypatch.setattr(mod, "Fast_RMS_Layernorm", FakeFastRMSLayernorm)

    X = torch.randn(2, 4, 256)
    _ = mod.fast_rms_layernorm(ln, X)

    assert captured["W"] is None
    assert captured["gemma"] is False
    assert captured["eps"] == pytest.approx(1e-5)


def test_fast_rms_layernorm_passes_real_weight_when_not_weightless(monkeypatch):
    """Negative: when the weight is real, it must be forwarded to apply()."""
    from unsloth.kernels import rms_layernorm as mod

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.copy_(torch.rand(256) + 0.5)

    captured = {}

    class FakeFastRMSLayernorm:
        @staticmethod
        def apply(X, W, eps, gemma):
            captured["W"] = W
            return X

    monkeypatch.setattr(mod, "Fast_RMS_Layernorm", FakeFastRMSLayernorm)

    X = torch.randn(2, 4, 256)
    _ = mod.fast_rms_layernorm(ln, X)

    assert captured["W"] is ln.weight


def test_fast_rms_layernorm_always_passes_weight_when_gemma(monkeypatch):
    """For the Gemma `(w+1)` path, weightless detection is intentionally
    skipped — we always pass the real weight, even if it's all-ones, so
    the kernel computes `normed * (W + 1.0)` correctly."""
    from unsloth.kernels import rms_layernorm as mod

    ln = _make_layernorm()
    with torch.no_grad():
        ln.weight.fill_(1.0)  # would trigger weightless in non-gemma path

    captured = {}

    class FakeFastRMSLayernorm:
        @staticmethod
        def apply(X, W, eps, gemma):
            captured["W"] = W
            captured["gemma"] = gemma
            return X

    monkeypatch.setattr(mod, "Fast_RMS_Layernorm", FakeFastRMSLayernorm)

    X = torch.randn(2, 4, 256)
    _ = mod.fast_rms_layernorm(ln, X, gemma = True)

    assert captured["W"] is ln.weight
    assert captured["gemma"] is True
