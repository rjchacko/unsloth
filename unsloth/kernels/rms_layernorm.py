# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

import triton
import triton.language as tl
import torch
from typing import Optional
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    if HAS_WEIGHT:
        W_row = tl.load(W + col_offsets, mask = mask, other = 0)  # .to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    if HAS_WEIGHT:
        normed = normed.to(W_row.dtype)  # Exact copy from HF
        output = normed * W_row
    else:
        output = normed.to(Y.dtype.element_ty)
    tl.store(Y + col_offsets, output, mask = mask)


def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    # dW, dW_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel for the backward pass
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if GEMMA:
        dX += row_idx * dY_row_stride
    else:
        dX = dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    if HAS_WEIGHT:
        W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if HAS_WEIGHT:
        if GEMMA:
            dY_W = dY_row * (W_row + 1.0)
        else:
            dY_W = dY_row * W_row
    else:
        dY_W = dY_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
        "HAS_WEIGHT": lambda args: bool(args["HAS_WEIGHT"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
    # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
    # exactly. Essentially all in float32!
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: Optional[torch.Tensor], eps: float, gemma: bool = False):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim)
        n_rows: int
        n_cols: int
        n_rows, n_cols = X.shape
        BLOCK_SIZE: int
        num_warps: int
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device)
        r = torch.empty(n_rows, dtype = torch.float32, device = device)

        has_weight = W is not None
        # Triton needs a real tensor argument; pass X as a dummy when weightless.
        # The guarded tl.load never executes, so the pointer value is unused.
        W_arg = W if has_weight else X
        W_stride = W.stride(0) if has_weight else 0

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        with torch_gpu_device(device):
            if gemma:
                fx[(n_rows,)](
                    Y,
                    Y.stride(0),
                    X,
                    X.stride(0),
                    W_arg,
                    W_stride,
                    r,
                    r.stride(0),
                    n_cols,
                    eps,
                    BLOCK_SIZE = BLOCK_SIZE,
                    num_warps = num_warps,
                )
            else:
                fx[(n_rows,)](
                    Y,
                    Y.stride(0),
                    X,
                    X.stride(0),
                    W_arg,
                    W_stride,
                    r,
                    r.stride(0),
                    n_cols,
                    eps,
                    BLOCK_SIZE = BLOCK_SIZE,
                    num_warps = num_warps,
                    HAS_WEIGHT = has_weight,
                )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.has_weight = has_weight
        if has_weight:
            ctx.save_for_backward(X, W, r)
        else:
            ctx.save_for_backward(X, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim: int = shape[-1]
        dY = dY.reshape(-1, dim)
        if ctx.has_weight:
            X, W, r = ctx.saved_tensors
            W_stride = W.stride(0)
        else:
            X, r = ctx.saved_tensors
            W = X  # dummy pointer; the guarded tl.load never runs.
            W_stride = 0
        n_rows: int
        n_cols: int
        n_rows, n_cols = dY.shape
        # dW = X
        dX = torch.empty_like(dY) if ctx.GEMMA else dY

        with torch_gpu_device(dY.device):
            _rms_layernorm_backward[(n_rows,)](
                dY,
                dY.stride(0),
                dX,
                dX.stride(0),
                X,
                X.stride(0),
                W,
                W_stride,
                r,
                r.stride(0),
                # dW, dW.stride(0),
                n_cols,
                ctx.eps,
                GEMMA = ctx.GEMMA,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps = ctx.num_warps,
                HAS_WEIGHT = ctx.has_weight,
            )
        dX = dX.view(*shape)
        return dX, None, None, None


def _is_weightless(layernorm) -> bool:
    """Detect whether an RMSNorm module is FlashNorm-folded (no useful scale).

    Returns True when the layernorm has no weight, or when its weight is
    exactly all-ones (the convention HF uses to keep param shape on
    FlashNorm-folded checkpoints). Caches the per-module result on first
    call to avoid rescanning the weight every forward.
    """
    W = getattr(layernorm, "weight", None)
    if W is None:
        return True
    cached = getattr(layernorm, "_unsloth_weightless", None)
    if cached is not None:
        return cached
    cached = bool(torch.equal(W, torch.ones_like(W)))
    layernorm._unsloth_weightless = cached
    return cached


# [TODO] Unsure why RMS Layernorm is not torch.compiling properly
@torch.compiler.disable
def fast_rms_layernorm(layernorm, X: torch.Tensor, gemma: bool = False):
    eps: float = (
        layernorm.variance_epsilon
        if hasattr(layernorm, "variance_epsilon")
        else layernorm.eps
    )
    if gemma:
        # Gemma's RMSNorm uses (w + 1) semantics; weightless detection differs
        # (identity = w == 0). Defer Gemma weightless support to a follow-up.
        W = layernorm.weight
    else:
        W = None if _is_weightless(layernorm) else layernorm.weight
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out


from transformers.models.llama.modeling_llama import LlamaRMSNorm


class Unsloth_LlamaRMSNorm(LlamaRMSNorm):
    def forward(self, X):
        return fast_rms_layernorm(self, X, gemma = False)


try:
    from transformers.models.mllama.modeling_mllama import MllamaTextRMSNorm

    class Unsloth_MllamaTextRMSNorm(MllamaTextRMSNorm):
        def forward(self, X):
            return fast_rms_layernorm(self, X, gemma = False)


except:
    pass


def patch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = Unsloth_LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama

        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = (
            Unsloth_MllamaTextRMSNorm
        )
    except:
        pass
    return


def unpatch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama

        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = MllamaTextRMSNorm
    except:
        pass
    return


def test_rms_layernorm(
    dim = 1024,
    eps = 1e-5,
    dtype = torch.float16,
    bsz = 21,
    random_state = 3407,
    seqlen = 3341,
):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X.requires_grad_(True)
    XX.requires_grad_(True)
    Y = layernorm(X)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y.backward(YY)
    correct_grad = X.grad.clone()
    # from unsloth.kernels import fast_rms_layernorm
    Y = fast_rms_layernorm(layernorm, XX)
    Y.backward(YY)
    assert torch.amax(correct_grad - XX.grad).item() <= 0.05


def test_rms_layernorm_weightless(
    dim = 1024,
    eps = 1e-5,
    dtype = torch.float16,
    bsz = 21,
    random_state = 3407,
    seqlen = 3341,
    mode = "ones",  # "ones" → weight=torch.ones, "none" → weight=None
):
    """Check the weightless path: with weight=None or weight=all-ones, the
    forward output and dX gradient must match an unweighted torch reference
    `x / sqrt(mean(x²) + eps)`."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    if mode == "none":
        # True weightless: drop the parameter entirely.
        layernorm.weight = None
    else:
        # Identity weight (HF convention for FlashNorm-folded checkpoints).
        with torch.no_grad():
            layernorm.weight.fill_(1.0)

    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X.requires_grad_(True)
    XX.requires_grad_(True)

    # Torch reference: identity-scale RMSNorm = x / sqrt(mean(x²) + eps)
    Xf = X.to(torch.float32)
    inv_rms = (Xf.square().mean(-1, keepdim = True) + eps).rsqrt()
    Y_ref = (Xf * inv_rms).to(dtype)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y_ref.backward(YY)
    correct_grad = X.grad.clone()

    Y = fast_rms_layernorm(layernorm, XX)
    Y.backward(YY)
    assert torch.amax((Y - Y_ref).abs()).item() <= 0.05, (
        f"weightless forward mismatch in mode={mode}, dim={dim}, dtype={dtype}"
    )
    assert torch.amax((correct_grad - XX.grad).abs()).item() <= 0.05, (
        f"weightless backward mismatch in mode={mode}, dim={dim}, dtype={dtype}"
    )

    # When mode='ones', the cache flag should be set after the first call.
    if mode == "ones":
        assert getattr(layernorm, "_unsloth_weightless", None) is True


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )
                        for mode in ("ones", "none"):
                            test_rms_layernorm_weightless(
                                dim = dim,
                                eps = 1e-5,
                                dtype = dtype,
                                bsz = 21,
                                random_state = random_state,
                                seqlen = seqlen,
                                mode = mode,
                            )
