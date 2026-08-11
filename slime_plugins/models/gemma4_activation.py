"""Gemma-4 activation kernels aligned with SGLang rollout."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gemma4_gelu_tanh_and_mul_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    half_width: tl.constexpr,
    output_numel,
    LARGE_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if LARGE_TENSOR:
        offsets = offsets.to(tl.int64)
    mask = offsets < output_numel
    row = offsets // half_width
    column = offsets - row * half_width
    gate_offsets = row * (2 * half_width) + column
    up_offsets = gate_offsets + half_width

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(input_ptr + gate_offsets, mask=mask).to(tl.float32)
    up = tl.load(input_ptr + up_offsets, mask=mask).to(tl.float32)

    alpha = 0.044715
    beta = 0.7978845608028654
    gate_squared = gate * gate
    tanh_value = libdevice.tanh(beta * (gate + alpha * gate * gate_squared))
    gelu = 0.5 * gate * (1.0 + tanh_value)
    gelu_grad = 0.5 * (1.0 + tanh_value) + 0.5 * gate * (1.0 - tanh_value * tanh_value) * beta * (
        1.0 + 3.0 * alpha * gate_squared
    )

    tl.store(grad_input_ptr + gate_offsets, grad_output * up * gelu_grad, mask=mask)
    tl.store(grad_input_ptr + up_offsets, grad_output * gelu, mask=mask)


class _Gemma4GeluTanhAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        from sgl_kernel import gelu_tanh_and_mul

        output = gelu_tanh_and_mul(inputs)
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (inputs,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(inputs)
        half_width = inputs.shape[-1] // 2
        output_numel = grad_output.numel()
        block_size = 256
        _gemma4_gelu_tanh_and_mul_backward_kernel[(triton.cdiv(output_numel, block_size),)](
            grad_output,
            inputs,
            grad_input,
            half_width,
            output_numel,
            LARGE_TENSOR=inputs.numel() > 2**31,
            BLOCK_SIZE=block_size,
        )
        return grad_input


class Gemma4GeluAndMul(nn.Module):
    """Apply Gemma-4 GeGLU with rollout-identical CUDA arithmetic."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if os.environ.get("SLIME_GEMMA4_BATCH_INVARIANT") == "1" and inputs.is_cuda:
            return _Gemma4GeluTanhAndMul.apply(inputs)

        gate, up = torch.chunk(inputs, 2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Return no checkpoint shards because this activation is stateless."""
        return {}
