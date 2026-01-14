import torch
import triton
import triton.language as tl
import math

@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def GeLU_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = 0.5 * x * (1.0 + _tanh(tl.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    tl.store(y_ptr + offsets, y, mask = mask)

def GeLU(x, y, N):
    n_elements = N
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    GeLU_kernel[grid](x, y, n_elements, BLOCK_SIZE)
    return y

if __name__ == "__main__":
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    GeLU(x, y, x.numel())
    print(y)
