import tabulate
import torch

import triton
import triton.language as tl

@triton.jit
def baseline_dropout_kernel(x_ptr, x_keep_ptr, y_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(y_ptr + offsets, output, mask=mask)

def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float):
    y = torch.empty_like(x)
    n_elements = y.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    baseline_dropout_kernel[grid](x, x_keep, y, n_elements, p, BLOCK_SIZE=1024)

    return y

if __name__ == "__main__":
    N = 16
    x = torch.randn(N, device='cuda')
    p = 0.5
    x_keep = (torch.rand(size=(N,), device='cuda') > p).to(torch.int32)

    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))