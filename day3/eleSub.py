import torch
import triton
import triton.language as tl


@triton.jit
def eleSub_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a - b
    tl.store(c_ptr + offsets, c, mask=mask)

def eleSub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 32

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    eleSub_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return c


if __name__ == "__main__":
    N = 32
    a = torch.full((N,), 0.0, dtype=torch.float32, device='cuda')
    b = torch.arange(N, dtype=torch.float32, device='cuda')

    c = eleSub(a, b)
    
    print(f"\nOutput (c = a - b):")
    for i in range(N):
        print(f"c[{i}] = {c[i].item():.6f} (a[{i}] - b[{i}])")
