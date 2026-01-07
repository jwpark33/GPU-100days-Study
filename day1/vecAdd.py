import torch
import triton
import triton.language as tl


@triton.jit
def vecAdd_kernel(
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
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def vecAdd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vecAdd_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return c


if __name__ == "__main__":
    N = 256
    a = torch.arange(N, dtype=torch.float32, device='cuda')
    b = torch.arange(N, dtype=torch.float32, device='cuda')

    c = vecAdd(a, b)
    
    print(f"\nOutput (c = a + b):")
    for i in range(N):
        print(f"c[{i}] = {c[i].item():.6f} (a[{i}] + b[{i}])")
