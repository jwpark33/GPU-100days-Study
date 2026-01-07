import torch
import triton
import triton.language as tl


@triton.jit
def vecScale_kernel(
    a_ptr,
    b_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = a * scale
    tl.store(b_ptr + offsets, b, mask=mask)

def vecScale(a: torch.Tensor, scale: float) -> torch.Tensor:
    b = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vecScale_kernel[grid](
        a, b, scale, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return b


if __name__ == "__main__":
    N = 16
    a = torch.arange(N, dtype=torch.float32, device='cuda')

    scale = 2.0

    b = vecScale(a, scale)
    
    print(f"\nOutput (b = a * {scale}):")
    for i in range(N):
        print(f"b[{i}] = {b[i].item():.6f} (a[{i}] * {scale:.1f})")