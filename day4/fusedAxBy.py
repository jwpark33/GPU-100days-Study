import torch
import triton
import triton.language as tl


@triton.jit
def fusedAxBy_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    alpha,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = alpha * a + beta * b
    tl.store(c_ptr + offsets, c, mask=mask)

def fusedAxBy(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    c = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 32

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fusedAxBy_kernel[grid](
        a, b, c, n_elements,
        alpha, beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return c


if __name__ == "__main__":
    N = 16
    a = torch.arange(N, dtype=torch.float32, device='cuda')
    b = torch.arange(N, dtype=torch.float32, device='cuda') * 2
    alpha = 2.0
    beta = 3.0

    c = fusedAxBy(a, b, alpha, beta)
    
    for i in range(N):
        print(f"c[{i}] = {c[i]} = ({a[i]} * {alpha} + {b[i]} * {beta})")
