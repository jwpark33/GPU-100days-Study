import torch
import triton
import triton.language as tl


@triton.jit
def ReLU_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)

def ReLU(x,y,N):
    n_elements = N
    BLOCK_SIZE = 32

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    ReLU_kernel[grid](
        x, y, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

if __name__ == "__main__":
    N = 16
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    ReLU(x, y, N)
    
    for i in range(N):
        assert y[i] >= 0
        print(f'Before ReLU: {x[i]} -> After ReLU: {y[i]}')
