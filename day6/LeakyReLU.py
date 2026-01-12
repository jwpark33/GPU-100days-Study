import torch
import triton
import triton.language as tl


@triton.jit
def LeakyReLU_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0) + tl.minimum(x, 0.0) * alpha
    tl.store(y_ptr + offsets, y, mask=mask)

def LeakyReLU(x,y,N,alpha):
    n_elements = N
    BLOCK_SIZE = 32

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    LeakyReLU_kernel[grid](
        x, y, n_elements, alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

if __name__ == "__main__":
    N = 16
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    alpha = 0.1
    LeakyReLU(x, y, N, alpha)
    
    for i in range(N):
        if x[i] >= 0:
            assert y[i] == x[i]
        else:
            assert y[i] == alpha * x[i]
        print(f'Before LeakyReLU: {x[i]} -> After LeakyReLU: {y[i]}')
