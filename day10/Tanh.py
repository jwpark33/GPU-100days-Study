import torch
import triton
import triton.language as tl

@triton.jit
def Tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)

def Tanh(x, y, N):
    n_elements = N
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    Tanh_kernel[grid](x, y, n_elements, BLOCK_SIZE)

if __name__ == "__main__":
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    Tanh(x, y, x.numel())
    
    torch_tanh = torch.tanh
    torch_y = torch_tanh(x)
    
    assert torch.allclose(y, torch_y), "Tanh kernel functionality check failed"

    print("Functionality check passed!!")
