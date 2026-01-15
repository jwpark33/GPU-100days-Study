import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = x * tl.sigmoid(x)
    tl.store(y_ptr + offsets, y, mask = mask)

def SiLU(x, y, N):
    n_elements = N
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](x, y, n_elements, BLOCK_SIZE)

if __name__ == "__main__":
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    SiLU(x, y, x.numel())
    
    torch_siLU = torch.nn.SiLU()
    torch_y = torch_siLU(x)
    
    assert torch.allclose(y, torch_y)
   
    print(y)
