import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(x_ptr, scale_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets, mask=mask)
    output = x.to(tl.float32) * scale

    tl.store(y_ptr + offsets, output, mask=mask)

def dequantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and scale.is_cuda
    n_elements = x.numel()
    y = torch.empty(n_elements, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE=256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dequantize_kernel[grid](
        x, scale, y, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

def main():
    torch.manual_seed(0)
    n = 1024 * 1024

    print(f"Vector size: {n}")
    
    x = torch.randint(-128, 128, (n,), device='cuda', dtype=torch.int8)
    scale = torch.full((n,), 0.1, device='cuda', dtype=torch.float32)

    y = dequantize(x, scale)

    y_ref = x.to(torch.float32) * scale
    
    if torch.allclose(y, y_ref, atol=1e-4):
        print("All results are correct!")
    else:
        print("Mismatch found")
        max_diff = torch.max(torch.abs(y - y_ref))
        print(f"Max difference: {max_diff}")
        print(f"First 10 Triton: {y[:10]}")
        print(f"First 10 Reference: {y_ref[:10]}")

if __name__ == "__main__":
    main()