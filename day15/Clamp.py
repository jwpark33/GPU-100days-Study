import torch
import triton
import triton.language as tl

@triton.jit
def clamp_kernel(
    input_ptr,
    output_ptr,
    min_val,
    max_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Clamp logic: max(min(x, max_val), min_val)
    # Using element-wise min/max from triton.language
    output = tl.minimum(tl.maximum(x, min_val), max_val)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def clamp(x: torch.Tensor, min_val: float, max_val: float):
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    clamp_kernel[grid](
        x,
        output,
        min_val,
        max_val,
        n_elements,
        BLOCK_SIZE=256,
    )
    return output

def main():
    torch.manual_seed(0)
    N = 256 * 256
    min_val = 2.0
    max_val = 8.0
    
    # Initialize input tensor
    x = torch.rand(N, device='cuda', dtype=torch.float32) * 10.0
    
    # Run Triton kernel
    output_triton = clamp(x, min_val, max_val)
    
    # Run PyTorch reference
    output_torch = torch.clamp(x, min_val, max_val)
    
    # Verify
    if torch.allclose(output_triton, output_torch):
        print(f"Verification Success! Processed {N} elements.")
    else:
        print("Verification Failed!")
        print(f"Max difference: {torch.max(torch.abs(output_triton - output_torch))}")

if __name__ == "__main__":
    main()
