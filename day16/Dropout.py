import torch
import triton
import triton.language as tl

@triton.jit
def dropout_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    keep_mask = tl.load(mask_ptr + offsets, mask=mask)
    
    output = x * keep_mask * scale
    
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x: torch.Tensor, mask: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Apply dropout with a given mask.
    
    Args:
        x: Input tensor
        mask: Dropout mask (1.0 for keep, 0.0 for drop). Must be same shape as x.
        prob: Dropout probability (used for scaling)
    
    Returns:
        Output tensor with dropout applied.
    """
    n_elements = x.numel()
    output = torch.empty_like(x)
    scale = 1.0 / (1.0 - prob)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dropout_kernel[grid](
        x, output, mask, scale, n_elements,
        BLOCK_SIZE=1024
    )
    
    return output

if __name__ == "__main__":
    torch.manual_seed(0)        
    device = torch.device('cuda')
    
    size = 1024 * 1024
    prob = 0.5
    
    print(f"Testing Triton Dropout with N={size}, p={prob}")
    
    x = torch.randn(size, device=device, dtype=torch.float32)
    
    mask = torch.bernoulli(torch.full((size,), 1.0 - prob, device=device)).to(torch.float32)
    
    output = dropout(x, mask, prob)
    
    scale = 1.0 / (1.0 - prob)
    expected = x * mask * scale
    
    if torch.allclose(output, expected):
        print("Triton Dropout kernel Success!")
    else:
        print("Triton Dropout kernel Failed!")
        diff = (output - expected).abs()
        print(f"Max diff: {diff.max().item()}")
        print(f"Indices with diff: {torch.where(diff > 1e-5)[0][:10]}")
