import torch
import triton
import triton.language as tl

@triton.jit
def vector_mean_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Sum within the block
    block_sum = tl.sum(x, axis=0)

    # Each block adds its contribution to the mean: sum(block) / N
    # We cast n_elements to float32 to ensure floating point division
    mean_contribution = block_sum / n_elements.to(tl.float32)

    tl.atomic_add(output_ptr, mean_contribution)

def vector_mean(x: torch.Tensor):
    n_elements = x.numel()
    output = torch.zeros(1, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    vector_mean_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 1024 * 1024
    x = torch.rand(size, device='cuda', dtype=torch.float32)

    triton_mean = vector_mean(x)
    torch_mean = torch.mean(x)

    print(f"Triton Mean: {triton_mean.item()}")
    print(f"Torch Mean:  {torch_mean.item()}")

    if torch.allclose(triton_mean, torch_mean, atol=1e-5):
        print("Test PASSED")
    else:
        print("Test FAILED")
