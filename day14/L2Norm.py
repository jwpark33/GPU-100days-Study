import torch
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    block_sum = tl.sum(x_sq, axis=0)

    tl.atomic_add(output_ptr, block_sum)

def l2_norm(x):
    n_elements = x.numel()
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    l2_norm_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return torch.sqrt(output)

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 256 * 256
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    
    triton_result = l2_norm(x)
    torch_result = torch.norm(x)
    
    print(f"Triton L2 Norm: {triton_result.item()}")
    print(f"PyTorch L2 Norm: {torch_result.item()}")
    
    if torch.allclose(triton_result, torch_result, atol=1e-4):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
