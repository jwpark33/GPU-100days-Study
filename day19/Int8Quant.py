import torch
import triton
import triton.language as tl

@triton.jit
def quantize_kernel(
    x_ptr,
    y_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    scaled = x / scale
    # rounded = tl.math.round(scaled)
    rounded = tl.extra.cuda.libdevice.round(scaled) # Round Half Away From Zero (-120.5 -> -121)
    clamped = tl.clamp(rounded, -128.0, 127.0)
    
    y = clamped.to(tl.int8)

    tl.store(y_ptr + offsets, y, mask=mask)

def quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    n_elements = x.numel()
    y = torch.empty(n_elements, device=x.device, dtype=torch.int8)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    quantize_kernel[grid](x, y, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return y

def main():
    torch.manual_seed(0)
    n = 1024 * 1024
    x = torch.rand(n, device='cuda', dtype=torch.float32) * 200.0 - 100.0
    
    max_abs_val = torch.max(torch.abs(x)).item()
    scale = max_abs_val / 127.0
    print(f"Scale: {scale}, Max Abs Val: {max_abs_val}")

    y_triton = quantize(x, scale)
    
    # PyTorch reference implementation
    # Note: torch.round rounds to nearest even (banker's rounding). # Banker's Rounding (Round to Nearest Even) (-120.5 -> -120)
    # CUDA roundf (and thus Triton's libdevice.round) rounds halfway cases away from zero.
    # We need to match CUDA behavior for verification.
    def round_half_away_from_zero(x):
        return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5)) 
        
    y_torch_float = torch.clamp(round_half_away_from_zero(x / scale), -128, 127)
    y_torch = y_torch_float.to(torch.int8)
    
    if torch.allclose(y_triton.float(), y_torch.float()):
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")
        diff = (y_triton != y_torch)
        print(f"Number of mismatches: {diff.sum().item()}")
        idxs = torch.where(diff)[0]
        if len(idxs) > 0:
            first_idx = idxs[0].item()
            print(f"Mismatch at index {first_idx}:")
            print(f"Input: {x[first_idx]}")
            print(f"Triton: {y_triton[first_idx]}")
            print(f"PyTorch: {y_torch[first_idx]}")

if __name__ == "__main__":
    main()