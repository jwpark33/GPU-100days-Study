import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr,
    mask_ptr,
    p,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    rand_val = tl.rand(seed, offsets)

    keep_mask = tl.where(rand_val < p, 0.0, 1.0)

    # Apply mask
    output = x * keep_mask

    # Store results
    tl.store(x_ptr + offsets, output, mask=mask)
    tl.store(mask_ptr + offsets, keep_mask, mask=mask)


def dropout(x, p=0.5, seed=1234):
    n_elements = x.numel()
    mask = torch.empty_like(x)
    
    # Grid calculation
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    dropout_kernel[grid](
        x,
        mask,
        p,
        n_elements,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x, mask


def main():
    torch.manual_seed(0)
    size = 1024
    p = 0.5
    seed = 1234
    
    # Input data
    x = torch.ones(size, device='cuda', dtype=torch.float32)
    
    print(f"Input size: {size}")
    print(f"Dropout p: {p}")
    print(f"Seed: {seed}")
    
    # Run kernel
    # Note: x is modified in-place in our kernel logic matching the CUDA one
    x_out, mask_out = dropout(x, p, seed)
    
    # Verification
    # Count zeros
    dropped_count = (mask_out == 0).sum().item()
    total_count = size
    rate = dropped_count / total_count
    
    print(f"Total elements: {total_count}")
    print(f"Dropped elements: {dropped_count}")
    print(f"Dropout rate: {rate:.4f} (Target: {p})")
    
    # Verify values
    if torch.all((x_out == 0) | (x_out == 1)):
        print("Values check passed (either 0 or 1)")
    else:
        print("Values check FAILED")
        
    # Consistency check
    if torch.all((mask_out == 0) == (x_out == 0)):
        print("Mask consistency passed")
    else:
        print("Mask consistency FAILED")


if __name__ == "__main__":
    main()