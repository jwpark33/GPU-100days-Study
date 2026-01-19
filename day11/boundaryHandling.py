import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

def boundary_handling(size):
    output = torch.empty(size, device='cuda')
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )

    BLOCK_SIZE = 256
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=BLOCK_SIZE)
    
    return output, x + y

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 100000
    print(f"Vector size: {size}")

    triton_output, torch_output = boundary_handling(size)
    
    if torch.allclose(triton_output, torch_output):
        print("Success! Triton kernel output matches Torch output.")
    else:
        print("Failure! Output mismatch.")
        print(f"Max diff: {torch.max(torch.abs(triton_output - torch_output))}")
