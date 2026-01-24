import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    A_ptr, B_ptr,
    stride_a_h, stride_a_w,
    stride_b_h, stride_b_w,
    H, W,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    mask_h = offs_h[:, None] < H
    mask_w = offs_w[None, :] < W
    mask_a = mask_h & mask_w

    a_ptrs = A_ptr + (offs_h[:, None] * stride_a_h + offs_w[None, :] * stride_a_w)
    
    a_val = tl.load(a_ptrs, mask=mask_a, other=0.0)

    b_val = tl.trans(a_val) 
    
    offs_b_row = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W) # Range of size BW
    offs_b_col = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H) # Range of size BH
    
    mask_b_row = offs_b_row[:, None] < W
    mask_b_col = offs_b_col[None, :] < H
    mask_b = mask_b_row & mask_b_col
    
    b_ptrs = B_ptr + (offs_b_row[:, None] * stride_b_h + offs_b_col[None, :] * stride_b_w)
    
    tl.store(b_ptrs, b_val, mask=mask_b)

def transpose(A):
    H, W = A.shape
    B = torch.empty((W, H), device=A.device, dtype=A.dtype)
    
    stride_a_h, stride_a_w = A.stride()
    stride_b_h, stride_b_w = B.stride()
    
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    grid = (triton.cdiv(H, BLOCK_SIZE_H), triton.cdiv(W, BLOCK_SIZE_W))
    
    transpose_kernel[grid](
        A, B,
        stride_a_h, stride_a_w,
        stride_b_h, stride_b_w,
        H, W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return B

def main():
    torch.manual_seed(0)
    H, W = 1024, 1024
    A = torch.randn(H, W, device='cuda', dtype=torch.float32)
    
    B_triton = transpose(A)
    
    B_ref = A.T.contiguous()
    
    if torch.allclose(B_triton, B_ref):
        print("Success! Triton transpose matches PyTorch.")
    else:
        print("Mismatch found!")
        print("Max diff:", torch.max(torch.abs(B_triton - B_ref)))

if __name__ == "__main__":
    main()