import torch
import triton
import triton.language as tl

@triton.jit
def quantization_kernel(
    input_ptr,
    quantized_ptr,
    reconstructed_ptr,
    shift_bits,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(input_ptr + offsets, mask=mask)
    val_int = val.to(tl.int32)
    q = val_int >> shift_bits
    recon_int = q << shift_bits
    recon = recon_int.to(tl.float32)

    tl.store(quantized_ptr + offsets, q, mask=mask)
    tl.store(reconstructed_ptr + offsets, recon, mask=mask)

def quantization(x, shift_bits):
    n_elements = x.numel()
    quantized = torch.empty(n_elements, dtype=torch.int32, device=x.device)
    reconstructed = torch.empty(n_elements, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    quantization_kernel[grid](
        x,
        quantized,
        reconstructed,
        shift_bits,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return quantized, reconstructed

def main():
    N = 16
    SHIFT_BITS = 2
    
    h_input = [i * 3.123 for i in range(N)]
    x = torch.tensor(h_input, device='cuda', dtype=torch.float32)
    
    quantized, reconstructed = quantization(x, SHIFT_BITS)
    
    h_x = x.cpu().numpy()
    h_q = quantized.cpu().numpy()
    h_r = reconstructed.cpu().numpy()
    
    print(f"Shift Bits: {SHIFT_BITS} (Quantization Step: {1 << SHIFT_BITS})")
    print("Idx\tInput\tInt(Cast)\tQuant(>>)\tRecon(<<)\tLoss")
    print("-" * 72)
    
    for i in range(N):
        diff = h_x[i] - h_r[i]
        cast_int = int(h_x[i])
        
        print(f"{i}\t{h_x[i]:.3f}\t{cast_int}\t\t{h_q[i]}\t\t{h_r[i]:.3f}\t\t{diff:.3f}")

if __name__ == "__main__":
    main()