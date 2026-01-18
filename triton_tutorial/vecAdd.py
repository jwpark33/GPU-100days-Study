import torch
import triton
import triton.language as tl

@triton.jit
def vecAdd_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)
    
def vecAdd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n_elements = a.numel()

    BLOCK_SIZE = 2048

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    vecAdd_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return c

@triton.testing.perf_report(
triton.testing.Benchmark(
    x_names=['size'],  # Argument names to use as an x-axis for the plot.
    x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
    x_log=True,  # x axis is logarithmic.
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
    line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
    line_names=['Triton', 'Torch'],  # Label name for the lines.
    styles=[('blue', '-'), ('green', '-')],  # Line styles.
    ylabel='GB/s',  # Label name for the y-axis.
    plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
    args={},  # Values for function arguments not in `x_names` and `y_name`.
))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vecAdd(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    N = 1024
    a = torch.arange(N, dtype=torch.float32, device='cuda')
    b = torch.arange(N, dtype=torch.float32, device='cuda')
    c = vecAdd(a, b)
    
    assert torch.allclose(a + b, c), "VecAdd functionality check failed"

    print("Functionality check passed!!")
    
    benchmark.run(print_data=True, show_plots=True)

