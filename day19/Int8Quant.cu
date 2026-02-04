#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void quantize_kernel(const float* __restrict__ input, int8_t* __restrict__ output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        float scaled = val / scale;
        float rounded = roundf(scaled);
        float clamped = fminf(fmaxf(rounded, -128.0f), 127.0f);
        
        output[idx] = (int8_t)clamped;
    }
}

int main() {
    int n = 1 << 20;
    size_t size_bytes = n * sizeof(float);
    size_t output_size_bytes = n * sizeof(int8_t);

    float *h_input, *d_input;
    int8_t *h_output, *d_output;

    h_input = (float*)malloc(size_bytes);
    h_output = (int8_t*)malloc(output_size_bytes);

    float max_abs_val = 0.0f;
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 200.0f - 100.0f;
        max_abs_val = fmaxf(max_abs_val, fabsf(h_input[i]));
    }

    float scale = max_abs_val / 127.0f;
    printf("Computed scale: %f, Max Abs Val: %f\n", scale, max_abs_val);

    cudaMalloc(&d_input, size_bytes);
    cudaMalloc(&d_output, output_size_bytes);

    cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 256;
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    quantize_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, scale, n);


    cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        float val = h_input[i];
        float scaled = val / scale;
        float rounded = roundf(scaled);
        float clamped = fminf(fmaxf(rounded, -128.0f), 127.0f);
        int8_t expected = (int8_t)clamped;
        
        if (h_output[i] != expected) {
            printf("Mismatch at index %d: expected %d, got %d (input %f, scale %f)\n", 
                   i, expected, h_output[i], val, scale);
        }
    }
    printf("Verification PASSED!\n");

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}