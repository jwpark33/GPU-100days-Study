#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecScale(float *a, float *b, float scale) {
    int i = threadIdx.x;
    b[i] = a[i] * scale;
}

int main() {
    const int N = 16;
    const int size = N * sizeof(float);
    float *h_a, *h_b;
    float *d_a, *d_b;
    
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    
    // Initialize a with values
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }
    
    float scale = 2.0;
    
    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(N, 1, 1);
    dim3 gridDim(1, 1, 1);
    vecScale<<<gridDim, blockDim>>>(d_a, d_b, scale);
    
    // Copy result back to host
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < N; i++) {
        printf("a[%d] = %f\n", i, h_a[i]);
    }
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    
}