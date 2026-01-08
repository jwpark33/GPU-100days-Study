#include <stdio.h>
#include <cuda_runtime.h>

__global__ void eleSub(float *a, float *b, float *c) {
    int idx = threadIdx.x;
    c[idx] = a[idx] - b[idx];
}

int main() {
    const int N = 32;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on host
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));
    
    // Allocate memory on device
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));
    
    // Initialize data on host
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // Copy data from host to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    eleSub<<<1, N>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %f (a[%d] - b[%d])\n", i, c[i], i, i);
    }
    
    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}