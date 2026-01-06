#include <iostream>
#include <cuda_runtime.h>

__global__ void vecAdd(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 256;
    const int size = N * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i + 2;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 1, 1);
    vecAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    printf("Result (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_c[i]);
    }
    printf("\n");
    
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
