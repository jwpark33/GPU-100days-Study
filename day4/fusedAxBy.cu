#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fusedAxBy(float *a, float *b, float *c, float alpha, float beta) {
    int idx = threadIdx.x;
    c[idx] = alpha * a[idx] + beta * b[idx];
}

int main() {
    int N = 16;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    float alpha = 2.0, beta = 3.0;
    
    // Allocate memory on host
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    // Allocate memory on device
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    fusedAxBy<<<1, N>>>(d_a, d_b, d_c, alpha, beta);
    
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf(" %f= %f * %f + %f * %f\n", c[i], a[i], alpha, b[i], beta);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(a);
    free(b);
    free(c);
}
