#include <iostream>
#include <cuda_runtime.h>

__global__ void SiLU_kernel(float* x, float* y, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        y[idx] = x[idx] / (1 + exp(-x[idx]));
    }
}


int main() {
    int size = 16;
    float* x;
    float* y;
    float* d_x;
    float* d_y;
    
    x = (float*)malloc(size * sizeof(float));
    y = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        x[i] = i;
    }

    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_y, size * sizeof(float));

    cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

    SiLU_kernel<<<1, size>>>(d_x, d_y, size);
    
    cudaMemcpy(y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; i++) {
        std::cout << y[i] << " ";
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

}
