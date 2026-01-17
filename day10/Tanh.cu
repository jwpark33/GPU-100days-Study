#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

__global__ void tanh_kernel(float *x, float *y, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = tanh(x[i]);
    }
}

int main() {
    const int n = 1024;
    float *h_x, *h_y;
    float *d_x, *d_y;
    
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));
    
    srand(42);  // seed for reproducibility
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)rand() / RAND_MAX;  // random value between 0 and 1
    }
    
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    tanh_kernel<<<gridSize, blockSize>>>(d_x, d_y, n);

    
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    int errorCount = 0;
    for (int i = 0; i < n; i++) {
        float expected = tanh(h_x[i]);
        float diff = std::abs(h_y[i] - expected);
        if (diff > 1e-6) {
            errorCount++;
        }               
    }
    assert(errorCount == 0 && "Error count is not zero");

    std::cout << "All " << n << " tests passed!" << std::endl;   

    cudaFree(d_x);
    cudaFree(d_y);

    free(h_x);
    free(h_y);
}
