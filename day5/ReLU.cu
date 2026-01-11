#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void ReLU(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = max(0.0f, x[idx]);
    }
}

int main() {
    int N = 16;
    float *x, *y;
    float *d_x, *d_y;

    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    
    cudaMalloc(&d_x, N * sizeof(float));    
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // -10.0 ~ 10.0 random
    }
    
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    ReLU<<<1, N>>>(d_x, d_y, N);
    
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool allNonNegative = true;
    for (int i = 0; i < N; i++) {
        std::cout << x[i] << "\t\t" << y[i] << std::endl;
        if (y[i] < 0) {
            allNonNegative = false;
        }
    }
    
    if (allNonNegative) {
        std::cout << "PASS!!!" << std::endl;
    } else {
        std::cout << "FAIL!!!!!" << std::endl;
    }
    
    cudaFree(d_x);
    cudaFree(d_y);

    free(x);
    free(y);

}
