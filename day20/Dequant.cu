#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

// Int8 -> FP32 Dequantization Kernel
__global__ void dequantize(const int8_t* x, const float* scale, float* y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = static_cast<float>(x[idx]) * scale[idx];
    }
}

int main() {
    const int n = 1024 * 1024;
    
    int8_t* h_x = new int8_t[n];
    float* h_scale = new float[n];
    float* h_y = new float[n];

    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<int8_t>(i % 256 - 128);
        h_scale[i] = 0.1f;
    }

    int8_t* d_x;
    float* d_scale;
    float* d_y;

    cudaMalloc(&d_x, n * sizeof(int8_t));
    cudaMalloc(&d_scale, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    dequantize<<<numBlocks, blockSize>>>(d_x, d_scale, d_y, n);

    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = static_cast<float>(h_x[i]) * h_scale[i];
        float diff = h_y[i] - expected;
        if (diff < 0) diff = -diff;
        if (diff > 1e-3) {
            std::cout << "Mismatch at index " << i << ": Expected " << expected << ", got " << h_y[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) std::cout << "All results are correct!" << std::endl;

    delete[] h_x;
    delete[] h_scale;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_scale);
    cudaFree(d_y);

}
