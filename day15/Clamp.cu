#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

__global__ void clampKernel(float* input, float* output, float min, float max, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float value = input[idx];
        output[idx] = value < min ? min : (value > max ? max : value);
    }
}



int main() {
    const int N = 256 * 256;
    const float min_val = 2.0;
    const float max_val = 8.0;

    std::vector<float> h_input(N);
    std::vector<float> h_output_gpu(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 10.0);

    for (int i = 0; i < N; ++i) {
        h_input[i] = dis(gen);
    }

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    clampKernel<<<gridSize, blockSize>>>(d_input, d_output, min_val, max_val, N);
    
    cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool match = true;
    for (int i = 0; i < N; i++) {
        float cpu_val = std::clamp(h_input[i], min_val, max_val);
        if (std::abs(h_output_gpu[i] - cpu_val) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": GPU " << h_output_gpu[i] 
                      << ", CPU " << cpu_val << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Verification Success! Processed " << N << " elements." << std::endl;
    } else {
        std::cout << "Verification Failed!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

}