#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Dropout Kernel
// Applies dropout mask and scaling to the input.
// output = input * mask * scale
// mask should contain 0.0 or 1.0.
// scale should be 1.0 / (1.0 - dropout_probability) if keeping magnitude same during training.

__global__ void dropoutKernel(const float* input, float* output, const float* mask, float scale, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

int main() {

    const int N = 1024 * 1024;
    const float dropout_prob = 0.5;
    const float scale = 1.0 / (1.0 - dropout_prob);

    std::cout << "Dropout Probability: " << dropout_prob << std::endl;
    std::cout << "Scale Factor: " << scale << std::endl;


    float *h_input, *h_mask, *h_output_gpu;
    h_input = (float*)malloc(N * sizeof(float));
    h_mask = (float*)malloc(N * sizeof(float));
    h_output_gpu = (float*)malloc(N * sizeof(float));


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_val(-10.0f, 10.0f);
    std::bernoulli_distribution dis_mask(1.0 - dropout_prob);

    for (int i = 0; i < N; ++i) {
        h_input[i] = dis_val(gen);
        h_mask[i] = dis_mask(gen) ? 1.0 : 0.0;
    }


    float *d_input, *d_output, *d_mask;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_mask, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));


    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, N * sizeof(float), cudaMemcpyHostToDevice);


    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    dropoutKernel<<<gridSize, blockSize>>>(d_input, d_output, d_mask, scale, N);
    

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    

    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);


    bool match = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_input[i] * h_mask[i] * scale;
        if (std::abs(h_output_gpu[i] - expected) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": GPU " << h_output_gpu[i] 
                      << ", CPU " << expected << std::endl;
            match = false;
            if (i > 10) break; // Don't print too many errors
        }
    }

    if (match) {
        std::cout << "Verification Success! Processed " << N << " elements." << std::endl;
    } else {
        std::cout << "Verification Failed!" << std::endl;
    }


    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

}
