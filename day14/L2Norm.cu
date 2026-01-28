#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void l2NormKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float localSum = 0.0f;
    if (idx < n) {
        float val = input[idx];
        localSum = val * val;
    }
    sdata[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    int n = 1 << 20; // 1M elements
    std::vector<float> h_input(n);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    float *d_input, *d_output;
    size_t bytes = n * sizeof(float);
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    l2NormKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);
    
    float sumOfSquares;
    cudaMemcpy(&sumOfSquares, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpu_result = sqrt(sumOfSquares);

    cudaFree(d_input);
    cudaFree(d_output);

    double cpu_sum_sq = 0.0;
    for (float val : h_input) {
        cpu_sum_sq += val * val;
    }
    float cpu_result = sqrt(cpu_sum_sq);
    
    std::cout << "L2 Norm Results:" << std::endl;
    std::cout << "GPU: " << gpu_result << std::endl;
    std::cout << "CPU: " << cpu_result << std::endl;
    
    if (std::abs(gpu_result - cpu_result) < 1e-4) {
        std::cout << "Verification PASSED" << std::endl;
    } else {
        std::cout << "Verification FAILED" << std::endl;
    }

}