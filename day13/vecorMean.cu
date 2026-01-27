#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void vectorMean(float *A, float *B, int width) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (; i < width; i += stride) {
        sum += A[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(B, sdata[0] / width);
    }
}

int main() {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float h_B = 0.0f;

    srand(time(0));
    float host_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        host_sum += h_A[i];
    }
    float expected = host_sum / N;

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, sizeof(float));

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    vectorMean<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, N);

    cudaMemcpy(&h_B, d_B, sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "Calculated Mean: " << h_B << std::endl;
    std::cout << "Expected Mean: " << expected << std::endl;

    if (std::abs(h_B - expected) < 1e-5) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);

}
