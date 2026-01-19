#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>


__global__ void vecAdd_boundary_check(float *a, float *b, float *c, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (id < N) {
        c[id] = a[id] + b[id];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 10000; // 10000 is not a multiple of 256 (10000 % 256 = 16)

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Total Threads: " << blockSize * gridSize << std::endl;

    vecAdd_boundary_check<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": Expected " << expected << ", Got " << h_c[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Success! Vector addition with boundary handling verified." << std::endl;
    } else {
        std::cout << "Verification with boundary handling Failed." << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

}