#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

#define N 1024

__global__ void dropout(float* x, float* mask, float p, int n, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        float m = (rand_val < p) ? 0.0f : 1.0f;
        mask[idx] = m;
        x[idx] *= m;
    }
}

int main() {
    float *h_x, *h_mask;
    float *d_x, *d_mask;
    float p = 0.5f;
    unsigned long long seed = 1234ULL;
    int size = N * sizeof(float);

    h_x = (float*)malloc(size);
    h_mask = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
    }

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_mask, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    dropout<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_mask, p, N, seed);

    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask, d_mask, size, cudaMemcpyDeviceToHost);

    int dropped_count = 0;
    for (int i = 0; i < N; ++i) {
        if (h_mask[i] == 0.0f) {
            dropped_count++;
            if (h_x[i] != 0.0f) {
                std::cerr << "Mismatch at " << i << ": mask is 0 but x is " << h_x[i] << std::endl;
            }
        } else {
            if (h_x[i] != 1.0f) { // Since initialized to 1.0
                 std::cerr << "Mismatch at " << i << ": mask is 1 but x is " << h_x[i] << std::endl;
            }
        }
    }

    std::cout << "Total elements: " << N << std::endl;
    std::cout << "Dropped elements: " << dropped_count << std::endl;
    std::cout << "Dropout rate: " << (float)dropped_count / N << " (Target: " << p << ")" << std::endl;

    free(h_x);
    free(h_mask);
    cudaFree(d_x);
    cudaFree(d_mask);

}
