#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define TILE_HEIGHT 32

__global__ void stridedLoad(float *A, float *B, int width, int height) {
    // Padding to avoid shared memory bank conflicts
    __shared__ float tile[TILE_WIDTH][TILE_HEIGHT + 1];
    
    // Calculate global indices for loading (Row-major)
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    
    // Load data from A into shared memory
    // Data is loaded in a coalesced manner (consecutive x -> consecutive threads)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    }
    
    __syncthreads();
    
    int x_out = blockIdx.x * TILE_WIDTH + threadIdx.y;
    int y_out = blockIdx.y * TILE_HEIGHT + threadIdx.x;
    
    if (x_out < width && y_out < height) {
        // Read from shared memory with swapped indices to achieve transpose
        // We need A[y_out][x_out] which is at tile[threadIdx.x][threadIdx.y]
        B[x_out * height + y_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    
    for (int i = 0; i < width * height; ++i) {
        h_A[i] = (float)i;
    }
    
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 numBlocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    stridedLoad<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width, height);
    
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    
    // Verify Transpose (B should be A transposed)
    // A[y * width + x] should be at B[x * height + y]
    bool correct = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float valA = h_A[y * width + x];
            float valB = h_B[x * height + y];
            if (valA != valB) {
                correct = false;
                std::cout << "Mismatch at x=" << x << " y=" << y << " A=" << valA << " B=" << valB << std::endl;
                break;
            }
        }
        if (!correct) break;
    }
    
    if (correct) {
        std::cout << "Transpose successful!" << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

}
