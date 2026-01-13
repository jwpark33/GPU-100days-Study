#include <iostream>
#include <cuda_runtime.h>

__global__ void Sigmoid(float *input, float *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}

int main() {
    const int size = 16;
    float *input = new float[size];
    float *output = new float[size];
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < size; i++) {
        input[i] = (i);
    }
    
    // Launch kernel
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    Sigmoid<<<1, size>>>(d_input, d_output, size);
    
    // Copy result back to host
    
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print result
    for (int i = 0; i < size; i++) {
        std::cout << "Input: " << input[i] << ", Output: " << output[i] << std::endl;
    } 
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input;
    delete[] output;
}
