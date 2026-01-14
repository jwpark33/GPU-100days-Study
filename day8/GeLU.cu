#include <iostream>
#include <cuda_runtime.h>

__global__ void GeLU(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * powf(x, 3.0))));
    }
}

int main() {
    int size = 16;
    float *h_input;
    float *h_output;
    float *d_input;
    float *d_output;

    h_input = (float*)malloc(size * sizeof(float));
    h_output = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_input[i] = i;
    }

    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    GeLU<<<1, size>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

}
