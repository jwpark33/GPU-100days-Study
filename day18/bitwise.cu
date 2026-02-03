#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Quantization Kernel
// float 입력을 int로 변환하여 비트 시프트(양자화) 수행
// Loss가 발생하는 것을 명확히 확인 가능
__global__ void quantization_kernel(float *input, int *quantized, float *reconstructed, int shift_bits) {
    int tid = threadIdx.x;
    
    float val = input[tid];
    
    int val_int = (int)val; 
    int q = val_int >> shift_bits;
    quantized[tid] = q;
    reconstructed[tid] = (float)(q << shift_bits);
}

int main() {
    const int N = 16;
    const int SHIFT_BITS = 2;
    
    float *d_input, *d_reconstructed;
    int *d_quantized;
    
    float h_input[N];
    int h_quantized[N];
    float h_reconstructed[N];
    
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 3.123;
    }

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_quantized, N * sizeof(int));
    cudaMalloc(&d_reconstructed, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    quantization_kernel<<<1, N>>>(d_input, d_quantized, d_reconstructed, SHIFT_BITS);

    cudaMemcpy(h_quantized, d_quantized, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reconstructed, d_reconstructed, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Shift Bits: " << SHIFT_BITS << " (Quantization Step: " << (1 << SHIFT_BITS) << ")\n";
    std::cout << "Idx\tInput\tInt(Cast)\tQuant(>>)\tRecon(<<)\tLoss\n";
    std::cout << "------------------------------------------------------------------------\n";
    
    for (int i = 0; i < N; i++) {
        float diff = h_input[i] - h_reconstructed[i];
        int cast_int = (int)h_input[i];

        std::cout << i << "\t" 
                  << h_input[i] << "\t"
                  << cast_int << "\t\t"
                  << h_quantized[i] << "\t\t"
                  << h_reconstructed[i] << "\t\t"
                  << diff << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_reconstructed);
    
}