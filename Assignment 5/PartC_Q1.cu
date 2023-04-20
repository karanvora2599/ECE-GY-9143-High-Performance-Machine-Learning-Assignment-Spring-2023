#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64

__global__ void simple_convolution(const double *I0, const double *F, double *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (x < W && y < H) {
        double sum = 0;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FW; i++) {
                for (int j = 0; j < FH; j++) {
                    sum += F[k * C * FW * FH + c * FW * FH + (FW - 1 - i) * FH + (FH - 1 - j)] * I0[c * (W + 2) * (H + 2) + (x + i) * (H + 2) + (y + j)];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

int main() {
    double *I, *F, *I0, *O;
    double *d_I0, *d_F, *d_O;
    size_t input_size = C * H * W * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t padded_input_size = C * (W + 2) * (H + 2) * sizeof(double);
    size_t output_size = K * H * W * sizeof(double);

    // Allocate memory for the input, filter, and output tensors on the CPU
    cudaMallocHost((void **)&I, input_size);
    cudaMallocHost((void **)&F, filter_size);
    cudaMallocHost((void **)&I0, padded_input_size);
    cudaMallocHost((void **)&O, output_size);

    // Generate the input tensor I and filter F
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }

    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
                    }
    }

    // Generate the padded input tensor I0
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < W + 2; x++) {
            for (int y = 0; y < H + 2; y++) {
                if (x == 0 || x == W + 1 || y == 0 || y == H + 1) {
                    I0[c * (W + 2) * (H + 2) + x * (H + 2) + y] = 0;
                } else {
                    I0[c * (W + 2) * (H + 2) + x * (H + 2) + y] = I[c * H * W + (x - 1) * W + (y - 1)];
                }
            }
        }
    }

    // Allocate memory for the input, filter, and output tensors on the GPU
    cudaMalloc((void **)&d_I0, padded_input_size);
    cudaMalloc((void **)&d_F, filter_size);
    cudaMalloc((void **)&d_O, output_size);

    // Copy the input tensor and filter to the GPU
    cudaMemcpy(d_I0, I0, padded_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, filter_size, cudaMemcpyHostToDevice);

    // Set the kernel's execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    // Launch the kernel and measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    simple_convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy the output tensor back to the CPU
    cudaMemcpy(O, d_O, output_size, cudaMemcpyDeviceToHost);

    // Calculate the checksum
    double checksum = 0;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }

    std::cout << "Checksum: " << checksum << std::endl;

    // Free the allocated memory on the GPU and CPU
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFreeHost(I);
    cudaFreeHost(F);
    cudaFreeHost(I0);
    cudaFreeHost(O);

    return 0;
}