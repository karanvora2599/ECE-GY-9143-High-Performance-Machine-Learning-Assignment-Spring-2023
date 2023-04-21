#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void convolution_kernel(const double *I0, const double *F, double *O, int H, int W, int C, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && k < K) {
        double sum = 0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum += F[k * C * 9 + c * 9 + (2 - i) * 3 + (2 - j)] * I0[c * (W + 2) * (H + 2) + (x + i) * (H + 2) + (y + j)];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

int main() {
    int H = 1024, W = 1024, C = 3, FW = 3, FH = 3, K = 64, P = 1;

    double *I = new double[C * H * W];
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }

    double *F = new double[K * C * FH * FW];
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    double *I0 = new double[C * (H + 2 * P) * (W + 2 * P)]();
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I0[c * (H + 2 * P) * (W + 2 * P) + (x + P) * (H + 2 * P) + (y + P)] = I[c * H * W + x * W + y];
            }
        }
    }

    double *I0_device, *F_device, *O_device;
    cudaMalloc(&I0_device, C * (H + 2 * P) * (W + 2 * P) * sizeof(double));
    cudaMalloc(&F_device, K * C * FH * FW * sizeof(double));
    cudaMalloc(&O_device, K * H * W * sizeof(double));

    cudaMemcpy(I0_device, I0, C * (H + 2 * P) * (W + 2 * P) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_device, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 4);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (K + blockDim.z - 1) / blockDim.z);

    auto start = std::chrono::high_resolution_clock::now();
    convolution_kernel<<<gridDim, blockDim>>>(I0_device, F_device, O_device, H, W, C, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double *O = new double[K * H * W];
    cudaMemcpy(O, O_device, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    double checksum = 0;
    for (int i = 0; i < K * H * W; ++i) {
        checksum += O[i];
    }

    std::cout << "Checksum: " << checksum << std::endl;
    std::cout << "Kernel execution time: " << elapsed.count() << " seconds" << std::endl;

    delete[] I;
    delete[] F;
    delete[] I0;
    delete[] O;

    cudaFree(I0_device);
    cudaFree(F_device);
    cudaFree(O_device);

    return 0;
}