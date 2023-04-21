#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define H 1024
#define W 1024
#define C 3
#define K 64
#define FW 3
#define FH 3
#define P 1
#define TILE_WIDTH 16

__global__ void tiled_convolution(const double *I0, const double *F, double *O) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    __shared__ double I0_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ double F_tile[TILE_WIDTH][TILE_WIDTH];

    double sum = 0;
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < FW; i++) {
            for (int j = 0; j < FH; j++) {
                int xI0 = x + i - P;
                int yI0 = y + j - P;

                if (xI0 >= 0 && xI0 < W && yI0 >= 0 && yI0 < H) {
                    for (int c = 0; c < C; c++) {
                        I0_tile[ty][tx] = I0[c * H * W + yI0 * W + xI0];
                        F_tile[ty][tx] = F[k * C * FH * FW + c * FH * FW + (FH - 1 - j) * FW + (FW - 1 - i)];

                        __syncthreads();

                        sum += I0_tile[ty][tx] * F_tile[ty][tx];
                    }
                }
            }
        }
        O[k * H * W + y * W + x] = sum;
    }
}

int main() {
    double *I = new double[C * H * W];
    double *F = new double[K * C * FH * FW];
    double *I0 = new double[C * (H + 2 * P) * (W + 2 * P)];
    double *O;

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

    for (int c = 0; c < C; c++) {
        for (int x = 0; x < H + 2 * P; x++) {
            for (int y = 0; y < W + 2 * P; y++) {
                if (x >= P && x < H + P && y >= P && y < W + P) {
                    I0[c * (H + 2 * P  * (W + 2 * P) + x * (W + 2 * P) + y] = I[c * H * W + (x - P) * W + (y - P)];
                } else {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = 0;
                }
            }
        }
    }

    double *I0_device, *F_device, *O_device;
    cudaMalloc(&I0_device, C * (H + 2 * P) * (W + 2 * P) * sizeof(double));
    cudaMalloc(&F_device, K * C * FH * FW * sizeof(double));
    cudaMalloc(&O_device, K * H * W * sizeof(double));

    cudaMemcpy(I0_device, I0, C * (H + 2 * P) * (W + 2 * P) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(F_device, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((H + blockDim.x - 1) / blockDim.x, (W + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::high_resolution_clock::now();
    tiled_convolution<<<gridDim, blockDim>>>(I0_device, F_device, O_device);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    O = new double[K * H * W];
    cudaMemcpy(O, O_device, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    double checksum = 0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                checksum += O[k * H * W + x * W + y];
            }
        }
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