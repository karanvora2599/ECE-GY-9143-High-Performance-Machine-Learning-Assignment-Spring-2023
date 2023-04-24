#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

constexpr int H = 1024, W = 1024, C = 3, FH = 3, FW = 3, K = 64;
constexpr int TILE_SIZE = 32;
constexpr int SHARED_I0_SIZE = (TILE_SIZE + FW - 1) * (TILE_SIZE + FH - 1);

__global__ void tiled_convolution_kernel(double *I0, double *F, double *O) {
    // ... (Same tiled_convolution_kernel code as before)
}

void initialize_data(double *I, double *I0, double *F) {
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
        for (int x = 0; x < H + 2; x++) {
            for (int y = 0; y < W + 2; y++) {
                if (x == 0 || y == 0 || x == H + 1 || y == W + 1) {
                    I0[c * (H + 2) * (W + 2) + x * (W + 2) + y] = 0.0;
                } else {
                    I0[c * (H + 2) * (W + 2) + x * (W + 2) + y] = I[c * H * W + (x - 1) * W + (y - 1)];
                }
            }
        }
    }
}

int main() {
    double *I = new double[C * H * W];
    double *I0 = new double[C * (H + 2) * (W + 2)];
    double *F = new double[K * C * FH * FW];
    double *O = new double[K * H * W];

    initialize_data(I, I0, F);

    double *d_I0, *d_F, *d_O;
    cudaMalloc((void **)&d_I0, C * (H + 2) * (W + 2) * sizeof(double));
    cudaMalloc((void **)&d_F, K * C * FH * FW * sizeof(double));
    cudaMalloc((void **)&d_O, K * H * W * sizeof(double));

    cudaMemcpy(d_I0, I0, C * (H + 2) * (W + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    auto start = high_resolution_clock::now();
    tiled_convolution_kernel<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    cudaMemcpy(O, d_O, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < H; x++) {
            for (int y = 0; y < W; y++) {
                checksum += O[k * H * W + x * W + y];
            }
        }
    }

    cout << "Checksum: " << checksum << endl;

    auto duration = duration_cast<microseconds>(stop - start).count();
    cout << "Time taken for the tiled CUDA kernel: " << duration << " microseconds" << endl;

    delete[] I;
    delete[] I0;
    delete[] F;
    delete[] O;

    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}