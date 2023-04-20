#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define TILE_SIZE 16

__global__ void tiled_convolution(const double *I0, const double *F, double *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    __shared__ double shared_I0[C * (TILE_SIZE + FW - 1) * (TILE_SIZE + FH - 1)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load input tile into shared memory
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < ceil((float)(TILE_SIZE + FW - 1) / TILE_SIZE); i++) {
            for (int j = 0; j < ceil((float)(TILE_SIZE + FH - 1) / TILE_SIZE); j++) {
                int xIndex = i * blockDim.x + tx;
                int yIndex = j * blockDim.y + ty;
                if (xIndex < TILE_SIZE + FW - 1 && yIndex < TILE_SIZE + FH - 1) {
                    int xGlobal = x - tx + xIndex;
                    int yGlobal = y - ty + yIndex;
                    if (xGlobal >= 0 && xGlobal < W && yGlobal >= 0 && yGlobal < H) {
                        shared_I0[c * (TILE_SIZE + FW - 1) * (TILE_SIZE + FH - 1) + xIndex * (TILE_SIZE + FH - 1) + yIndex] = I0[c * (W + 2) * (H + 2) + xGlobal * (H + 2) + yGlobal];
                    } else {
                        shared_I0[c * (TILE_SIZE + FW - 1) * (TILE_SIZE + FH - 1) + xIndex * (TILE_SIZE + FH - 1) + yIndex] = 0;
                    }
                }
            }
        }
    }

    __syncthreads();

    if (x < W && y < H) {
        double sum = 0;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FW; i++) {
                for (int j = 0; j < FH; j++) {
                    sum += F[k * C * FW * FH + c * FW * FH + (FW - 1 - i) * FH + (FH - 1 - j)] * shared_I0[c * (TILE_SIZE + FW - 1) * (TILE_SIZE + FH - 1 + (tx + i) * (TILE_SIZE + FH - 1) + (ty + j)];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

int main() {
    // ... (same as previous example, except for the kernel function name) ...

    // Set the kernel's execution configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    // Launch the kernel and measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    tiled_convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // ... (same as previous example, except for the kernel function name) ...

    return 0;
}