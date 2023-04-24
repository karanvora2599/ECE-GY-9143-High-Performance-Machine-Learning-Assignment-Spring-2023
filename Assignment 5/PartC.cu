#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda.h>
#include <cudnn.h>
#include <cudnn_version.h>

#define H 1024
#define W 1024
#define C 3
#define K 64
#define FW 3
#define FH 3
#define P 1 // padding
#define TILE_SIZE 16 // tile size for tiling convolution

double checksum(double* data, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

__global__ void simpleConvolution(double* I, double* F, double* O) {
    int k = blockIdx.x;
    int x = threadIdx.x + blockDim.x * blockIdx.y;
    int y = threadIdx.y + blockDim.y * blockIdx.z;

    if (x < W && y < H) {
        double sum = 0.0;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FW; i++) {
                for (int j = 0; j < FH; j++) {
                    sum += F[k * C * FW * FH + c * FW * FH + (FW - 1 - i) * FH + (FH - 1 - j)] *
                           I[c * (W + 2 * P) * (H + 2 * P) + (x + i) * (H + 2 * P) + (y + j)];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

__global__ void tiledConvolution(double* I, double* F, double* O) {
    __shared__ double I_shared[C][(TILE_SIZE + FW - 1)][(TILE_SIZE + FH - 1)];

    int k = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x0 = blockIdx.y * TILE_SIZE;
    int y0 = blockIdx.z * TILE_SIZE;

    double sum = 0.0;

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < FW; i++) {
            for (int j = 0; j < FH; j++) {
                int x = x0 + tx + i - P;
                int y = y0 + ty + j - P;
                if (x >= 0 && x < W && y >= 0 && y < H) {
                    I_shared[c][tx + i][ty + j] = I[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y];
                } else {
                    I_shared[c][tx + i][ty + j] = 0.0;
                }
            }
        }
    }

    __syncthreads();

    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FW; i++) {
                for (int j = 0; j < FH; j++) {
                    sum += F[k * C * FW * FH + c * FW * FH + (FW - 1 - i) * FH + (FH - 1 - j)] * I_shared[c][tx + i][ty + j];
                }
            }
        }
        O[k * W * H + (x0 + tx) * H + (y0 + ty)] = sum;
    }
}

void initialize_data(double* I, double* F, int h, int w, int c, int fh, int fw, int k) {
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            for (int ch = 0; ch < c; ch++) {
                I[ch * (w + 2 * P) * (h + 2 * P) + (x + P) * (h + 2 * P) + (y + P)] = ch * (x + y);
            }
        }
    }

    for (int x = 0; x < fw; x++) {
        for (int y = 0; y < fh; y++) {
            for (int ch = 0; ch < c; ch++) {
                for (int filter = 0; filter < k; filter++) {
                    F[filter * c * fw * fh + ch * fw * fh + x * fh + y] = (ch + filter) * (x + y);
                }
            }
        }
    }
}

int main() {
    double* I = new double[C * (W + 2 * P) * (H + 2 * P)];
    double* F = new double[K * C * FW * FH];
    double* O1 = new double[K * W * H];
    double* O2 = new double[K * W * H];

    initialize_data(I, F, H, W, C, FH, FW, K);

    double* d_I, *d_F, *d_O;
    cudaMalloc(&d_I, sizeof(double) * C * (W + 2 * P) * (H + 2 * P));
    cudaMalloc(&d_F, sizeof(double) * K * C * FH * FW);
    cudaMalloc(&d_O, sizeof(double) * K * W * H);
    cudaMemcpy(d_I, I, sizeof(double) * C * (W + 2 * P) * (H + 2 * P), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, sizeof(double) * K * C * FH * FW, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    simpleConvolution<<<dim3(K, (W + 15) / 16, (H + 15) / 16), dim3(16, 16, 1)>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    cudaMemcpy(O1, d_O, sizeof(double) * K * W * H, cudaMemcpyDeviceToHost);
    std::cout << "C1_Checksum: " << checksum(O1, K * W * H) << ", C1_Execution_Time: " << std::fixed << std::setprecision(3) << duration.count() << " Milliseconds" << std::endl;

    // execute tiled convolution kernel
    start = std::chrono::high_resolution_clock::now();
    tiledConvolution<<<dim3(K, (W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE), dim3(TILE_SIZE + FW - 1, TILE_SIZE + FH - 1, 1)>>>(d_I, d_F, d_O);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cudaMemcpy(O2, d_O, sizeof(double) * K * W * H, cudaMemcpyDeviceToHost);
    std::cout << "C2_Checksum: " << checksum(O2, K * W * H) << ", C2_Execution_Time: " << std::fixed << std::setprecision(3) << duration.count() << " Milliseconds" << std::endl;

    // execute cuDNN convolution
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, P, P, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, W + 2 * P, H + 2 * P);
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, W, H);
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount;
    cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDesc, filterDesc, convDesc, outputDesc, 1, &returnedAlgoCount, &algoPerf);
    cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;
    size_t workspaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize);
    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspaceSize);
    double alpha = 1.0, beta = 0.0;

    start = std::chrono::high_resolution_clock::now();
    cudnnConvolutionForward(handle, &alpha, inputDesc, d_I, filterDesc, d_F, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_O);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cudaMemcpy(O1, d_O, sizeof(double) * K * W * H, cudaMemcpyDeviceToHost);
    std::cout << "C3_Checksum: " << checksum(O1, K * W * H) << ", C3_Execution_Time: " << std::fixed << std::setprecision(3) << duration.count() << " Milliseconds" << std::endl;
    // free memory
    delete[] I;
    delete[] F;
    delete[] O1;
    delete[] O2;
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(d_workspace);

    return 0;
}