#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cudnn.h>

using namespace std;
using namespace std::chrono;

constexpr int H = 1024, W = 1024, C = 3, FH = 3, FW = 3, K = 64;

void initialize_data(double *I, double *F) {
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
}

int main() {
    double *I = new double[C * H * W];
    double *F = new double[K * C * FH * FW];
    double *O = new double[K * H * W];

    initialize_data(I, F);

    double *d_I, *d_F, *d_O;
    cudaMalloc((void **)&d_I, C * H * W * sizeof(double));
    cudaMalloc((void **)&d_F, K * C * FH * FW * sizeof(double));
    cudaMalloc((void **)&d_O, K * H * W * sizeof(double));

    cudaMemcpy(d_I, I, C * H * W * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);

    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W);

    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

    cudnnConvolutionFwdAlgo_t conv_algorithm;
    cudnnConvolutionFwdAlgoPerf_t conv_algo_perf;
    int returned_algo_count;
    cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor,
                                        1, &returned_algo_count, &conv_algo_perf);

    conv_algorithm = conv_algo_perf.algo;

    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, conv_descriptor,
                                            output_descriptor, conv_algorithm, &workspace_size);

    void *d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    double alpha = 1.0, beta = 0.0;
    auto start = high_resolution_clock::now();
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_I, filter_descriptor, d_F, conv_descriptor,
                            conv_algorithm, d_workspace, workspace_size, &beta, output_descriptor, d_O);
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
    cout << "Time taken for the cuDNN convolution: " << duration << " microseconds" << endl;

    delete[] I;
    delete[] F;
    delete[] O;

    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}