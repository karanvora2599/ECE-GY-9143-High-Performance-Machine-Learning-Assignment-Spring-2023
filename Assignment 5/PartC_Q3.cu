#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>

#define C 3
#define H 1024
#define W 1024
#define K 64
#define FW 3
#define FH 3
#define P 1

void generate_input(double *I, double *F) {
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
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

    generate_input(I, F);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);

    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, P, P, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor, &out_n, &out_c, &out_h, &out_w);

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w);

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm);

    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_size);

    void *workspace;
    cudaMalloc(&workspace, workspace_size);

    double *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, C * H * W * sizeof(double));
    cudaMalloc((void **)&d_kernel, K * C * FH * FW * sizeof(double));
    cudaMalloc((void **)&d_output, K * H * W * sizeof(double));

    cudaMemcpy(d_input, I, C * H * W * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.0;
    double beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel, convolution_descriptor, convolution_algorithm, workspace, workspace_size, &beta, output_descriptor, d_output);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convolution execution time: " << elapsed.count() << " seconds" << std::endl;

    cudaMemcpy(O, d_output, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    double checksum = 0;
    for (int i = 0; i < K * H * W; ++i) {
        checksum += O[i];
    }
    std::cout << "Checksum: " << checksum << std::endl;

    cudaFree(workspace);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    delete[] I;
    delete[] F;
    delete[] O;

    return 0;
}