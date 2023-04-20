#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    double *I, *F, *O;
    double *d_I, *d_F, *d_O;

    size_t input_size = C * H * W * sizeof(double);
    size_t filter_size = K * C * FH * FW * sizeof(double);
    size_t output_size = K * H * W * sizeof(double);

    // Allocate and initialize the input, filter, and output tensors on the CPU
    cudaMallocHost((void **)&I, input_size);
    cudaMallocHost((void **)&F, filter_size);
    cudaMallocHost((void **)&O, output_size);

    // Generate the input tensor I and filter F
    // ... (same as previous example) ...

    // Allocate memory for the input, filter, and output tensors on the GPU
    cudaMalloc((void **)&d_I, input_size);
    cudaMalloc((void **)&d_F, filter_size);
    cudaMalloc((void **)&d_O, output_size);

    // Copy the input tensor and filter to the GPU
    cudaMemcpy(d_I, I, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, filter_size, cudaMemcpyHostToDevice);

    // Create the necessary cuDNN descriptors
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    // Set the descriptors' properties
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W);
    cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    // Choose the fastest convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_    descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    // Allocate workspace memory
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, algo, &workspace_size);
    void *d_workspace;
    cudaMalloc((void **)&d_workspace, workspace_size);

    // Perform the convolution using the cuDNN library function and measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    double alpha = 1.0, beta = 0.0;
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_I, filter_descriptor, d_F, convolution_descriptor, algo, d_workspace, workspace_size, &beta, output_descriptor, d_O);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy the output tensor back to the CPU
    cudaMemcpy(O, d_O, output_size, cudaMemcpyDeviceToHost);

    // Calculate the checksum
    double checksum = 0;
    for (int i = 0; i < K * H * W; i++) {
        checksum += O[i];
    }
    std::cout << "Checksum: " << checksum << std::endl;

    // Measure the execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Free the allocated memory on the GPU and CPU
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(d_workspace);
    cudaFreeHost(I);
    cudaFreeHost(F);
    cudaFreeHost(O);

    // Destroy the cuDNN handles and descriptors
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}