// vecAddKernel_coalesced.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// This Kernel adds two Vectors A and B in C on GPU
// using coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = global_index; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}