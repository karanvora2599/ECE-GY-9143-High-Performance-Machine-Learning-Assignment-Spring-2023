// vecAddKernel.h
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey

// Kernels written for use with this header
// add two Vectors A and B in C on GPU



__global__ void AddVectors(const float* A, const float* B, float* C, int N);

