///
/// vecAddKernel.h
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// Kernels written for use with this header
/// add two Vectors A and B in C on GPU
/// 


__global__ void AddVectors(const float* A, const float* B, float* C, int N);

