// matmult.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey
// Based on code from the CUDA Programming Guide

// DO NOT MODIFY FOR THE ASSIGNMENT

// Includes
#include <stdio.h>
#include "timer.h"
#include "matmultKernel.h"

// Defines
#define epsilon (float)1e-4
#define verbose 0

Matrix MakeDeviceMatrix(Matrix M, bool copy){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.stride = M.width;
  newDeviceMatrix.height = M.height;
  size_t size = M.width * M.height * sizeof(float);
  cudaMalloc((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

// Host code for matrix multiplication.
// Matrix dimensions must be multiples of size 
// This code assumes that the matrix is square.
void MatMul(const Matrix A, const Matrix B, Matrix C, int dimension){

  // Create device data structures.
  Matrix device_A = MakeDeviceMatrix(A, true);
  Matrix device_B = MakeDeviceMatrix(B, true);
  Matrix device_C = MakeDeviceMatrix(C, false);

  // Define grid topology
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(B.width/dimension, A.height/dimension);

  // Invoke kernel for warm up
  MatMulKernel<<<dimGrid, dimBlock>>>(device_A, device_B, device_C);

  // Synchronize to make sure everyone is done in the warmup.
  cudaThreadSynchronize();

  // Set up timer
  initialize_timer();
  start_timer();


  // Invoke kernel for real
  MatMulKernel<<<dimGrid, dimBlock>>>(device_A, device_B, device_C);
 
  // Synchronize to make sure everyone is done.
  cudaThreadSynchronize() ;

  // Compute and report the timing results

  stop_timer();
  double time = elapsed_time();

  double nFlops = (double)A.width*A.height*B.width*2;
  double nFlopsPerSec = nFlops/time;
  double nGFlopsPerSec = nFlopsPerSec*1e-9;
  printf( "Data dimensions: %dx%d \n", C.height, C.width);
  printf( "Grid Dimensions: %dx%d \n",dimGrid.x,dimGrid.y);
  printf( "Block Dimensions: %dx%d \n",dimBlock.x,dimBlock.y);
  printf( "Footprint Dimensions: %dx%d \n",FOOTPRINT_SIZE,FOOTPRINT_SIZE);
  
  printf( "Time: %lf (sec), nFlops: %0.0lf, GFlopsS: %lf\n",
            time, nFlops, nGFlopsPerSec);

  // Copy the result to the host memory from device memory
  size_t size = C.width * C.height * sizeof(float);
  cudaMemcpy(C.elements, device_C.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(device_A.elements);
  cudaFree(device_B.elements);
  cudaFree(device_C.elements);
   
}


// Create a matrix in host memory.
Matrix MakeHostMatrix(int width, int height){
  Matrix newHostMatrix;
  newHostMatrix.width = width;
  newHostMatrix.height = height;
  size_t size = newHostMatrix.width * newHostMatrix.height * sizeof(float);
  newHostMatrix.elements = (float*)malloc(size);
  return newHostMatrix;
}

// Print a matrix stored in host memory.
void printMatrix(Matrix M, const char* name) {
  printf("\n%s \n",name);
  for(int y=0; y<M.height; y++){
   for(int x=0; x<M.width; x++) {
      printf("%f ", M.elements[y * M.width + x]);
   }
   printf("\n");
  }
}

// Initialize dummy data in a matrix stored in host memory.
void initMatrix(Matrix M, bool horizontal) {
  for(int y=0; y<M.height; y++) {
    for(int x=0; x<M.width; x++) {
      M.elements[y*M.width+x] = (float)(horizontal?x:y);
    }
  }
}

// Check the specified matrix to be sure it is correct.
// That is, make sure it is the result of multiplying the
// dummy data we created earlier.
void checkResult(Matrix M) {

  Matrix correct = MakeHostMatrix(M.width, M.height);

  for(int y=0; y<M.height; y++) {
    for(int x=0; x<M.width; x++) {
       correct.elements[y*correct.width+x] = (float)M.width*(float)x*y;
    }
  }

  if(verbose){
   // print correct
   printMatrix(correct, "correct");

   // print host_C
   printMatrix(M, "result");
  }


  double maxerror = 0.0;
  int errCnt = 0;
  for(int y=0; y<correct.height; y++) {
    for(int x=0; x<correct.width; x++) {
      float it = correct.elements[y*correct.width+x];
      if(fabs(it - M.elements[y*M.width+x])> epsilon*it) {
        errCnt++;
        double error = fabs(it - M.elements[y*M.width+x])/it;
        if (error > maxerror) maxerror = error;
      }      
    }
  }

  if(errCnt>0){
    printf("\n\nTEST FAILED: number of errors:  %d, max rel error: %f\n", errCnt, maxerror);
  }
  
  free(correct.elements);
}

//
// main
//
int main(int argc, char** argv) {

  // Grid dimension
  int num_blocks;
  // Matrix dimensions in multiples of FOOTPRINT_SIZE
  // Matrices will be of size data_size * data_size
  int data_size;

  // Read command line argument
  if(argc == 2){
    sscanf(argv[1], "%d", &num_blocks);
    data_size = num_blocks * FOOTPRINT_SIZE;
  } else {
     printf("Usage: %s NumBlocks\n", argv[0]);
     exit(0);
  }     

  // Create matrices in host.
  Matrix host_A = MakeHostMatrix(data_size, data_size);
  Matrix host_B = MakeHostMatrix(data_size, data_size);
  Matrix host_C = MakeHostMatrix(data_size, data_size);

  // Initialize values in host A and B
  initMatrix(host_A,false);
  initMatrix(host_B,true);
 
  // debugging
  if(verbose){
    printMatrix(host_A, "host_A");
    printMatrix(host_B, "host_B");
  }

  // Perform CUDA matrix Multiplication
  // MatMul is a host function that calls
  // the device kernel MatMulKernel and
  // times its performance.
  MatMul(host_A,host_B,host_C,FOOTPRINT_SIZE);

  // Verify that the result is correct.
  checkResult(host_C);
  
  // Free allocated memory.
  free(host_A.elements);
  free(host_B.elements);
  free(host_C.elements);
}

