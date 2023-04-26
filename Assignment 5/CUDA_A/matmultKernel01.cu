/// Multiplies two matrices using CUDA: A x B = C
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  int halfFootprint = FOOTPRINT_SIZE / 2;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes four elements of Csub in its copies of Cvalues
  float Cvalue00 = 0;
  float Cvalue01 = 0;
  float Cvalue10 = 0;
  float Cvalue11 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Each thread copies four elements of shared_A and four elements of shared_B
    shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

    shared_A[thread_row][thread_col + halfFootprint] = Asub[thread_row * A.stride + thread_col + halfFootprint];
    shared_B[thread_row][thread_col + halfFootprint] = Bsub[thread_row * B.stride + thread_col + halfFootprint];

    shared_A[thread_row + halfFootprint][thread_col] = Asub[(thread_row + halfFootprint) * A.stride + thread_col];
    shared_B[thread_row + halfFootprint][thread_col] = Bsub[(thread_row + halfFootprint) * B.stride + thread_col];

    shared_A[thread_row + halfFootprint][thread_col + halfFootprint] = Asub[(thread_row + halfFootprint) * A.stride + thread_col + halfFootprint];
    shared_B[thread_row + halfFootprint][thread_col + halfFootprint] = Bsub[(thread_row + halfFootprint) * B.stride + thread_col + halfFootprint];

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of two rows of shared_A and two cols of shared_B
    // computing four Cvalues by accumulation
#pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e) {
       Cvalue00 += shared_A[thread_row][e] * shared_B[e][thread_col];
       Cvalue01 += shared_A[thread_row][e] * shared_B[e][thread_col + halfFootprint];
       Cvalue10 += shared_A[thread_row + halfFootprint][e] * shared_B[e][thread_col];
       Cvalue11 += shared_A[thread_row + halfFootprint][e] * shared_B[e][thread_col + halfFootprint];
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own four cell values.
  Csub[thread_row * C.stride + thread_col] = Cvalue00;
  Csub[thread_row * C.stride + thread_col + halfFootprint] = Cvalue01;
  Csub[(thread_row + halfFootprint) * C.stride + thread_col] = Cvalue10;
  Csub[(thread_row + halfFootprint) * C.stride + thread_col + halfFootprint] = Cvalue11;
}
