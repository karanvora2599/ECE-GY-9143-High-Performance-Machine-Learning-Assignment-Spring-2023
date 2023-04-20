#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

__global__ void addArrays(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int K[] = {1, 5, 10, 50, 100};
    const int BLOCK_SIZE = 256;

    for (int k = 0; k < 5; k++) {
        const int n = K[k] * 1000000;

        int* A, *B, *C;
        cudaMallocManaged(&A, n * sizeof(int));
        cudaMallocManaged(&B, n * sizeof(int));
        cudaMallocManaged(&C, n * sizeof(int));

        for (int i = 0; i < n; i++) {
            A[i] = i;
            B[i] = i;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Scenario 1: using one block with 1 thread
        cudaEventRecord(start);
        addArrays<<<1, 1>>>(A, B, C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time1;
        cudaEventElapsedTime(&time1, start, stop);

        // Scenario 2: using one block with 256 threads
        cudaEventRecord(start);
        addArrays<<<1, BLOCK_SIZE>>>(A, B, C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time2;
        cudaEventElapsedTime(&time2, start, stop);

        // Scenario 3: using multiple blocks with 256 threads per block
        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(start);
        addArrays<<<numBlocks, BLOCK_SIZE>>>(A, B, C, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time3;
        cudaEventElapsedTime(&time3, start, stop);

        cout << "K=" << K[k] << endl;
        cout << "Scenario 1: " << time1 << " ms" << endl;
        cout << "Scenario 2: " << time2 << " ms" << endl;
        cout << "Scenario 3: " << time3 << " ms" << endl;

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    return 0;
}
