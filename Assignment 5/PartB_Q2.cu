#include <iostream>
#include <chrono>

__global__ void addArrays(int* a, int* b, int* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int K = 1; // change K to test different values

    // allocate memory on host
    int* array1 = new int[K * 1000000];
    int* array2 = new int[K * 1000000];
    int* result = new int[K * 1000000];

    // initialize arrays
    for (int i = 0; i < K * 1000000; i++) {
        array1[i] = i;
        array2[i] = i * 2;
    }

    // allocate memory on device
    int* d_array1, * d_array2, * d_result;
    cudaMalloc(&d_array1, K * 1000000 * sizeof(int));
    cudaMalloc(&d_array2, K * 1000000 * sizeof(int));
    cudaMalloc(&d_result, K * 1000000 * sizeof(int));

    // copy data from host to device
    cudaMemcpy(d_array1, array1, K * 1000000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2, K * 1000000 * sizeof(int), cudaMemcpyHostToDevice);

    // perform addition on device
    auto start_time = std::chrono::high_resolution_clock::now(); // start timer
    // scenario 1: using one block with 1 thread
    //addArrays<<<1, 1>>>(d_array1, d_array2, d_result, K * 1000000);
    // scenario 2: using one block with 256 threads
    //addArrays<<<1, 256>>>(d_array1, d_array2, d_result, K * 1000000);
    // scenario 3: using multiple blocks with 256 threads per block
    addArrays<<<(K * 1000000 + 255) / 256, 256>>>(d_array1, d_array2, d_result, K * 1000000);
    auto end_time = std::chrono::high_resolution_clock::now(); // end timer
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Time elapsed: " << elapsed_time.count() << " ms" << std::endl;

    // copy result from device to host
    cudaMemcpy(result, d_result, K * 1000000 * sizeof(int), cudaMemcpyDeviceToHost);

    // free memory on device and host
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_result);
    delete[] array1;
    delete[] array2;
    delete[] result;

    return 0;
}