#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void add_arrays_kernel(double *array1, double *array2, double *output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = array1[idx] + array2[idx];
    }
}

int main() {
    int K_values[] = {1, 5, 10, 50, 100};

    for (int K : K_values) {
        int num_elements = K * 1000000;

        double *array1_host, *array2_host, *output_host;
        cudaMallocManaged(&array1_host, num_elements * sizeof(double));
        cudaMallocManaged(&array2_host, num_elements * sizeof(double));
        cudaMallocManaged(&output_host, num_elements * sizeof(double));

        for (int i = 0; i < num_elements; ++i) {
            array1_host[i] = i * 1.0;
            array2_host[i] = i * 2.0;
        }

        int scenarios[][2] = {
            {1, 1},
            {1, 256},
            {(num_elements + 255) / 256, 256}
        };

        for (int i = 0; i < 3; ++i) {
            int num_blocks = scenarios[i][0];
            int threads_per_block = scenarios[i][1];

            auto start = std::chrono::high_resolution_clock::now();
            add_arrays_kernel<<<num_blocks, threads_per_block>>>(array1_host, array2_host, output_host, num_elements);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            std::cout << "Scenario " << i + 1 << " with K = " << K << " million elements: "
                      << elapsed.count() << " seconds" << std::endl;
        }

        cudaFree(array1_host);
        cudaFree(array2_host);
        cudaFree(output_host);
    }

    return 0;
}