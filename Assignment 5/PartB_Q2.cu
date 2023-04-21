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

        double *array1_host = (double *) malloc(num_elements * sizeof(double));
        double *array2_host = (double *) malloc(num_elements * sizeof(double));
        double *output_host = (double *) malloc(num_elements * sizeof(double));

        for (int i = 0; i < num_elements; ++i) {
            array1_host[i] = i * 1.0;
            array2_host[i] = i * 2.0;
        }

        double *array1_device, *array2_device, *output_device;
        cudaMalloc(&array1_device, num_elements * sizeof(double));
        cudaMalloc(&array2_device, num_elements * sizeof(double));
        cudaMalloc(&output_device, num_elements * sizeof(double));

        cudaMemcpy(array1_device, array1_host, num_elements * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(array2_device, array2_host, num_elements * sizeof(double), cudaMemcpyHostToDevice);

        int scenarios[][2] = {
            {1, 1},
            {1, 256},
            {(num_elements + 255) / 256, 256}
        };

        for (int i = 0; i < 3; ++i) {
            int num_blocks = scenarios[i][0];
            int threads_per_block = scenarios[i][1];

            auto start = std::chrono::high_resolution_clock::now();
            add_arrays_kernel<<<num_blocks, threads_per_block>>>(array1_device, array2_device, output_device, num_elements);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            cudaMemcpy(output_host, output_device, num_elements * sizeof(double), cudaMemcpyDeviceToHost);

            std::cout << "Scenario " << i + 1 << " with K = " << K << " million elements: "
                      << elapsed.count() << " seconds" << std::endl;
        }

        cudaFree(array1_device);
        cudaFree(array2_device);
        cudaFree(output_device);
        free(array1_host);
        free(array2_host);
        free(output_host);
    }

    return 0;
}