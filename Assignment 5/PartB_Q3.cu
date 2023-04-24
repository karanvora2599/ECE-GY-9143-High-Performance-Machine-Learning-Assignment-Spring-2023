#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

__global__ void add_arrays_kernel(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int K_values[] = {1, 5, 10, 50, 100};
    int scenarios[][2] = {
        {1, 1},
        {1, 256},
        {0, 256},
    };

    for (int K : K_values) {
        int N = K * 1000000;

        int *a, *b, *c;
        cudaMallocManaged((void **)&a, N * sizeof(int));
        cudaMallocManaged((void **)&b, N * sizeof(int));
        cudaMallocManaged((void **)&c, N * sizeof(int));

        for (int i = 0; i < N; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        for (int i = 0; i < 3; i++) {
            int blocks = scenarios[i][0] == 0 ? (N + scenarios[i][1] - 1) / scenarios[i][1] : scenarios[i][0];
            int threads = scenarios[i][1];

            auto start = high_resolution_clock::now();
            add_arrays_kernel<<<blocks, threads>>>(a, b, c, N);
            cudaDeviceSynchronize();
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start).count();

            cout << "Time taken for K = " << K << ", scenario " << i + 1 << ": " << duration << " microseconds" << endl;
        }

        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }

    return 0;
}