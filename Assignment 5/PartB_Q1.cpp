#include <iostream>
#include <chrono>

int main() {
    const int K = 100; // change K to test different values

    // allocate memory for arrays
    int* array1 = new int[K * 1000000];
    int* array2 = new int[K * 1000000];
    int* result = new int[K * 1000000];

    // initialize arrays
    for (int i = 0; i < K * 1000000; i++) {
        array1[i] = i;
        array2[i] = i * 2;
    }

    // add arrays and store result in third array
    auto start_time = std::chrono::high_resolution_clock::now(); // start timer
    for (int i = 0; i < K * 1000000; i++) {
        result[i] = array1[i] + array2[i];
    }
    auto end_time = std::chrono::high_resolution_clock::now(); // end timer
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Time elapsed: " << elapsed_time.count() << " ms" << std::endl;

    // free memory
    delete[] array1;
    delete[] array2;
    delete[] result;

    return 0;
}