#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Set the value of K (in millions)
    int K = 1; // Change this value to 1, 5, 10, 50, 100 to profile the program

    // Calculate the number of elements in each array
    int num_elements = K * 1000000;

    // Allocate memory for two input arrays and the output array
    double *array1 = (double *) malloc(num_elements * sizeof(double));
    double *array2 = (double *) malloc(num_elements * sizeof(double));
    double *output = (double *) malloc(num_elements * sizeof(double));

    // Initialize the input arrays with some values
    for (int i = 0; i < num_elements; ++i) {
        array1[i] = i * 1.0;
        array2[i] = i * 2.0;
    }

    // Measure the time taken to add the elements of two arrays
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_elements; ++i) {
        output[i] = array1[i] + array2[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the time taken for the addition operation
    std::cout << "Time taken for addition with K = " << K << " million elements: " << elapsed.count() << " seconds" << std::endl;

    // Free the memory allocated for the arrays
    free(array1);
    free(array2);
    free(output);

    return 0;
}