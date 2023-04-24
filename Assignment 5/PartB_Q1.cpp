#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

void add_arrays(int *a, int *b, int *c, int K) {
    for (int i = 0; i < K * 1000000; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int K_values[] = {1, 5, 10, 50, 100};

    for (int K : K_values) {
        int *a = (int *)malloc(K * 1000000 * sizeof(int));
        int *b = (int *)malloc(K * 1000000 * sizeof(int));
        int *c = (int *)malloc(K * 1000000 * sizeof(int));

        for (int i = 0; i < K * 1000000; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        auto start = high_resolution_clock::now();
        add_arrays(a, b, c, K);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start).count();

        cout << "Time taken for K = " << K << ": " << duration << " microseconds" << endl;

        free(a);
        free(b);
        free(c);
    }

    return 0;
}