
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int n = N * gridDim.x * blockDim.x; // array length
    int stride = gridDim.x * blockDim.x; // number of threads
    int threadStartIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int i;

    for( i=threadStartIndex; i<n; i+=stride ){
        C[i] = A[i] + B[i];
    }
}
