#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mkl_cblas.h>
struct timespec start, end;

float bdp(long N, float *pA, float *pB)
{
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

int main(int argc, char *argv[])
{
    int Repeatation = atoi(argv[2]), i, ArraySize = atoi(argv[1]);
    float *pA = malloc(ArraySize * sizeof(float)), *pB = malloc(ArraySize * sizeof(float));
    double TimeMeasurment[Repeatation], AverageTime = 0.0, TotalMemory, FLOP;
    
    //Initializing the Array
    for(i=0; i<ArraySize; i++)
    {
        pA[i] = 1.0;
        pB[i] = 1.0;
    }
    
    //Measuring the performance
    for(i=0; i<Repeatation; i++)
    {
        clock_gettime(CLOCK_MONOTONIC,&start);
        bdp(ArraySize, pA, pB);
        clock_gettime(CLOCK_MONOTONIC,&end);
        TimeMeasurment[i] = ((double)end.tv_nsec - (double)start.tv_nsec);
    }

    for(i = Repeatation; i>Repeatation%2; i--)
    {
        AverageTime += TimeMeasurment[i];
    }

    AverageTime = AverageTime/(Repeatation/2);
    TotalMemory = ((double)ArraySize * (double)sizeof(float) * 2) / 1000000000;
    FLOP = ((double)ArraySize * 2.0) / 1000000000.0;
    printf("N: %d, AverageTime: %.20f second, Bandwitdh: %.20f GB/sec, FLOPS: %.20f GFLOP/sec\n", ArraySize, AverageTime/1000000000.0, TotalMemory/AverageTime, FLOP/AverageTime);

    return 0;
}