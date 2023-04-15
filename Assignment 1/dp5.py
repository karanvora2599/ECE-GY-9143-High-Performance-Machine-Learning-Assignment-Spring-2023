import time
import sys
import numpy as np

def dp(A,B):
    R = np.dot(A, B)
    return R

ArraySize = int(sys.argv[1])
Repeatation = int(sys.argv[2])

A = np.ones(ArraySize,dtype=np.float32)
B = np.ones(ArraySize,dtype=np.float32)
TimeMeasurement = []
AverageTime = 0.0

for loop in range(0, Repeatation):
    start = time.monotonic()
    value = dp(A, B)
    end = time.monotonic()
    TimeMeasurement.append(end - start)

for loop in range(len(TimeMeasurement)//2, len(TimeMeasurement)):
    AverageTime += TimeMeasurement[loop]

TotalMemory = ( sys.getsizeof(A) + sys.getsizeof(B) ) / 1000000000
FLOP = ( ArraySize * 2 ) / 1000000000
if(Repeatation == 1):
    AverageTime = AverageTime
else:
    AverageTime = AverageTime/(Repeatation//2)
print("N: {}, Average Time: {} second, Bandwidth: {} GB/sec, FLOPS: {} GFLOP/sec".format(ArraySize, AverageTime, TotalMemory/AverageTime, FLOP/AverageTime))