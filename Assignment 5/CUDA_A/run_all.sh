#!/bin/bash

echo "PART A "

echo "running ./vecadd00 500"
./vecadd00 500
echo "running ./vecadd00 1000"
./vecadd00 1000
echo "running ./vecadd00 2000"
./vecadd00 2000

echo "running ./vecadd01 500"
./vecadd01 500
echo "running ./vecadd01 1000"
./vecadd01 1000
echo "running ./vecadd01 2000"
./vecadd01 2000

echo "running ./matmult00 16"
./matmult00 16
echo "running ./matmult00 32"
./matmult00 32
echo "running ./matmult00 64"
./matmult00 64

echo "running ./matmult01 8"
./matmult01 8
echo "running ./matmult01 16"
./matmult01 16
echo "running ./matmult01 32"
./matmult01 32