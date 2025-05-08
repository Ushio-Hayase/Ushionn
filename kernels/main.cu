#include "main.cuh"

__global__ void addImpl(int n, float* x, float* y)
{
    const int i = threadIdx.x;
    y[i] = x[i] + y[i];
}

void add(int n, float* x, float* y)
{
    addImpl<<<1, 1>>>(n, x, y);
}
