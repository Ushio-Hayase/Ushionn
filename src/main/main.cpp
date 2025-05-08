#include <math.h>

#include <iostream>
#include <main.cuh>

int main()
{
    int N = 1 << 20;  // 1M

    float* x = new float[N];
    float* y = new float[N];

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(N, x, y);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    return 0;
}