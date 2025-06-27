#include <iostream>

float dot(float* v1, float* v2, int len)
{
    float sum = 0;
    for (int i = 0; i < len; ++i)
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}

float step(float v)
{
    return v > 0 ? 1 : 0;
}

float forward(float* x, float* w, int len)
{
    float u = dot(x, w, len);
    return step(u);
}

void train(float* w)
