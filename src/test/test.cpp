#include <gtest/gtest.h>

#include "tensor.h"

TEST(TensorGeneraotrFunc, ArrayCopy)
{
    const int arrSize = 10000;
    int* defaultArr = new int[10000];
    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    tesron::Tensor<int> test(defaultArr, arrSize);
    test.setDims({arrSize});

    for (size_t i = 0; i < arrSize; ++i)
    {
        int ptr = test.Index({i});
        EXPECT_EQ(ptr, defaultArr[i]);
    }
}