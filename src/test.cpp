#include <gtest/gtest.h>

#include <iostream>

#include "tensor.cuh"

TEST(TensorGeneraotrFunc, ArrayCopy)
{
    const int arrSize = 10000;
    int* defaultArr = new int[10000];
    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    ushionn::Tensor test(defaultArr, arrSize);
    test.setDims({arrSize});

    for (size_t i = 0; i < arrSize; ++i)
    {
        int ptr = test.Index({i});
        EXPECT_EQ(ptr, defaultArr[i]);
    }
}

TEST(TensorAddFunc, AddUnder1024CPU)
{
    const int arrSize = 1024;
    int* defaultArr = new int[arrSize];

    int multiple = 1;

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);

    test1.setDims({arrSize});
    test2.setDims({arrSize});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});
    test2.setDims({arrSize / 4, 4});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1, 1}), arr2D[1][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorAddFunc, AddUnder1024CUDA)
{
    const int arrSize = 1024;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);

    test1.setDims({arrSize});
    test2.setDims({arrSize});

    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({11}), defaultArr[11] * multiple);

    test1.setDims({arrSize / 4, 4});
    test2.setDims({arrSize / 4, 4});
    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({11, 1}), arr2D[11][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});
    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorAddFunc, AddOver1024CPU)
{
    const int arrSize = 32'768;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);

    test1.setDims({arrSize});
    test2.setDims({arrSize});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});
    test2.setDims({arrSize / 4, 4});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1, 1}), arr2D[1][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorAddFunc, AddOver1024CUDA)
{
    const int arrSize = 32'768;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);

    test1.setDims({arrSize});
    test2.setDims({arrSize});

    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});
    test2.setDims({arrSize / 4, 4});
    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({5, 1}), arr2D[5][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});
    test1.Cuda();
    test2.Cuda();

    multiple += 1;
    test1.Add(test2);

    test1.Cpu();
    test2.Cpu();

    EXPECT_EQ(test1.Index({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorAddFunc, AddTimeTestCPU)
{
    const int arrSize = 131'072;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);
    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});

    for (int i = 0; i < 64; ++i) test1.Add(test2);
}

TEST(TensorAddFunc, AddTimeTestCUDA)
{
    const int arrSize = 131'072;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);
    ushionn::Tensor<int> test2(defaultArr, arrSize);

    test1.setDims({arrSize / 8 / 8, 8, 8});
    test2.setDims({arrSize / 8 / 8, 8, 8});

    test1.Cuda();
    test2.Cuda();

    for (int i = 0; i < 64; ++i) test1.Add(test2);
}

TEST(TensorMultiplyFunc, MultiplyUnder1024CPU)
{
    const int arrSize = 1024;
    int* defaultArr = new int[arrSize];

    int multiple = 1;

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1, 1}), arr2D[1][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorMultiplyFunc, MultiplyUnder1024CUDA)
{
    const int arrSize = 1024;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({11}), defaultArr[11] * multiple);

    test1.setDims({arrSize / 4, 4});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({11, 1}), arr2D[11][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorMultiplyFunc, MultiplyOver1024CPU)
{
    const int arrSize = 32'768;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1, 1}), arr2D[1][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    multiple *= 2;
    test1.Multiply(2);

    EXPECT_EQ(test1.Index({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}

TEST(TensorMultiplyFunc, MultiplyOver1024CUDA)
{
    const int arrSize = 32'768;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int** arr2D = new int*[arrSize / 4];
    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;

    for (int i = 0; i < arrSize / 4; ++i) arr2D[i] = defaultArr + i * 4;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({1}), defaultArr[1] * multiple);

    test1.setDims({arrSize / 4, 4});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({5, 1}), arr2D[5][1] * multiple);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    test1.Cuda();

    multiple *= 2;
    test1.Multiply(2);

    test1.Cpu();

    EXPECT_EQ(test1.Index({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arrSize / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] defaultArr;
}
TEST(TensorMultiplyFunc, MultiplyTimeTestCPU)
{
    const int arrSize = 131'072;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    for (int i = 0; i < 64; ++i) test1.Multiply(1);
}

TEST(TensorMultiplyFunc, MultiplyTimeTestCUDA)
{
    const int arrSize = 131'072;

    int multiple = 1;

    int* defaultArr = new int[arrSize];

    int*** arr3D = new int**[arrSize / 8];

    for (int i = 0; i < arrSize; ++i) defaultArr[i] = i;
    for (int i = 0; i < arrSize / 8; ++i)
    {
        arr3D[i] = new int*[arrSize / 8];
        for (int j = 0; j < arrSize / 8 / 8; ++j) arr3D[i][j] = defaultArr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor<int> test1(defaultArr, arrSize);

    test1.setDims({arrSize / 8 / 8, 8, 8});

    test1.Cuda();

    for (int i = 0; i < 64; ++i) test1.Multiply(1);
}