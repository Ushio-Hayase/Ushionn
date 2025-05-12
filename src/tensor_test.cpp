#include "tensor.h"

#include <gtest/gtest.h>

#include <iostream>

TEST(TensorGeneraotrFunc, ArrayCopy)
{
    const int arr_size = 10000;
    int* default_arr = new int[10000];
    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    ushionn::Tensor test({arr_size}, default_arr, arr_size);
    ;

    for (size_t i = 0; i < arr_size; ++i)
    {
        int ptr = test.Index<int>({i});
        EXPECT_EQ(ptr, default_arr[i]);
    }
}

TEST(TensorAddFunc, AddUnder1024CPU)
{
    const int arr_size = 1024;
    int* default_arr = new int[arr_size];

    int multiple = 1;

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});
    test2.SetDims({arr_size / 4, 4});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1, 1}), arr2D[1][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorAddFunc, AddUnder1024CUDA)
{
    const int arr_size = 1024;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);

    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({11}), default_arr[11] * multiple);

    test1.SetDims({arr_size / 4, 4});
    test2.SetDims({arr_size / 4, 4});
    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1}), arr2D[11][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});
    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorAddFunc, AddOver1024CPU)
{
    const int arr_size = 32'768;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});
    test2.SetDims({arr_size / 4, 4});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1, 1}), arr2D[1][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});

    multiple += 1;
    test1.Add(test2);

    EXPECT_EQ(test1.Index<int>({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorAddFunc, AddOver1024CUDA)
{
    const int arr_size = 32'768;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);

    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});
    test2.SetDims({arr_size / 4, 4});
    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({5, 1}), arr2D[5][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});
    test1.CUDA();
    test2.CUDA();

    multiple += 1;
    test1.Add(test2);

    test1.CPU();
    test2.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorAddFunc, AddTimeTestCPU)
{
    const int arr_size = 131'072;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);
    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});

    for (int i = 0; i < 64; ++i) test1.Add(test2);
}

TEST(TensorAddFunc, AddTimeTestCUDA)
{
    const int arr_size = 131'072;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);
    ushionn::Tensor test2({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size / 8 / 8, 8, 8});
    test2.SetDims({arr_size / 8 / 8, 8, 8});

    test1.CUDA();
    test2.CUDA();

    for (int i = 0; i < 64; ++i) test1.Add(test2);
}

TEST(TensorMulFunc, MulUnder1024CPU)
{
    const int arr_size = 1024;
    int* default_arr = new int[arr_size];

    int multiple = 1;

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1, 1}), arr2D[1][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorMulFunc, MulUnder1024CUDA)
{
    const int arr_size = 1024;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({11}), default_arr[11] * multiple);

    test1.SetDims({arr_size / 4, 4});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1}), arr2D[11][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorMulFunc, MulOver1024CPU)
{
    const int arr_size = 32'768;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1, 1}), arr2D[1][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    multiple *= 2;
    test1.Mul(2);

    EXPECT_EQ(test1.Index<int>({1, 1, 1}), arr3D[1][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}

TEST(TensorMulFunc, MulOver1024CUDA)
{
    const int arr_size = 32'768;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int** arr2D = new int*[arr_size / 4];
    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;

    for (int i = 0; i < arr_size / 4; ++i) arr2D[i] = default_arr + i * 4;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({1}), default_arr[1] * multiple);

    test1.SetDims({arr_size / 4, 4});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({5, 1}), arr2D[5][1] * multiple);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    test1.CUDA();

    multiple *= 2;
    test1.Mul(2);

    test1.CPU();

    EXPECT_EQ(test1.Index<int>({11, 1, 1}), arr3D[11][1][1] * multiple);

    delete[] arr2D;
    for (int i = 0; i < arr_size / 8; ++i) delete[] arr3D[i];
    delete[] arr3D;
    delete[] default_arr;
}
TEST(TensorMulFunc, MulTimeTestCPU)
{
    const int arr_size = 131'072;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    for (int i = 0; i < 64; ++i) test1.Mul(1);
}

TEST(TensorMulFunc, MulTimeTestCUDA)
{
    const int arr_size = 131'072;

    int multiple = 1;

    int* default_arr = new int[arr_size];

    int*** arr3D = new int**[arr_size / 8];

    for (int i = 0; i < arr_size; ++i) default_arr[i] = i;
    for (int i = 0; i < arr_size / 8; ++i)
    {
        arr3D[i] = new int*[arr_size / 8];
        for (int j = 0; j < arr_size / 8 / 8; ++j) arr3D[i][j] = default_arr + i * 8 * 8 + j * 8;
    }

    ushionn::Tensor test1({arr_size}, default_arr, arr_size);

    test1.SetDims({arr_size / 8 / 8, 8, 8});

    test1.CUDA();

    for (int i = 0; i < 64; ++i) test1.Mul(1);
}