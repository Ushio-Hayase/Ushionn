#include "core/tensor.h"

__global__ void multiply_kernel_float(const float* a, const float* b, float* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void multiply_kernel_double(const double* a, const double* b, double* result, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] * b[idx];
    }
}

namespace ushionn
{
void Tensor::multiply(const Tensor& b, Tensor& r)
{
    USHIONN_ASSERT(shape_ == b.shape_, "The dimension of the tensor calculating does not match");
    USHIONN_ASSERT(shape_ == r.shape_, "The dimension of the tensor calculating does not match");

    USHIONN_ASSERT(location_ == b.location_, "The location of the data exists must be the same.");
    USHIONN_ASSERT(location_ == r.location_, "The location of the data exists must be the same.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        multiply_kernel_float<<<grid_size, block_size>>>(static_cast<const float*>(gpu_data_ptr_.get()),
                                                         static_cast<const float*>(b.gpu_data_ptr_.get()),
                                                         static_cast<float*>(r.gpu_data_ptr_.get()), total_elements);
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;

        multiply_kernel_double<<<grid_size, block_size>>>(static_cast<const double*>(gpu_data_ptr_.get()),
                                                          static_cast<const double*>(b.gpu_data_ptr_.get()),
                                                          static_cast<double*>(r.gpu_data_ptr_.get()), total_elements);
    }
    else if (location_ == DataLocation::HOST)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        if (type_ == DataType::FLOAT32)
        {
            const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
            const float* b_data = static_cast<const float*>(b.cpu_data_ptr_.get());
            float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

            for (size_t i = 0; i < total_elements; ++i)
            {
                r_data[i] = a_data[i] * b_data[i];
            }
        }
        else if (type_ == DataType::FLOAT64)
        {
            const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
            const double* b_data = static_cast<const double*>(b.cpu_data_ptr_.get());
            double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

            for (size_t i = 0; i < total_elements; ++i)
            {
                r_data[i] = a_data[i] * b_data[i];
            }
        }
    }
}
}  // namespace ushionn
