#include "core/tensor.h"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemset etc.

#include <atomic>    // for generate_unique_id_internal
#include <iostream>  // for print_meta_info

namespace ushionn
{

// --- 생성자 구현 ---
template <typename T>
Tensor::Tensor(std::vector<size_t> shape)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();
    std::copy(shape.begin(), shape.end(), shape_);

    shape_size_ = shape_.size();

    size_t total_byte = std::accumulate(shape_.begin(), shape_.end(), 0) * sizeof(T);

    total_bytes_ = total_byte;

    cpu_data_ptr_ = new char[total_bytes_];

    location_ = DataLocation::HOST;

    type_ = ushionn::utils::primitiveTypeToDataType<T>();
}

void Tensor::add(const Tensor& b, Tensor& r)
{
    USHIONN_ASSERT(shape_ == b.shape_, "The dimension of the tensor calculating does not match");
    USHIONN_ASSERT(shape_ == r.shape_, "The dimension of the tensor calculating does not match");

    USHIONN_ASSERT(location_ == b.location_, "The location of the data exists must be the same.");
    USHIONN_ASSERT(location_ == r.location_, "The location of the data exists must be the same.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        float alpha = 1.f;
        float beta = 1.f;

        size_t row;
        if (shape_size_ >= 2)
            row = shape_[shape_size_ - 2];
        else
            row = 1;
        size_t col = shape_[shape_size_ - 1];
        auto state = cublasSgeam(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, row, col, &alpha,
                                 static_cast<const float*>(gpu_data_ptr_.get()), row, &beta,
                                 static_cast<const float*>(b.gpu_data_ptr_.get()), row,
                                 static_cast<float*>(r.gpu_data_ptr_.get()), row);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : " << state << std::endl;
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        double alpha = 1.f;
        double beta = 1.f;

        size_t row;
        if (shape_size_ >= 2)
            row = shape_[shape_size_ - 2];
        else
            row = 1;
        size_t col = shape_[shape_size_ - 1];
        auto state = cublasDgeam(r.cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, row, col, &alpha,
                                 static_cast<const double*>(gpu_data_ptr_.get()), row, &beta,
                                 static_cast<const double*>(b.gpu_data_ptr_.get()), row,
                                 static_cast<double*>(r.gpu_data_ptr_.get()), row);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : " << state << std::endl;
    }
}

Tensor Tensor::operator+(const Tensor& other)
{
    if (type_ == DataType::FLOAT32)
    {
        Tensor<float> result(this->shape_);
    }
}

void Tensor::calculate_strides()
{
    strides_.resize(shape_.size());

    strides_[0] = 1;

    size_t stride = 1;

    for (int i = 1; i < shape_size_; ++i)
    {
        stride *= shape_[i];
        strides_[i] = stride;
    }
}

}  // namespace ushionn