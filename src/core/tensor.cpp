#include "core/tensor.h"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemset etc.

#include <atomic>    // for generate_unique_id_internal
#include <iostream>  // for print_meta_info

namespace ushionn
{

// --- 생성자 구현 ---
Tensor::Tensor(std::vector<size_t> shape, DataType type) : cpu_data_ptr_(nullptr, type)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();
    std::copy(shape.begin(), shape.end(), shape_);

    shape_size_ = shape_.size();
    type_ = type;
    if (type_ == DataType::FLOAT32)
        total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(float);
    else if (type_ == DataType::FLOAT64)
        total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(double);
}

template <typename T>
Tensor::Tensor(std::vector<size_t> shape, const T* ptr)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();

    std::copy(shape.begin(), shape.end(), shape_);

    shape_size_ = shape_.size();
    total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(T);

    cpu_data_ptr_.reset(new T[total_bytes_ / sizeof(T)]);

    memcpy(cpu_data_ptr_.get(), ptr, total_bytes_);

    type_ = utils::primitiveTypeToDataType<T>();
    location_ = DataLocation::HOST;
}

Tensor::Tensor(std::vector<size_t> shape, void* gpu_ptr, DataType type) : cpu_data_ptr_(nullptr, type)
{
    cublasCreate(&cublas_handle_);
    cudaThreadSynchronize();

    std::copy(shape.begin(), shape.end(), shape_);

    shape_size_ = shape_.size();
    type_ = type;

    if (type == DataType::FLOAT32)
        total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(float);
    else if (type == DataType::FLOAT64)
        total_bytes_ = std::accumulate(shape.begin(), shape.end(), 0) * sizeof(double);

    location_ = DataLocation::DEVICE;

    gpu_data_ptr_.reset(gpu_ptr);
}

Tensor::Tensor(Tensor&& other)
    : cpu_data_ptr_(std::move(other.cpu_data_ptr_)), gpu_data_ptr_(std::move(other.gpu_data_ptr_))
{
    cublas_handle_ = other.cublas_handle_;
    total_bytes_ = other.total_bytes_;
    shape_ = std::move(other.shape_);
    shape_size_ = other.shape_size_;
    strides_ = std::move(other.strides_);
    location_ = other.location_;
    type_ = other.type_;
}

Tensor Tensor::operator+(const Tensor& other)
{
    USHIONN_ASSERT(location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");
    USHIONN_ASSERT(other.location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);
    result.allocate_cpu_mem(total_bytes_);
    add(other, result);
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    USHIONN_ASSERT(location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");
    USHIONN_ASSERT(other.location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");

    add(other, *this);

    return *this;
}

Tensor Tensor::operator*(const Tensor& other)
{
    USHIONN_ASSERT(location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");
    USHIONN_ASSERT(other.location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);
    result.allocate_cpu_mem(total_bytes_);

    multiply(other, result);

    return result;
}

template <typename T>
Tensor Tensor::operator*(const T& scalar)
{
    static_assert(std::is_arithmetic_v<T>, "스칼라는 숫자 타입이여야 합니다.")

        USHIONN_ASSERT(location_ != DataLocation::NONE, "텐서가 할당되지 않았습니다.");

    Tensor result(this->shape_, type_);
    result.allocate_cpu_mem(total_bytes_);

    multiply(scalar, result);

    return result;
}

template <typename T>
Tensor operator*(const T& scalar, const Tensor& tensor)
{
    return tensor * scalar;
}

Tensor Tensor::operator=(Tensor&& other)
{
    Tensor tensor(std::move(other));
    return tensor;
}

void Tensor::allocate_gpu_mem(size_t total_bytes)
{
    total_bytes_ = total_bytes;

    USHIONN_WARN(gpu_data_ptr_, "Gpu memory is already allocated. Ignore the command");

    if (!gpu_data_ptr_)
    {
        if (type_ == DataType::FLOAT32)
        {
            void* tmp_ptr;
            cudaMalloc(&tmp_ptr, total_bytes_);
            gpu_data_ptr_.reset(tmp_ptr);
        }
        else if (type_ == DataType::FLOAT64)
        {
            void* tmp_ptr;
            cudaMalloc(&tmp_ptr, total_bytes_);
            gpu_data_ptr_.reset(tmp_ptr);
        }
    }
}

void Tensor::allocate_cpu_mem(size_t total_bytes)
{
    total_bytes_ = total_bytes;

    USHIONN_WARN(cpu_data_ptr_, "Cpu memory is already allocated. Ignore the command");

    if (!cpu_data_ptr_)
    {
        if (type_ == DataType::FLOAT32)
            cpu_data_ptr_.reset(new float[total_bytes_ / sizeof(float)]);
        else if (type_ == DataType::FLOAT64)
            cpu_data_ptr_.reset(new double[total_bytes_ / sizeof(double)]);
    }
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

        size_t row = 1;
        if (shape_size_ >= 2)
            for (size_t i = 0; i < shape_size_ - 1; ++i) row *= shape_[i];
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

        size_t row = 1;
        if (shape_size_ >= 2)
            for (size_t i = 0; i < shape_size_ - 1; ++i) row *= shape_[i];
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
    else if (location_ == DataLocation::HOST)
    {
        add_cpu(b, r);
    }
}

template <typename T>
void Tensor::multiply(const T& b, Tensor& r)
{
    USHIONN_ASSERT(shape_ == r.shape_, "The dimension of the tensor calculating does not match");
    USHIONN_ASSERT(location_ == r.location_, "The location of the data exists must be the same.");

    if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT32)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }

        USHIONN_WARN(typeid(T) == typeid(float), "텐서와 주어진 스칼라의 타입이 일치하지 않아 캐스팅을 진행합니다.");

        auto state = cublasSscal(r.cublas_handle_, total_elements, &b, static_cast<float*>(gpu_data_ptr_.get()), 1);

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : " << state << std::endl;
    }
    else if (location_ == DataLocation::DEVICE && type_ == DataType::FLOAT64)
    {
        size_t total_elements = 1;
        for (size_t dim : shape_)
        {
            total_elements *= dim;
        }
        auto state = cublasDscal(r.cublas_handle_, total_elements, &b, static_cast<double*>(gpu_data_ptr_.get()), 1);

        USHIONN(typeid(T) == typeid(double), "텐서와 주어진 스칼라의 타입이 일치하지 않아 캐스팅을 진행합니다.");

        if (state != CUBLAS_STATUS_SUCCESS)
            std::cerr << "There was a problem adding tensor, Error state : " << state << std::endl;
    }
    else if (location_ == DataLocation::HOST)
    {
        schalar_multiply_cpu(b, r);
    }
}

void Tensor::add_cpu(const Tensor& b, Tensor& r)
{
    // 전체 원소 개수 계산
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // 타입별로 계산 수행
    if (type_ == DataType::FLOAT32)
    {
        const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
        const float* b_data = static_cast<const float*>(b.cpu_data_ptr_.get());
        float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] + b_data[i];
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
        const double* b_data = static_cast<const double*>(b.cpu_data_ptr_.get());
        double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] + b_data[i];
        }
    }
}

template <typename T>
void Tensor::schalar_multiply_cpu(const T& b, Tensor& r)
{
    // 전체 원소수 계산
    size_t total_elements = 1;
    for (size_t dim : shape_)
    {
        total_elements *= dim;
    }

    // 타입별로 계산 수행
    if (type_ == DataType::FLOAT32)
    {
        USHIONN_WARN(typeid(T) == typeid(float), "텐서와 주어진 스칼라의 타입이 일치하지 않아 캐스팅을 진행합니다.");

        const float* a_data = static_cast<const float*>(cpu_data_ptr_.get());
        float* r_data = static_cast<float*>(r.cpu_data_ptr_.get());

        const float alpha = static_cast<float>(b);

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] * alpha;
        }
    }
    else if (type_ == DataType::FLOAT64)
    {
        USHIONN(typeid(T) == typeid(double), "텐서와 주어진 스칼라의 타입이 일치하지 않아 캐스팅을 진행합니다.");

        const double* a_data = static_cast<const double*>(cpu_data_ptr_.get());
        double* r_data = static_cast<double*>(r.cpu_data_ptr_.get());

        const double alpha = static_cast<double>(b);

        for (size_t i = 0; i < total_elements; ++i)
        {
            r_data[i] = a_data[i] * alpha;
        }
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