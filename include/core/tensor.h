// include/ushionn/core/tensor.h
#pragma once

#include <functional>
#include <memory>  // for std::unique_ptr
#include <numeric>
#include <stdexcept>
#include <vector>

#include "core/common.h"      // USHIONN_ASSERT 등
#include "cuda/cuda_utils.h"  // CUDA_CHECK 등 (common.h를 통해 포함됨)

namespace ushionn
{
// 정적 멤버 초기화 (generate_unique_id_internal용)
static std::atomic<int64_t> tensor_uid_counter = 1000;
// CUDA 메모리 해제를 위한 커스텀 Deleter

class Tensor
{
   public:
    // 데이터 위치 상태를 나타내는 열거형
    enum class DataLocation
    {
        NONE,    // 데이터 없음 (메모리 할당 전)
        HOST,    // CPU 메모리에만 유효한 데이터 존재
        DEVICE,  // GPU 메모리에만 유효한 데이터 존재
    };

    // --- 생성자 및 소멸자 ---
    Tensor() = default;

    Tensor(std::vector<size_t> shape, DataType type);

    /// @brief HOST에 데이터를 채우며 텐서를 생성합니다.
    /// @tparam T 자료형
    /// @param shape 차원
    /// @param ptr 복사할 데이터의 포인터
    template <typename T>
    Tensor(std::vector<size_t> shape, const T* ptr);

    /// @brief DEVICE에 데이터를 참조하며 텐서를 생성합니다.
    /// @param shape 차원
    /// @param gpu_ptr 참조할 DEVICE 포인터
    /// @param type 자료형
    Tensor(std::vector<size_t> shape, void* gpu_ptr, DataType type);

    Tensor(const Tensor& other) = delete;

    Tensor(Tensor&& other);

    ~Tensor() = default;  // 스마트 포인터가 메모리 관리

    // 텐서 덧셈
    Tensor operator+(const Tensor& other);

    Tensor& operator+=(const Tensor& other);

    // 원소별 텐서 간 곱셈
    Tensor operator*(const Tensor& other);

    // 스칼라 배
    template <typename T>
    Tensor operator*(const T& scalar);

    template <typename T>
    friend Tensor operator*(const T& scalar, const Tensor& tensor);

    Tensor& operator*=(const Tensor& other);

    template <typename T>
    Tensor& operator*=(const T& scalar);

    Tensor operator=(const Tensor& other) = delete;

    Tensor operator=(Tensor&& other);

    /// @brief 두 텐서를 더합니다.
    /// @param b 더할 텐서
    /// @param r 결과 텐서
    void add(const Tensor& b, Tensor& r);

    /// @brief 두 텐서의 각 원소끼리 곱합니다.
    /// @param b 곱할 텐서
    /// @param r 결과 텐서
    void multiply(const Tensor& b, Tensor& r);

    template <typename T>
    void multiply(const T& b, Tensor& r);

    void allocate_gpu_mem(size_t total_bytes);
    void allocate_cpu_mem(size_t total_bytes);

   private:
    struct CudaDeleter
    {
        void operator()(void* ptr) const
        {
            if (ptr)
            {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    };

    /// @brief 메모리 해제를 위한 커스텀 Deleter,
    struct HostDeleter
    {
        std::function<void(void*)> deleter_func;

        HostDeleter(DataType type)
        {
            switch (type)
            {
                case DataType::FLOAT32:
                    deleter_func = [](void* ptr) { delete[] static_cast<float*>(ptr); };
                    break;
                case DataType::FLOAT64:
                    deleter_func = [](void* ptr) { delete[] static_cast<double*>(ptr); };
                    break;
            }
        }

        void operator()(void* ptr) const
        {
            if (ptr && deleter_func)
            {
                deleter_func(ptr);
            }
        }
    };

    void calculate_strides();

    void add_cpu(const Tensor& b, Tensor& r);

    template <typename T>
    void schalar_multiply_cpu(const T& b, Tensor& r);

    cublasHandle_t cublas_handle_;

    std::unique_ptr<void, HostDeleter> cpu_data_ptr_;
    std::unique_ptr<void, CudaDeleter> gpu_data_ptr_ = nullptr;

    size_t total_bytes_;

    std::vector<size_t> shape_;
    size_t shape_size_;

    std::vector<size_t> strides_;

    DataLocation location_ = DataLocation::NONE;
    DataType type_;
};

}  // namespace ushionn