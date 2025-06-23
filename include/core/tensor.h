// include/ushionn/core/tensor.h
#pragma once

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

// Host 메모리 해제를 위한 커스텀 Deleter (new [] 사용 시)
struct HostDeleter
{
    void operator()(void* ptr) const
    {
        if (ptr)
        {
            delete[] static_cast<char*>(ptr);
        }
    }
};

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

    template <typename T>
    Tensor(std::vector<size_t> shape);

    ~Tensor() = default;  // 스마트 포인터가 메모리 관리

    Tensor operator+(const Tensor& other);

   private:
    void calculate_strides();

    void add(const Tensor&, Tensor&);

    cublasHandle_t cublas_handle_;

    std::unique_ptr<void, HostDeleter> cpu_data_ptr_;
    std::unique_ptr<void, CudaDeleter> gpu_data_ptr_;

    size_t total_bytes_;

    std::vector<size_t> shape_;
    size_t shape_size_;

    std::vector<size_t> strides_;

    DataLocation location_ = DataLocation::NONE;
    DataType type_;
};

}  // namespace ushionn