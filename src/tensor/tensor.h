#pragma once

#include <memory>
#include <vector>

namespace tesron
{
enum class Device
{
    CPU,
    CUDA
};

template <typename T>
class Tensor
{
   public:
    /// @brief CPU에 배치된 아무것도 존재하지 않는 빈 텐서를 생성합니다.
    tesron::Tensor<T>::Tensor() : device_(tesron::Device::CPU), dimSize_(0) {}

    /// @brief 주어진 차원 개수와 크기에 맞는 텐서 생성
    /// @param dims 텐서 크기 이니셜라이저 리스트
    Tensor(std::initializer_list<size_t> dims);

    /// @brief 주어진 배열의 크기로 초기화
    /// @param arr 배열의 포인터
    /// @param size 배열의 크기
    Tensor(const T* arr, size_t size);

    void Cuda();
    void Cpu();

    Device getDevice() const;

    Tensor& Add();
    Tensor& Multiply();
    Tensor& Matmul();
    Tensor& Transpose();
    Tensor& Inverse();

    T Index(std::initializer_list<size_t>);

    bool setDims(std::initializer_list<size_t>);

   private:
    std::shared_ptr<T> data_;  // 메인 데이터 포인터
    size_t dataSize_;          // 데이터 크기

    std::vector<size_t> dims_;  // 차원 순서
    size_t dimSize_;            // 차원 개수

    Device device_;
};
}  // namespace tesron

#include "tensor.cu"