#pragma once

#include <memory>
#include <vector>

enum class Device
{
    CPU,
    CUDA
};

template <typename T>
class Tensor
{
   public:
    Tensor();
    template <typename... Args>
    Tensor(Args...)
    {
    }

    Tensor(std::initializer_list<T>);

    void Cuda();
    void Cpu();

    Device getDevice() const;

    Tensor& Add();
    Tensor& Multiply();
    Tensor& Matmul();
    Tensor& Transpose();
    Tensor& Inverse();

    std::weak_ptr<T> Data() const;

   private:
    std::unique_ptr<T> data_;  // 메인 데이터 포인터

    vector<size_t> dims_;  // 차원 순서
    size_t dimSize_;       // 차원 개수

    bool useCuda_;
};
