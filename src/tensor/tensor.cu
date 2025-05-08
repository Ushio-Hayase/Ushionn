#pragma once

#include <iostream>
#include <memory>
#include <vector>

namespace tesron
{

/**
 * #TODO 1024개의 스레드까지만 제대로 작동
 */
template <typename T>
__global__ void AddCUDA1D(const T* src, const T* target, T* out)
{
    const int id = threadIdx.x;
    out[id] = src[id] + target[id];
}

template <typename T>
__global__ void AddCUDA2D(const T** src, const T** target, T** out)
{
}

template <typename T>
__global__ void AddCUDA3D(const T** src, const T** target, T** out)
{
}

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
    Tensor() : useCuda_(false), dimSize_(0) {}

    /// @brief 주어진 차원 개수와 크기에 맞는 텐서 생성
    /// @param dims 텐서 크기 이니셜라이저 리스트
    Tensor(std::initializer_list<size_t> dims) : useCuda_(false), dimSize_(sizeof(dims)), dims_(dims) {}

    /// @brief 주어진 배열의 크기로 초기화
    /// @param arr 배열의 포인터
    /// @param size 배열의 크기
    Tensor(const T* arr, size_t size)
    {
        dataSize_ = size;
        memcpy(data_.get(), arr, size * sizeof(T));
    }

    virtual ~Tensor() = default;

    /// @brief 텐서을 CUDA로 이동
    void Cuda()
    {
        if (device_ == Device == CUDA) return;
        T* ptr = nullptr;
        cudaMalloc(&ptr, dataSize_);
        auto errCode = cudaMemcpy(ptr, data_, dataSize_, cudaMemcpyHostToDevice);

        if (errCode != cudaSuccess)
            std::cerr << "Error : failed to copy Tensor from host to device, Error Code : " << errCode << std::endl;
        data_.reset(ptr);
    }

    /// @brief 텐서를 CPU로 이동
    void Cpu()
    {
        if (device_ == Device::CPU) return;
        T* ptr = nullptr;
        ptr = new T[dataSize_];
        cudaMemcpy(ptr, data_, dataSize_, cudaMemcpyDeviceToHost);

        if (errCode != cudaSuccess)
            std::cerr << "Error : failed to copy Tensor from device to host, Error Code : " << errCode << std::endl;
        cudaFree(data_.get());

        data_.release();
        data_.reset(ptr);
    }

    /// @brief 현재 텐서가 위치한 디바이스 반환
    /// @return 디바이스 열거형
    Device getDevice() const { return device_; }

    /// @brief 행렬 덧셈 수행
    /// @param x 더할 텐서
    void Add(const Tensor& x)
    {
        if (device_ == Device::CUDA && x.getDevice() == Device::CUDA)
        {
            if (dimSize_ == 1)
                AddCUDA1D<T><<<(1, 1, 1), (dataSize_, 1, 1)>>>(data_.get(), x.Data().get(), data_.get());
            else if (dimSize_ == 2)
                AddCUDA2D<T> < < < (1, 1, 1)
        }
        else if (device_ == Device::CPU && x.getDevice() == Device::CPU)
        {
            if (dimSize_ == 1)
                for (int i = 0; i < dataSize_; ++i) data_.get()[i] += x.Data().get()[i];
        }
    }

    void Multiply();
    void Matmul();
    void Transpose();
    void Inverse();

    std::weak_ptr<T> Data() const;

   private:
    std::unique_ptr<T> data_;  // 메인 데이터 포인터
    size_t dataSize_;          // 데이터 크기

    std::vector<size_t> dims_;  // 차원 순서
    size_t dimSize_;            // 차원 개수

    Device device_;
};
}  // namespace tesron