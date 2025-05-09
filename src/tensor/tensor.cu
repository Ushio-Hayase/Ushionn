#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "constant.cu"

template <typename T, typename S>
__global__ void MultiplyCUDA1D(const T* src, const S target, T* out)
{
    const int tid = Grid1DTID(blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
    out[tid] = src[tid] * target;
}

template <typename T, typename S>
__global__ void MultiplyCUDA2D(const T* src, const S target, T* out, const size_t row, const size_t col)
{
    const int tid = Grid2DTID(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
    const int trow = BDim_Y * blockIdx.y + threadIdx.y;
    const int tcol = BDim_X * blockIdx.x + threadIdx.x;
    if (trow >= row || tcol >= col) return;
    out[tid] = src[tid] * target;
}

template <typename T, typename S>
__global__ void MultiplyCUDA3D(const T* src, const S target, T* out, const size_t dimX, const size_t dimY,
                               const size_t dimZ)
{
    const int tid = Grid3DTID(blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimZ = BDim_Z * blockIdx.z + threadIdx.x;
    const size_t tDimY = BDim_Y * blockIdx.y + threadIdx.y;
    const size_t tDimX = BDim_X * blockIdx.x + threadIdx.x;

    if (tDimX >= dimX || tDimY >= dimY || tDimZ >= dimZ) return;
    out[tid] = src[tid] * target;
}

template <typename T>
__global__ void AddCUDA1D(const T* src, const T* target, T* out)
{
    const int tid = Grid1DTID(blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
    out[tid] = src[tid] + target[tid];
}

template <typename T>
__global__ void AddCUDA2D(const T* src, const T* target, T* out, const size_t row, const size_t col)
{
    const int tid = Grid2DTID(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
    const int trow = BDim_Y * blockIdx.y + threadIdx.y;
    const int tcol = BDim_X * blockIdx.x + threadIdx.x;
    if (trow >= row || tcol >= col) return;
    out[tid] = src[tid] + target[tid];
}

template <typename T>
__global__ void AddCUDA3D(const T* src, const T* target, T* out, const size_t dimX, const size_t dimY,
                          const size_t dimZ)
{
    const int tid = Grid3DTID(blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimZ = BDim_Z * blockIdx.z + threadIdx.x;
    const size_t tDimY = BDim_Y * blockIdx.y + threadIdx.y;
    const size_t tDimX = BDim_X * blockIdx.x + threadIdx.x;

    if (tDimX >= dimX || tDimY >= dimY || tDimZ >= dimZ) return;
    out[tid] = src[tid] + target[tid];
}

template <typename T>
tesron::Tensor<T>::Tensor(std::initializer_list<size_t> dims)
    : device_(Device::CPU), dimSize_(sizeof(dims)), dims_(dims)
{
}

template <typename T>
tesron::Tensor<T>::Tensor(const T* arr, size_t size)
{
    dataSize_ = size;
    memcpy(data_.get(), arr, size * sizeof(T));
}

template <typename T>
void tesron::Tensor<T>::Cuda()
{
    if (device_ == Device::CUDA) return;
    T* ptr = nullptr;
    cudaMalloc(&ptr, dataSize_);
    auto errCode = cudaMemcpy(ptr, data_, dataSize_, cudaMemcpyHostToDevice);

    if (errCode != cudaSuccess)
        std::cerr << "Error : failed to copy Tensor from host to device, Error Code : " << errCode << std::endl;
    data_.reset(ptr);
}

template <typename T>
void tesron::Tensor<T>::Cpu()
{
    if (device_ == Device::CPU) return;
    T* ptr = nullptr;
    ptr = new T[dataSize_];
    auto errCode = cudaMemcpy(ptr, data_, dataSize_, cudaMemcpyDeviceToHost);

    if (errCode != cudaSuccess)
        std::cerr << "Error : failed to copy Tensor from device to host, Error Code : " << errCode << std::endl;
    cudaFree(data_.get());

    data_.release();
    data_.reset(ptr);
}

template <typename T>
Device tesron::Tensor<T>::getDevice() const
{
    return device_;
}

template <typename T, typename S>
void tesron::Tensor<T>::Multiply(const S x)
{
    if (device_ == Device::CUDA)
    {
        if (dimSize_ == 1 && dataSize_ <= 1024)
        {
            MultiplyCUDA1D<T, S><<<dim3(1, 1, 1), dim3(dims_[0], 1, 1)>>>(data_.get(), x, data_.get());
        }
        else if (dimSize_ == 1 && dataSize_ > 1024)
            MultiplyCUDA1D<T, S>
                <<<dim3(ceil(dataSize_ / 1024.f), 1, 1), dim3(1024, 1, 1)>>>(data_.get(), x, data_.get());
        else if (dimSize_ == 2 && dataSize_ <= 1024)
            MultiplyCUDA2D<T, S>
                <<<dim3(1, 1, 1), dim3(dims_[1], dims_[0], 1)>>>(data_.get(), x, data_.get(), dims_[0], dims_[1]);
        else if (dimSize_ == 2 && dataSize_ > 1024)
        {
            MultiplyCUDA2D<T, S>
                <<<dim3(ceil(dims_[1] / 1024.f), ceil(dims_[0] / 1024.f), 1), dim3(dims_[1], dims_[0], 1)>>>(
                    data_.get(), x, data_.get(), dims_[0], dims_[1]);
        }
        else if (dimSize_ == 3 && dataSize_ <= 1024)
        {
            MultiplyCUDA3D<T, S><<<dim3(1, 1, 1), dim3(dims_[2], dims_[1], dims_[0])>>>(data_.get(), x, data_.get(),
                                                                                        dims_[0], dims_[1], dims_[2]);
        }
        else if (dimSize_ == 3 && dataSize_ > 1024)
        {
            MultiplyCUDA3D<T, S>
                <<<dim3(ceil(dims_[2] / 1024.f), ceil(dims_[1] / 1024.f), ceil(dims_[0] / 1024.f)),
                   dim3(dims_[2], dims_[1], dims_[0])>>>(data_.get(), x, data_.get(), dims_[0], dims_[1], dims_[2]);
                }
        else
        {
            throw "Tensor dim length must be less than 4";
        }
    }
    else if (device_ == Device::CPU)
        for (int i = 0; i < dataSize_; ++i) data_.get()[i] *= x;
}

template <typename T>
void tesron::Tensor<T>::Add(const Tensor& x)
{
    if (device_ == Device::CUDA && x.getDevice() == Device::CUDA)
    {
        if (dimSize_ == 1 && dataSize_ <= 1024)
        {
            AddCUDA1D<T><<<dim3(1, 1, 1), dim3(dims_[0], 1, 1)>>>(data_.get(), x.Data().get(), data_.get());
        }
        else if (dimSize_ == 1 && dataSize_ > 1024)

            AddCUDA1D<T>
                <<<dim3(ceil(dataSize_ / 1024.f), 1, 1), dim3(1024, 1, 1)>>>(data_.get(), x.Data().get(), data_.get());
        else if (dimSize_ == 2 && dataSize_ <= 1024)
            AddCUDA2D<T><<<dim3(1, 1, 1), dim3(dims_[1], dims_[0], 1)>>>(data_.get(), x.Data().get(), data_.get(),
                                                                         dims_[0], dims_[1]);
        else if (dimSize_ == 2 && dataSize_ > 1024)
            AddCUDA2D<T><<<dim3(ceil(dims_[1] / 1024.f), ceil(dims_[0] / 1024.f), 1), dim3(dims_[1], dims_[0], 1)>>>(
                data_.get(), x.Data().get(), data_.get(), dims_[0], dims_[1]);
        else if (dimSize_ == 3 && dataSize_ <= 1024)
            AddCUDA3D<T><<<dim3(1, 1, 1), dim3(dims_[2], dims_[1], dims_[0])>>>(
                data_.get(), x.Data().get(), data_.get(), dims_[0], dims_[1], dims_[2]);
        else if (dimSize_ == 3 && dataSize_ > 1024)
            AddCUDA3D<T><<<dim3(ceil(dims_[2] / 1024.f), ceil(dims_[1] / 1024.f), ceil(dims_[0] / 1024.f)),
                           dim3(dims_[2], dims_[1], dims_[0])>>>(data_.get(), x.Data().get(), data_.get(), dims_[0],
                                                                 dims_[1], dims_[2]);
        else
            throw "Tensor dim length must be less than 4";
    }
    else if (device_ == Device::CPU && x.getDevice() == Device::CPU)
        for (int i = 0; i < dataSize_; ++i) data_.get()[i] += x.Data().get()[i];
}

template <typename T>
T tesron::Tensor<T>::Index(std::initializer_list<size_t> indexList)
{
    std::vector<size_t> tmp(indexList);
    if (tmp.size() != dimSize_) throw "given index list do not match dimention size";

    int idx = 0;

    for (int i = 0; i < dimSize_; ++i)
    {
        idx += tmp[i] * dims_[i];
    }

    return data_.get()[idx];
}

template <typename T>
bool tesron::Tensor<T>::setDims(std::initializer_list<size_t> dimList)
{
    std::vector<size_t> tmp(dimList);
    int size = 0;
    for (const auto& dim : tmp) size += dim;
    if (size != dataSize_) return false;
    dims_.assign(dimList);
    dimSize_ = tmp.size();
    return true;
}
