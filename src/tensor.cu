#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "constant.h"
#include "tensor.h"

ushionn::Tensor::Tensor(std::initializer_list<size_t> shapes, const std::vector<float>& data)
    : dtype_(DataType::FLOAT32),
      shape_(shapes),
      data_(std::malloc(sizeof(float) * data.size()), &std::free),
      data_size_(data.size()),
      shape_size_(shapes.size()),
      device_(Device::CPU)
{
    std::copy(data.begin(), data.end(), static_cast<float*>(data_.get()));
}

ushionn::Tensor::Tensor(std::initializer_list<size_t> shapes, const std::vector<double>& data)
    : dtype_(DataType::FLOAT64),
      shape_(shapes),
      data_(std::malloc(sizeof(double) * data.size()), &std::free),
      data_size_(data.size()),
      shape_size_(shapes.size()),
      device_(Device::CPU)
{
    std::copy(data.begin(), data.end(), static_cast<double*>(data_.get()));
}

ushionn::Tensor::Tensor(std::initializer_list<size_t> shapes, const float* arr, size_t size)
    : dtype_(DataType::FLOAT32),
      shape_(shapes),
      data_(std::malloc(sizeof(float) * size), &std::free),
      data_size_(size),
      shape_size_(shapes.size()),
      device_(Device::CPU)
{
    std::memcpy(static_cast<float*>(data_.get()), arr, sizeof(float) * size);
}

ushionn::Tensor::Tensor(std::initializer_list<size_t> shapes, const double* arr, size_t size)
    : dtype_(DataType::FLOAT64),
      shape_(shapes),
      data_(std::malloc(sizeof(double) * size), &std::free),
      data_size_(size),
      shape_size_(shapes.size()),
      device_(Device::CPU)
{
    std::memcpy(static_cast<double*>(data_.get()), arr, sizeof(double) * size);
}

ushionn::Tensor::~Tensor()
{
    this->CPU();
}

void ushionn::Tensor::CUDA()
{
    if (device_ == Device::CUDA) return;
    void* ptr = nullptr;

    cudaMalloc(&ptr, data_size_ * GetDTypeSize());

    auto err_code = cudaMemcpy(ptr, data_.get(), data_size_ * GetDTypeSize(), cudaMemcpyHostToDevice);
    device_ = Device::CUDA;

    if (err_code != cudaSuccess)
    {
        std::cerr << "Error : failed to copy Tensor from host to device, Error Code : " << err_code << std::endl;
        cudaFree(ptr);
        device_ = Device::CPU;
    }
    else
    {
        data_.reset(ptr);
    }
}

void ushionn::Tensor::CPU()
{
    if (device_ == Device::CPU) return;
    void* old_ptr = data_.get();
    void* new_ptr = nullptr;

    AllocCPUArray(&new_ptr, data_size_);

    auto err_code = cudaMemcpy(new_ptr, old_ptr, data_size_ * GetDTypeSize(), cudaMemcpyDeviceToHost);
    device_ = Device::CPU;

    if (err_code != cudaSuccess)
    {
        std::cerr << "Error : failed to copy Tensor from device to host, Error Code : " << cudaGetErrorString(err_code)
                  << std::endl;
    }
    else
    {
        cudaFree(data_.get());
        data_.release();
        data_.reset(new_ptr);
    }
}

ushionn::Device ushionn::Tensor::GetDevice() const
{
    return device_;
}

template <typename T>
T ushionn::Tensor::Index(std::initializer_list<size_t> indexList)
{
    std::vector<size_t> tmp(indexList);
    if (tmp.size() != shape_size_)
    {
        std::cerr << "Error : Given index list do not match dimension size" << std::endl;
        throw "given index list do not match dimension size";
    }

    int idx = 0;

    for (int i = 0; i < shape_size_; ++i)
    {
        int multiple = 1;
        for (int j = shape_size_ - 1; j > i; --j) multiple *= shape_[j];
        idx += tmp[i] * multiple;
    }

    return static_cast<T*>(data_.get())[idx];
}

bool ushionn::Tensor::SetDims(std::initializer_list<size_t> dimList)
{
    std::vector<size_t> tmp(dimList);
    int size = 1;
    for (const auto& dim : tmp) size *= dim;
    if (size != data_size_) return false;
    shape_.assign(dimList);
    shape_size_ = tmp.size();
    return true;
}

size_t ushionn::Tensor::GetDTypeSize()
{
    if (dtype_ == DataType::FLOAT32)
        return sizeof(float);
    else if (dtype_ == DataType::FLOAT64)
        return sizeof(double);
}

void ushionn::Tensor::AllocCPUArray(void** ptr, size_t size)
{
    if (dtype_ == DataType::FLOAT32)
        *ptr = std::malloc(size * sizeof(float));
    else if (dtype_ == DataType::FLOAT64)
        *ptr = std::malloc(size * sizeof(double));
}

template <typename T>
__global__ void MultiplyCUDA1D(const T* src, const T target, T* out, const size_t dimX)
{
    const int tid = Grid1DTID(blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimX = GDim_X * blockIdx.x + threadIdx.x;
    if (tDimX >= dimX) return;
    out[tid] = src[tid] * target;
}

template <typename T>
__global__ void MultiplyCUDA2D(const T* src, const T target, T* out, const size_t dimX, const size_t dimY)
{
    const int tid = Grid2DTID(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
    const int tDimY = GDim_Y * blockIdx.y + threadIdx.y;
    const int tDimX = GDim_X * blockIdx.x + threadIdx.x;
    if (tDimX >= dimX || tDimY >= dimY) return;
    out[tid] = src[tid] * target;
}

template <typename T>
__global__ void MultiplyCUDA3D(const T* src, const T target, T* out, const size_t dimX, const size_t dimY,
                               const size_t dimZ)
{
    const int tid = Grid3DTID(blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimZ = GDim_Z * blockIdx.z + threadIdx.z;
    const size_t tDimY = GDim_Y * blockIdx.y + threadIdx.y;
    const size_t tDimX = GDim_X * blockIdx.x + threadIdx.x;

    if (tDimX >= dimX || tDimY >= dimY || tDimZ >= dimZ) return;
    out[tid] = src[tid] * target;
}

template <typename T>
__global__ void AddCUDA1D(const T* src, const T* target, T* out, const size_t dimX)
{
    const int tid = Grid1DTID(blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimX = GDim_X * blockIdx.x + threadIdx.x;
    if (tDimX >= dimX) return;
    out[tid] = src[tid] + target[tid];
}

template <typename T>
__global__ void AddCUDA2D(const T* src, const T* target, T* out, const size_t dimX, const size_t dimY)
{
    const int tid = Grid2DTID(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z);
    const int tDimY = GDim_Y * blockIdx.y + threadIdx.y;
    const int tDimX = GDim_X * blockIdx.x + threadIdx.x;
    if (tDimX >= dimX || tDimY >= dimY) return;
    out[tid] = src[tid] + target[tid];
}

template <typename T>
__global__ void AddCUDA3D(const T* src, const T* target, T* out, const size_t dimX, const size_t dimY,
                          const size_t dimZ)
{
    const int tid = Grid3DTID(blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    const size_t tDimZ = GDim_Z * blockIdx.z + threadIdx.z;
    const size_t tDimY = GDim_Y * blockIdx.y + threadIdx.y;
    const size_t tDimX = GDim_X * blockIdx.x + threadIdx.x;

    if (tDimX >= dimX || tDimY >= dimY || tDimZ >= dimZ) return;
    out[tid] = src[tid] + target[tid];
}

template <typename S>
void ushionn::Tensor::Mul(const S x)
{
    if (dtype_ == DataType::FLOAT32)
    {
        MulImpl<float, S>(x);
    }
    else if (dtype_ == DataType::FLOAT64)
        MulImpl<double, S>(x);
}

void ushionn::Tensor::Add(const ushionn::Tensor& x)
{
    if (dtype_ == DataType::FLOAT32)
        AddImpl<float>(x);
    else if (dtype_ == DataType::FLOAT64)
        AddImpl<double>(x);
}

template <typename T>
void ushionn::Tensor::AddImpl(const ushionn::Tensor& x)
{
    if (device_ == Device::CUDA && x.GetDevice() == Device::CUDA && shape_ == x.shape_)
    {
        if (shape_size_ == 1 && data_size_ <= 1024)
        {
            AddCUDA1D<T><<<dim3(1, 1, 1), dim3(shape_[0], 1, 1)>>>(
                static_cast<T*>(data_.get()), static_cast<T*>(x.data_.get()), static_cast<T*>(data_.get()), shape_[0]);
        }
        else if (shape_size_ == 1 && data_size_ > 1024)
        {
            AddCUDA1D<T><<<dim3(ceil(data_size_ / 1024.f), 1, 1), dim3(kblockSize1D, 1, 1)>>>(
                static_cast<T*>(data_.get()), static_cast<T*>(x.data_.get()), static_cast<T*>(data_.get()), shape_[0]);
        }
        else if (shape_size_ == 2 && data_size_ <= 1024)
        {
            AddCUDA2D<T><<<dim3(1, 1, 1), dim3(shape_[1], shape_[0], 1)>>>(
                static_cast<T*>(data_.get()), static_cast<T*>(x.data_.get()), static_cast<T*>(data_.get()), shape_[1],
                shape_[0]);
        }
        else if (shape_size_ == 2 && data_size_ > 1024)
        {
            AddCUDA2D<T><<<dim3(ceil(shape_[1] / static_cast<float>(kblockSize2D)),
                                ceil(shape_[0] / static_cast<float>(kblockSize2D))),
                           dim3(kblockSize2D, kblockSize2D, 1)>>>(static_cast<T*>(data_.get()),
                                                                  static_cast<T*>(x.data_.get()),
                                                                  static_cast<T*>(data_.get()), shape_[1], shape_[0]);
        }
        else if (shape_size_ == 3 && data_size_ <= 1024)
        {
            AddCUDA3D<T><<<dim3(1, 1, 1), dim3(shape_[2], shape_[1], shape_[0])>>>(
                static_cast<T*>(data_.get()), static_cast<T*>(x.data_.get()), static_cast<T*>(data_.get()), shape_[2],
                shape_[1], shape_[0]);
        }
        else if (shape_size_ == 3 && data_size_ > 1024)
        {
            AddCUDA3D<T><<<dim3(ceil(shape_[2] / static_cast<float>(kblockSize3DX)),
                                ceil(shape_[1] / static_cast<float>(kblockSize3DYZ)),
                                ceil(shape_[0] / static_cast<float>(kblockSize3DYZ))),
                           dim3(kblockSize3DX, kblockSize3DYZ, kblockSize3DYZ)>>>(
                static_cast<T*>(data_.get()), static_cast<T*>(x.data_.get()), static_cast<T*>(data_.get()), shape_[2],
                shape_[1], shape_[0]);
        }
        else
        {
            std::cerr << "Error : Tensor dim must be 1, 2, 3" << std::endl;
            throw "Tensor dim must be 1, 2, 3";
        }
    }
    else if (device_ == Device::CPU && x.GetDevice() == Device::CPU && shape_ == x.shape_)
    {
        for (int i = 0; i < data_size_; ++i) static_cast<T*>(data_.get())[i] += static_cast<T*>(x.data_.get())[i];
    }
    else if (device_ == x.device_ && shape_ != x.shape_)
    {
        std::cerr << "Error : Tensors need to be in the same dimension" << std::endl;
        throw "Tensors need to be in the same dimension";
    }
    else
    {
        std::cerr << "Error : Tensors need to be in the same device" << std::endl;
        throw "Tensors need to be in the same device";
    }
}

template <typename T, typename S>
void ushionn::Tensor::MulImpl(const S x)
{
    if (device_ == Device::CUDA)
    {
        if (shape_size_ == 1 && data_size_ <= 1024)
        {
            MultiplyCUDA1D<T><<<dim3(1, 1, 1), dim3(shape_[0], 1, 1)>>>(static_cast<T*>(data_.get()), x,
                                                                        static_cast<T*>(data_.get()), shape_[0]);
        }
        else if (shape_size_ == 1 && data_size_ > 1024)
        {
            MultiplyCUDA1D<T><<<dim3(ceil(data_size_ / 1024.f), 1, 1), dim3(kblockSize1D, 1, 1)>>>(
                static_cast<T*>(data_.get()), x, static_cast<T*>(data_.get()), shape_[0]);
        }
        else if (shape_size_ == 2 && data_size_ <= 1024)
        {
            MultiplyCUDA2D<T><<<dim3(1, 1, 1), dim3(shape_[1], shape_[0], 1)>>>(
                static_cast<T*>(data_.get()), x, static_cast<T*>(data_.get()), shape_[1], shape_[0]);
        }
        else if (shape_size_ == 2 && data_size_ > 1024)
        {
            MultiplyCUDA2D<T><<<dim3(ceil(shape_[1] / static_cast<float>(kblockSize2D)),
                                     ceil(shape_[0] / static_cast<float>(kblockSize2D))),
                                dim3(kblockSize2D, kblockSize2D, 1)>>>(
                static_cast<T*>(data_.get()), x, static_cast<T*>(data_.get()), shape_[1], shape_[0]);
        }
        else if (shape_size_ == 3 && data_size_ <= 1024)
        {
            MultiplyCUDA3D<T><<<dim3(1, 1, 1), dim3(shape_[2], shape_[1], shape_[0])>>>(
                static_cast<T*>(data_.get()), x, static_cast<T*>(data_.get()), shape_[2], shape_[1], shape_[0]);
        }
        else if (shape_size_ == 3 && data_size_ > 1024)
        {
            MultiplyCUDA3D<T><<<dim3(ceil(shape_[2] / static_cast<float>(kblockSize3DX)),
                                     ceil(shape_[1] / static_cast<float>(kblockSize3DYZ)),
                                     ceil(shape_[0] / static_cast<float>(kblockSize3DYZ))),
                                dim3(kblockSize3DX, kblockSize3DYZ, kblockSize3DYZ)>>>(
                static_cast<T*>(data_.get()), x, static_cast<T*>(data_.get()), shape_[2], shape_[1], shape_[0]);
        }
        else
        {
            std::cerr << "Error : Tensor dim length must be less than 4" << std::endl;
            throw "Tensor dim length must be less than 4";
        }
    }
    else if (device_ == Device::CPU)
    {
        for (int i = 0; i < data_size_; ++i) static_cast<T*>(data_.get())[i] *= x;
    }
}

template int ushionn::Tensor::Index(std::initializer_list<size_t>);
template float ushionn::Tensor::Index(std::initializer_list<size_t>);
template double ushionn::Tensor::Index(std::initializer_list<size_t>);
template void ushionn::Tensor::Mul(int);
template void ushionn::Tensor::Mul(float);
template void ushionn::Tensor::Mul(double);