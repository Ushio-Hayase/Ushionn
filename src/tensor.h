#pragma once

#include <memory>
#include <vector>

namespace ushionn
{
enum class DataType
{
    FLOAT32,
    FLOAT64,
};

template <typename T>
struct TypeToEnum;

template <>
struct TypeToEnum<float>
{
    static constexpr DataType value = DataType::FLOAT32;
};
template <>
struct TypeToEnum<double>
{
    static constexpr DataType value = DataType::FLOAT64;
};

template <DataType DType>
struct EnumToType;
template <>
struct EnumToType<DataType::FLOAT32>
{
    using type = float;
};
template <>
struct EnumToType<DataType::FLOAT64>
{
    using type = double;
};

enum class Device
{
    CPU,
    CUDA
};

class Tensor
{
   public:
    Tensor() = delete;

    /// @brief 주어진 벡터로 초기화
    /// @param shapes 데이터 차원 모양
    /// @param data 데이터 벡터
    Tensor(std::initializer_list<size_t> shapes, const std::vector<float>& data);

    /// @brief 주어진 벡터로 초기화
    /// @param shapes 데이터 차원 모양
    /// @param data 데이터 벡터
    Tensor(std::initializer_list<size_t> shapes, const std::vector<double>& data);

    /// @brief 주어진 배열의 크기로 초기화
    /// @param arr 배열의 포인터
    /// @param size 배열의 크기
    Tensor(std::initializer_list<size_t> shapes, const float* arr, size_t size);

    /// @brief 주어진 배열의 크기로 초기화
    /// @param arr 배열의 포인터
    /// @param size 배열의 크기
    Tensor(std::initializer_list<size_t> shapes, const double* arr, size_t size);

    ~Tensor();

    void CUDA();
    void CPU();

    Device GetDevice() const;
    DataType DType() const;
    const std::vector<size_t>& Shape() const;

    template <typename T>
    T Index(std::initializer_list<size_t>);

    bool SetDims(std::initializer_list<size_t>);

    void Add(const Tensor&);

    template <typename S>
    void Mul(const S);

   private:
    size_t GetDTypeSize();

    /// @brief 메인 메모리에 dtype_의 지정한 크기의 배열을 할당합니다.
    /// @param ptr 할당받고 참조할 포인터
    /// @param size 할당받은 크기
    void AllocCPUArray(void** ptr, size_t size);

    template <typename T>
    void AddImpl(const Tensor& x);

    template <typename T, typename S>
    void MulImpl(const S x);

   private:
    std::unique_ptr<void, decltype(&std::free)> data_;  // 메인 데이터 포인터
    size_t data_size_;                                  // 데이터 크기

    std::vector<size_t> shape_;  // 차원 순서
    size_t shape_size_;          // 차원 개수

    Device device_;
    DataType dtype_;
};

}  // namespace ushionn
