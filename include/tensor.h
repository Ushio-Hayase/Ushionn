#pragma once

#include <array>
#include <vector>

template <typename Ty>
class Tensor
{
   public:
    Tensor();
    Tensor(const Tensor& tensor) = default;
    Tensor(Tensor&& tensor) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    /// @brief 텐서의 차원
    /// @return 텐서의 차원 수를 반환합니다.
    size_t Dims();

    /* 전치 */
    void T();
    void T(vector<int>&& axes);

    /// @brief 주어진 텐서의 크기로 주어진 값만큼 채운 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @param value 채울 값
    /// @return 값이 채워진 텐서
    static Tensor newFull(const Tensor& tensor, T value);

    /// @brief 주어진 텐서의 크기로 주어진 값만큼 채운 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @param value 채울 값
    /// @return 값이 채워진 텐서
    static Tensor newFull(Tensor&& tensor, T value);

    /// @brief 주어진 크기로 빈 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 빈 텐서
    static Tensor newEmpty(Tensor&& tensor);

    /// @brief 주어진 크기로 빈 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 빈 텐서
    static Tensor newEmpty(const Tensor& tensor);

    /// @brief 주어진 크기로 1로 채워진 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 1로 채워진 텐서
    static Tensor newOnes(const Tensor& tensor);

    /// @brief 주어진 크기로 1로 채워진 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 1로 채워진 텐서
    static Tensor newOnes(Tensor&& tensor);

    /// @brief 주어진 크기로 0으로 채워진 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 0으로 채워진 텐서
    static Tensor newZeros(const Tensor& tensor);

    /// @brief 주어진 크기로 0으로 채워진 텐서를 만듭니다.
    /// @param tensor 바탕으로 할 텐서
    /// @return 0으로 채워진 텐서
    static Tensor newZeros(Tensor&& tensor);

    /// @brief 쿠다를 사용중인지 여부를 반환합니다.
    /// @return 쿠다 사용 여부
    bool isCuda() noexcept;

    /// @brief 텐서를 쿠다를 사용하도록 변경합니다.
    void Cuda();

    /// @brief 텐서를 CPU를 사용하도록 변경합니다.
    void Cpu();

   private:
    Ty* data_;
    std::vector<size_t> lengths_;
    size_t dims_;

    bool usingCuda_;
};