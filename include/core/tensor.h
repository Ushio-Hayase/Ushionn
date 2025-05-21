// include/ushionn/core/tensor.h
#pragma once

#include <cudnn_frontend.h>

#include <memory>  // for std::unique_ptr
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/common.h"      // USHIONN_ASSERT 등
#include "cuda/cuda_utils.h"  // CUDA_CHECK 등 (common.h를 통해 포함됨)

namespace ushionn
{
// 정적 멤버 초기화 (generate_unique_id_internal용)
static std::atomic<int64_t> tensor_uid_counter = 1;
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

    Tensor(const std::vector<int64_t>& shape, cudnn_frontend::DataType_t data_type = cudnn_frontend::DataType_t::FLOAT,
           bool is_virtual = false, const std::string& name = "");

    // Host 데이터로 생성 (데이터는 즉시 CPU에 위치)
    Tensor(const std::vector<int64_t>& shape,
           const void* host_data_ptr,  // non-owning raw pointer
           size_t num_bytes,           // host_data_ptr의 총 바이트 수
           cudnn_frontend::DataType_t data_type = cudnn_frontend::DataType_t::FLOAT, const std::string& name = "");

    ~Tensor() = default;  // 스마트 포인터가 메모리 관리

    // 복사 및 이동 시맨틱
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // --- 핵심 속성 ---
    const std::vector<int64_t>& get_shape() const { return shape_; }
    const std::vector<int64_t>& get_strides() const;
    cudnn_frontend::DataType_t get_data_type() const { return data_type_; }
    int64_t get_uid() const { return uid_; }
    const std::string& get_name() const { return name_; }
    bool is_virtual() const { return is_virtual_; }
    size_t get_num_elements() const;
    size_t get_size_in_bytes() const;

    /// @brief 데이터의 현재 위치 확인
    /// @return 데이터의 위치 열거형
    DataLocation get_data_location() const { return data_location_; }

    /// @brief CPU에 유효한 데이터가 있는가
    /// @return 유효여부 반환
    bool is_on_host() const;

    /// @brief GPU에 유효한 데이터가 있는가
    /// @return 유효여부 반환
    bool is_on_device() const;  // GPU에 유효한 데이터가 있는가

    // --- 데이터 접근 및 관리 ---

    /// @brief CPU 데이터 수정 시 호출
    /// @return CPU 메모리 포인터 반환
    void* get_mutable_host_ptr();

    /// @brief CPU 데이터 읽기 전용 접근
    /// @return CPU 메모리 읽기 전용 포인터 반환
    const void* get_host_ptr() const;  //

    /// @brief GPU 데이터 수정 시 호출
    /// @return GPU 메모리 포인터 반환
    void* get_mutable_device_ptr();

    /// @brief GPU 데이터 읽기 전용 접근
    /// @return GPU 메모리 읽기 전용 포인터 반환
    const void* get_device_ptr() const;

    /// @brief CPU 메모리만 할당
    void allocate_host_memory(size_t bytes);

    /// @brief GPU 메모리만 할당
    void allocate_device_memory(size_t bytes);  // GPU 메모리만 할당 (DataLocation::DEVICE)

    /* 데이터 전송 및 동기화 */

    /// @brief 데이터를 CPU에서 GPU로 옮김
    /// @param[in] stream cudaStream 포인터
    void to_device(cudaStream_t stream = nullptr);  // 데이터를 GPU로 (최신 상태로 만듦)

    /// @brief 데이터를 GPU에서 CPU로 옮김
    /// @param[in] stream cudaStream 포인터
    void to_host(cudaStream_t stream = nullptr);  // 데이터를 CPU로 (최신 상태로 만듦)

    /// @brief 외부 CPU측 데이터로 텐서의 메모리 채우기
    /// @param[in] host_data_ptr 복사할 위치의 포인터
    /// @param[in] num_bytes 복사할 바이트 수
    /// @param[in, optional] stream cudaStream 포인터
    void fill_from_host(const void* host_data_ptr, size_t num_bytes, cudaStream_t stream = nullptr);

    // 디버깅용
    void print_meta_info(const std::string& header = "") const;

    // cuDNN Graph API 연동 헬퍼
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> create_graph_tensor_attributes(
        std::shared_ptr<cudnn_frontend::graph::Graph>& graph, bool is_input = false, bool is_output = false);

   private:
    std::vector<int64_t> shape_;
    mutable std::vector<int64_t> strides_;
    cudnn_frontend::DataType_t data_type_ = cudnn_frontend::DataType_t::FLOAT;
    int64_t uid_ = tensor_uid_counter.fetch_add(1);
    std::string name_;
    bool is_virtual_ = false;

    // 스마트 포인터로 메모리 관리
    std::unique_ptr<char[], HostDeleter> h_data_ptr_;  // CPU 데이터 (바이트 배열로 관리)
    std::unique_ptr<void, CudaDeleter> d_data_ptr_;    // GPU 데이터

    // 데이터 위치 및 상태 추적 변수
    DataLocation data_location_ = DataLocation::NONE;

    // 내부 상태
    mutable bool strides_dirty_ = true;
    size_t data_size_in_bytes_ = 0;  // 바이트 크기 캐시

    // 내부 헬퍼
    void calculate_strides_if_needed() const;
    void update_size_in_bytes_cache();
};

}  // namespace ushionn