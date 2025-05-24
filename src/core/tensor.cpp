#include "core/tensor.h"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemset etc.

#include <atomic>    // for generate_unique_id_internal
#include <iostream>  // for print_meta_info

namespace ushionn
{

// --- 생성자 구현 ---
Tensor::Tensor(const std::vector<int64_t>& shape, cudnn_frontend::DataType_t data_type, bool is_virtual,
               const std::string& name, const std::vector<int64_t>& stride)
    : shape_(shape), data_type_(data_type), is_virtual_(is_virtual), name_(name), strides_(stride)
{
    update_size_in_bytes_cache();
    data_location_ = DataLocation::NONE;
    if (is_virtual_)
    {  // 가상 텐서는 메모리를 직접 소유하지 않음
        h_data_ptr_ = nullptr;
        d_data_ptr_ = nullptr;
    }
}

Tensor::Tensor(const std::vector<int64_t>& shape, const void* host_data_ptr, size_t num_bytes,
               cudnn_frontend::DataType_t data_type, const std::string& name)
    : shape_(shape), data_type_(data_type), is_virtual_(false), name_(name)
{
    update_size_in_bytes_cache();
    USHIONN_ASSERT(num_bytes == data_size_in_bytes_, "Provided data size mismatch with tensor shape and data type.");

    if (host_data_ptr && num_bytes > 0)
    {
        h_data_ptr_.reset(new char[num_bytes]);  // HostDeleter가 관리
        std::memcpy(h_data_ptr_.get(), host_data_ptr, num_bytes);
        data_location_ = DataLocation::HOST;
    }
    else
    {
        data_location_ = DataLocation::NONE;
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(other.strides_),
      data_type_(other.data_type_),
      name_(other.name_ + "_copy"),  // 새 UID, 이름 변경
      is_virtual_(other.is_virtual_),
      data_location_(other.data_location_),  // 복사본은 일단 NONE
      strides_dirty_(other.strides_dirty_),
      data_size_in_bytes_(other.data_size_in_bytes_)
{
    if (is_virtual_) return;

    if (other.is_on_host())
    {
        allocate_host_memory(data_size_in_bytes_);  // 내부에서 data_location_ = DataLocation::HOST 설정
        std::memcpy(h_data_ptr_.get(), other.get_host_ptr(), data_size_in_bytes_);
    }
    if (other.is_on_device())
    {
        allocate_device_memory(data_size_in_bytes_);  // 내부에서 data_location_ 업데이트
        CUDA_CHECK(
            cudaMemcpy(d_data_ptr_.get(), other.get_device_ptr(), data_size_in_bytes_, cudaMemcpyDeviceToDevice));
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this == &other) return *this;

    shape_ = other.shape_;
    strides_ = other.strides_;
    data_type_ = other.data_type_;
    // uid_는 고유해야 하므로, 새로 할당하거나 기존것 유지 정책. 여기선 유지 안함.
    // uid_ = generate_unique_id_internal();
    name_ = other.name_ + "_assigned_copy";
    is_virtual_ = other.is_virtual_;
    strides_dirty_ = other.strides_dirty_;
    data_size_in_bytes_ = other.data_size_in_bytes_;

    // 기존 메모리 해제 (스마트 포인터가 자동으로 처리하지만 명시적으로 reset 가능)
    h_data_ptr_.reset();
    d_data_ptr_.reset();
    data_location_ = DataLocation::NONE;

    if (is_virtual_) return *this;

    if (other.is_on_host())
    {
        allocate_host_memory(data_size_in_bytes_);
        std::memcpy(h_data_ptr_.get(), other.get_host_ptr(), data_size_in_bytes_);
    }
    if (other.is_on_device())
    {
        allocate_device_memory(data_size_in_bytes_);
        CUDA_CHECK(
            cudaMemcpy(d_data_ptr_.get(), other.get_device_ptr(), data_size_in_bytes_, cudaMemcpyDeviceToDevice));
        if (is_on_host() && is_on_device())
        {
            data_location_ = other.data_location_;
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      data_type_(other.data_type_),
      uid_(other.uid_),
      name_(std::move(other.name_)),
      is_virtual_(other.is_virtual_),
      h_data_ptr_(std::move(other.h_data_ptr_)),
      d_data_ptr_(std::move(other.d_data_ptr_)),
      data_location_(other.data_location_),
      strides_dirty_(other.strides_dirty_),
      data_size_in_bytes_(other.data_size_in_bytes_)
{
    // 이동 후 other는 유효하지만 비어있는 상태로 만듦
    other.shape_.clear();
    other.strides_.clear();
    other.uid_ = 0;
    other.is_virtual_ = false;  // 또는 다른 기본 상태
    other.data_location_ = other.data_location_;
    other.data_size_in_bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this == &other) return *this;

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    data_type_ = other.data_type_;
    uid_ = other.uid_;
    name_ = std::move(other.name_);
    is_virtual_ = other.is_virtual_;
    h_data_ptr_ = std::move(other.h_data_ptr_);
    d_data_ptr_ = std::move(other.d_data_ptr_);
    data_location_ = other.data_location_;
    strides_dirty_ = other.strides_dirty_;
    data_size_in_bytes_ = other.data_size_in_bytes_;

    other.shape_.clear();
    other.strides_.clear();
    other.uid_ = 0;
    other.is_virtual_ = true;
    other.data_location_ = DataLocation::NONE;
    other.data_size_in_bytes_ = 0;

    return *this;
}

// --- 데이터 위치 상태 확인 ---
bool Tensor::is_on_host() const
{
    return data_location_ == DataLocation::HOST;
}

bool Tensor::is_on_device() const
{
    return data_location_ == DataLocation::DEVICE;
}

// --- 데이터 접근 ---
void* Tensor::get_mutable_host_ptr()
{
    USHIONN_ASSERT(!is_virtual_, "Cannot get mutable host pointer for a virtual tensor.");
    if (!is_on_host())
    {  // 호스트에 데이터가 없다면 (예: DEVICE 상태)
        // to_host(); // 필요시 자동으로 가져오거나, 에러 처리
        USHIONN_LOG_FATAL(
            "Attempting to get mutable host pointer but no data on host. Call to_host() first or "
            "allocate_host_memory().");
        return nullptr;  // 도달하지 않음
    }
    return h_data_ptr_.get();
}

const void* Tensor::get_host_ptr() const
{
    USHIONN_ASSERT(!is_virtual_, "Cannot get host pointer for a virtual tensor.");
    if (!is_on_host())
    {
        USHIONN_LOG_FATAL(
            "Attempting to get host pointer but no data on host. Call to_host() first or allocate_host_memory().");
        return nullptr;
    }
    // 읽기 전용 접근은 상태를 변경하지 않음 (단, 최신 데이터 보장 X)
    // 만약 최신 데이터 접근을 보장하려면, 여기서 to_host()를 내부적으로 호출하는 정책도 가능
    return h_data_ptr_.get();
}

void* Tensor::get_mutable_device_ptr()
{
    USHIONN_ASSERT(!is_virtual_, "Cannot get mutable device pointer for a virtual tensor.");
    if (!is_on_device())
    {
        // to_device(); // 필요시 자동으로 가져오거나, 에러 처리
        USHIONN_LOG_FATAL(
            "Attempting to get mutable device pointer but no data on device. Call to_device() first or "
            "allocate_device_memory().");
        return nullptr;
    }
    return d_data_ptr_.get();
}

const void* Tensor::get_device_ptr() const
{
    USHIONN_ASSERT(!is_virtual_, "Cannot get device pointer for a virtual tensor.");
    if (!is_on_device())
    {
        USHIONN_LOG_FATAL(
            "Attempting to get device pointer but no data on device. Call to_device() first or "
            "allocate_device_memory().");
        return nullptr;
    }
    return d_data_ptr_.get();
}

// --- 메모리 할당 ---
void Tensor::allocate_host_memory(size_t bytes)
{
    if (is_virtual_)
    {
        USHIONN_WARN("Attempting to allocate host memory for a virtual tensor. Operation ignored.");
        return;
    }
    if (!h_data_ptr_)
    {  // 아직 할당되지 않았을 때만
        data_size_in_bytes_ = bytes;
        USHIONN_ASSERT(data_size_in_bytes_ > 0, "Cannot allocate host memory for zero-sized tensor.");
        h_data_ptr_.reset(new char[data_size_in_bytes_]);  // HostDeleter가 관리
        if (data_location_ == DataLocation::NONE)
        {
            data_location_ = DataLocation::HOST;
        }
    }
}

void Tensor::allocate_device_memory(size_t bytes)
{
    if (is_virtual_)
    {
        USHIONN_WARN("Attempting to allocate device memory for a virtual tensor. Operation ignored.");
        return;
    }
    if (!d_data_ptr_)
    {  // 아직 할당되지 않았을 때만
        data_size_in_bytes_ = bytes;
        USHIONN_ASSERT(data_size_in_bytes_ > 0, "Cannot allocate device memory for zero-sized tensor.");
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, data_size_in_bytes_));
        d_data_ptr_.reset(ptr);  // CudaDeleter가 관리
        if (data_location_ == DataLocation::NONE)
        {
            data_location_ = DataLocation::DEVICE;
        }
    }
}

// --- 데이터 전송 및 동기화 ---
void Tensor::to_device()
{
    if (is_virtual_)
    {
        USHIONN_WARN("to_device() called on a virtual tensor. Operation ignored.");
        return;
    }
    USHIONN_ASSERT(data_size_in_bytes_ > 0, "Cannot transfer data for zero-sized tensor.");

    switch (data_location_)
    {
        case DataLocation::NONE:
            USHIONN_LOG_FATAL("Cannot move to device: No data allocated on host.");
            break;
        case DataLocation::HOST:                          // CPU에만 데이터, GPU로 복사
            allocate_device_memory(data_size_in_bytes_);  // GPU 메모리 할당 (내부에서 data_location_ 변경)
            USHIONN_ASSERT(h_data_ptr_ != nullptr, "Host data pointer is null in HOST state.");
            if (!d_data_ptr_)
                allocate_device_memory(
                    data_size_in_bytes_);  // 만약을 위해 (정상적으론 HOST_AHEAD면 d_data_ptr_도 있어야함)
            CUDA_CHECK(cudaMemcpy(d_data_ptr_.get(), h_data_ptr_.get(), data_size_in_bytes_, cudaMemcpyHostToDevice));
            data_location_ = DataLocation::DEVICE;
            h_data_ptr_.reset();
            break;
        case DataLocation::DEVICE:  // 이미 GPU에만 있음 (최신)
            // 아무것도 안 함
            break;
    }
}

void Tensor::to_host()
{
    if (is_virtual_)
    {
        USHIONN_WARN("to_host() called on a virtual tensor. Operation ignored.");
        return;
    }
    USHIONN_ASSERT(data_size_in_bytes_ > 0, "Cannot transfer data for zero-sized tensor.");

    switch (data_location_)
    {
        case DataLocation::NONE:
            USHIONN_LOG_FATAL("Cannot move to host: No data allocated on device.");
            break;
        case DataLocation::DEVICE:                      // GPU에만 데이터, CPU로 복사
            allocate_host_memory(data_size_in_bytes_);  // CPU 메모리 할당 (내부에서 data_location_ 변경)
            USHIONN_ASSERT(d_data_ptr_ != nullptr, "Device data pointer is null in DEVICE state.");
            if (!h_data_ptr_) allocate_host_memory(data_size_in_bytes_);
            CUDA_CHECK(cudaMemcpy(h_data_ptr_.get(), d_data_ptr_.get(), data_size_in_bytes_, cudaMemcpyDeviceToHost));
            data_location_ = DataLocation::HOST;
            d_data_ptr_.reset();
            break;
        case DataLocation::HOST:
            break;
    }
}

void Tensor::fill_from_host(const void* host_data_ptr, size_t num_bytes, cudaStream_t stream)
{
    if (is_virtual_)
    {
        USHIONN_WARN("fill_from_host() called on a virtual tensor. Operation ignored.");
        return;
    }
    USHIONN_ASSERT(host_data_ptr != nullptr, "Source host_data_ptr cannot be null.");
    USHIONN_ASSERT(num_bytes == data_size_in_bytes_, "Data size mismatch for fill_from_host.");

    USHIONN_ASSERT(data_location_ != DataLocation::NONE, "Memory must be allocated before the value is filled");

    if (data_location_ == DataLocation::HOST)
    {
        allocate_host_memory(data_size_in_bytes_);  // CPU 메모리가 없으면 할당
        std::memcpy(h_data_ptr_.get(), host_data_ptr, num_bytes);
    }
    else if (data_location_ == DataLocation::DEVICE)
    {
        allocate_device_memory(data_size_in_bytes_);
        cudaMemcpy(d_data_ptr_.get(), host_data_ptr, num_bytes, cudaMemcpyHostToDevice);
    }
}
// --- 기타 헬퍼 ---
void Tensor::update_size_in_bytes_cache()
{
    if (shape_.empty() || is_virtual_)
    {
        data_size_in_bytes_ = 0;
        return;
    }
    size_t num_elements = 1;
    for (int64_t dim : shape_)
    {
        num_elements *= dim;
    }
    size_t element_size = 0;
    switch (data_type_)
    {
        case cudnn_frontend::DataType_t::FLOAT:
            element_size = sizeof(float);
            break;
        case cudnn_frontend::DataType_t::HALF:
            element_size = sizeof(uint16_t);
            break;  // half float
        case cudnn_frontend::DataType_t::INT8:
            element_size = sizeof(int8_t);
            break;
        // 다른 데이터 타입 추가
        default:
            USHIONN_LOG_FATAL("Unsupported data type for size calculation.");
    }
    data_size_in_bytes_ = num_elements * element_size;
}

const std::vector<int64_t>& Tensor::get_strides() const
{
    calculate_strides_if_needed();
    return strides_;
}

size_t Tensor::get_num_elements() const
{
    if (shape_.empty()) return 0;
    size_t num = 1;
    for (int64_t d : shape_) num *= d;
    return num;
}

size_t Tensor::get_size_in_bytes() const
{
    return data_size_in_bytes_;
}

void Tensor::calculate_strides_if_needed() const
{
    if (!strides_dirty_ || shape_.empty())
    {
        return;
    }
    strides_.resize(shape_.size());
    if (shape_.empty())
    {  // 스칼라 또는 빈 텐서
        strides_dirty_ = false;
        return;
    }
    // NCHW 레이아웃 가정 (또는 설정 가능하도록 변경)
    // 가장 안쪽 차원부터 스트라이드 계산
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
    {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
    strides_dirty_ = false;
}

void Tensor::print_meta_info(const std::string& header) const
{
    if (!header.empty())
    {
        std::cout << "--- " << header << " ---" << std::endl;
    }
    std::cout << "Tensor '" << (name_.empty() ? "Unnamed" : name_) << "' (UID: " << uid_ << ")" << std::endl;
    std::cout << "  Virtual: " << (is_virtual_ ? "Yes" : "No") << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::cout << "  Data Type: ";
    switch (data_type_)
    {
        case cudnn_frontend::DataType_t::FLOAT:
            std::cout << "FLOAT";
            break;
        case cudnn_frontend::DataType_t::HALF:
            std::cout << "HALF";
            break;
        case cudnn_frontend::DataType_t::INT8:
            std::cout << "INT8";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "  Size in Bytes: " << get_size_in_bytes() << std::endl;
    std::cout << "  Strides: [";
    const auto s = get_strides();
    for (size_t i = 0; i < s.size(); ++i)
    {
        std::cout << s[i] << (i == s.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::cout << "  Data Location: ";
    switch (data_location_)
    {
        case DataLocation::NONE:
            std::cout << "NONE";
            break;
        case DataLocation::HOST:
            std::cout << "HOST";
            break;
        case DataLocation::DEVICE:
            std::cout << "DEVICE";
            break;
    }
    std::cout << std::endl;
    std::cout << "  Host Ptr: " << (h_data_ptr_ ? "Valid" : "Null")
              << ", Device Ptr: " << (d_data_ptr_ ? "Valid" : "Null") << std::endl;
}

std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> Tensor::create_graph_tensor_attributes(
    std::shared_ptr<cudnn_frontend::graph::Graph>& graph, bool is_input, bool is_output) const
{
    USHIONN_ASSERT(graph != nullptr, "Graph object cannot be null.");
    auto tensor_attrs = cudnn_frontend::graph::Tensor_attributes()
                            .set_name(name_.c_str())    // c_str() 사용
                            .set_dim(shape_)            // std::vector<int64_t> 직접 사용 가능
                            .set_stride(get_strides())  // std::vector<int64_t> 직접 사용 가능
                            .set_data_type(data_type_)
                            .set_is_virtual(is_virtual_)
                            .set_uid(uid_);  // Tensor 객체의 UID 사용

    // 실제 cudnn_frontend::graph::Tensor 객체 생성
    auto graph_tensor_ref = graph->tensor(tensor_attrs);
    USHIONN_ASSERT(graph_tensor_ref != nullptr, "Failed to create graph tensor object from attributes.");

    // 입력 또는 출력으로 표시
    if (is_input && !is_output)
    {  // 순수 입력
        graph_tensor_ref->set_output(false);
    }
    else if (is_output)
    {  // 출력이면 is_input 여부와 관계없이 output=true (일반적으로)
        graph_tensor_ref->set_output(true);
    }
    // 그 외 경우는 (예: 가중치, 상수) set_output 기본값(false) 유지 또는 명시적 설정

    return graph_tensor_ref;
}

}  // namespace ushionn