// tests/test_tensor.cpp
#include <gtest/gtest.h>

#include <algorithm>  // for std::equal
#include <numeric>    // for std::iota
#include <vector>

#include "core/tensor.h"  // UshioNN Tensor 클래스 헤더

// 테스트 환경 초기화 (선택적, 모든 테스트 전에 CUDA 장치 설정 등)
class TensorTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        // 모든 테스트 케이스 시작 전에 실행될 코드
        // 예: 특정 GPU 장치 선택
        // CUDA_CHECK(cudaSetDevice(0));
        ushionn::cuda::utils::printGpuMemoryUsage("Before test case");
    }

    void TearDown() override
    {
        // 모든 테스트 케이스 종료 후에 실행될 코드
        ushionn::cuda::utils::printGpuMemoryUsage("After test case");
    }

    // Helper 함수: 두 float 배열 비교
    static bool AreArraysEqual(const float* arr1, const float* arr2, size_t size, float tolerance = 1e-6f)
    {
        if (!arr1 || !arr2) return false;
        for (size_t i = 0; i < size; ++i)
        {
            if (std::abs(arr1[i] - arr2[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }
};

// --- 기본 생성 및 속성 테스트 ---
TEST_F(TensorTest, DefaultConstructor)
{
    ushionn::Tensor t;
    EXPECT_TRUE(t.get_shape().empty());
    EXPECT_EQ(t.get_data_type(), cudnn_frontend::DataType_t::FLOAT);  // 기본값 확인
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::NONE);
    EXPECT_EQ(t.get_num_elements(), 0);
    EXPECT_EQ(t.get_size_in_bytes(), 0);
    EXPECT_FALSE(t.is_on_host());
    EXPECT_FALSE(t.is_on_device());
    EXPECT_NE(t.get_uid(), 0);  // UID는 할당되어야 함
}

TEST_F(TensorTest, ShapeConstructor)
{
    std::vector<int64_t> shape = {2, 3, 4};
    ushionn::Tensor t(shape, cudnn_frontend::DataType_t::FLOAT, false, "test_tensor");

    EXPECT_EQ(t.get_shape(), shape);
    EXPECT_EQ(t.get_name(), "test_tensor");
    EXPECT_EQ(t.get_data_type(), cudnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::NONE);  // 아직 메모리 할당 안 함
    EXPECT_EQ(t.get_num_elements(), 2 * 3 * 4);
    EXPECT_EQ(t.get_size_in_bytes(), 2 * 3 * 4 * sizeof(float));
    EXPECT_NE(t.get_uid(), 0);
}

TEST_F(TensorTest, StridesCalculationNCHW)
{
    std::vector<int64_t> shape = {2, 3, 4, 5};  // N, C, H, W
    ushionn::Tensor t(shape);
    const auto& strides = t.get_strides();
    ASSERT_EQ(strides.size(), 4);
    EXPECT_EQ(strides[0], 3 * 4 * 5);  // Stride N
    EXPECT_EQ(strides[1], 4 * 5);      // Stride C
    EXPECT_EQ(strides[2], 5);          // Stride H
    EXPECT_EQ(strides[3], 1);          // Stride W
}

// --- 호스트 메모리 관리 테스트 ---
TEST_F(TensorTest, AllocateHostMemory)
{
    ushionn::Tensor t({2, 2});
    ASSERT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::NONE);
    t.allocate_host_memory();
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    EXPECT_TRUE(t.is_on_host());
    EXPECT_FALSE(t.is_on_device());
    ASSERT_NE(t.get_host_ptr(), nullptr);  // get_host_ptr()은 const void* 반환
}

TEST_F(TensorTest, FillFromHostAndAccess)
{
    std::vector<int64_t> shape = {2, 2};
    size_t num_elements = 4;
    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 1.0f);  // 1.0, 2.0, 3.0, 4.0

    ushionn::Tensor t(shape);
    t.fill_from_host(host_data.data(), num_elements * sizeof(float));

    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    const float* t_host_ptr = static_cast<const float*>(t.get_host_ptr());
    ASSERT_NE(t_host_ptr, nullptr);
    EXPECT_TRUE(AreArraysEqual(t_host_ptr, host_data.data(), num_elements));

    // Mutable access
    float* t_mutable_host_ptr = static_cast<float*>(t.get_mutable_host_ptr());
    ASSERT_NE(t_mutable_host_ptr, nullptr);
    t_mutable_host_ptr[0] = 100.0f;
    EXPECT_EQ(t_mutable_host_ptr[0], 100.0f);
    // 이 시점에서 DataLocation은 HOST_AHEAD가 될 수 있음 (이전 설계)
    // 현재 단순화된 단일 위치 설계에서는 여전히 HOST
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
}

TEST_F(TensorTest, ConstructorWithHostData)
{
    std::vector<int64_t> shape = {2, 2};
    size_t num_elements = 4;
    std::vector<float> host_data = {1.1f, 2.2f, 3.3f, 4.4f};

    ushionn::Tensor t(shape, host_data.data(), num_elements * sizeof(float));
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    const float* t_host_ptr = static_cast<const float*>(t.get_host_ptr());
    ASSERT_NE(t_host_ptr, nullptr);
    EXPECT_TRUE(AreArraysEqual(t_host_ptr, host_data.data(), num_elements));
}

// --- GPU 메모리 관리 테스트 ---
TEST_F(TensorTest, AllocateDeviceMemory)
{
    ushionn::Tensor t({3, 3});
    ASSERT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::NONE);
    t.allocate_device_memory();
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::DEVICE);
    EXPECT_FALSE(t.is_on_host());
    EXPECT_TRUE(t.is_on_device());
    ASSERT_NE(t.get_device_ptr(), nullptr);
}

// --- 데이터 전송 테스트 (단일 위치 강제 설계 기준) ---
TEST_F(TensorTest, TransferToDevice)
{
    std::vector<int64_t> shape = {2, 1};
    size_t num_elements = 2;
    std::vector<float> host_data = {10.0f, 20.0f};

    ushionn::Tensor t(shape);
    t.fill_from_host(host_data.data(), num_elements * sizeof(float));  // HOST 상태
    ASSERT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    ASSERT_NE(t.get_host_ptr(), nullptr);
    EXPECT_THROW(t.get_device_ptr(), std::runtime_error);  // 단순화된 설계에서는 아직 device ptr 없음

    t.to_device();  // 데이터 이동 (HOST -> DEVICE)
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::DEVICE);
    EXPECT_EQ(t.get_host_ptr(), nullptr);  // Host 메모리는 해제되어야 함
    ASSERT_NE(t.get_device_ptr(), nullptr);

    // GPU 데이터 검증 (CPU로 다시 가져와서 확인)
    std::vector<float> host_data_check(num_elements);
    CUDA_CHECK(
        cudaMemcpy(host_data_check.data(), t.get_device_ptr(), num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(AreArraysEqual(host_data_check.data(), host_data.data(), num_elements));
}

TEST_F(TensorTest, TransferToHost)
{
    std::vector<int64_t> shape = {1, 3};
    size_t num_elements = 3;
    float val_to_fill = 7.0f;

    ushionn::Tensor t(shape);
    t.allocate_device_memory();  // DEVICE 상태
    // GPU 메모리를 특정 값으로 채우기 (테스트용)
    CUDA_CHECK(cudaMemset(t.get_mutable_device_ptr(), 0, num_elements * sizeof(float)));  // 0으로 초기화
    // 간단한 커널로 채우거나, 직접 값을 설정하는 CUDA 함수가 있다면 사용. 여기서는 cudaMemset 사용.
    // 또는, CPU에서 데이터를 만들고 to_device()로 보낸 후 테스트 시작.

    // 여기서는 이미 Device에 있는 상태에서 Host로 옮기는 것을 테스트
    // 만약 Device 데이터를 설정하고 싶다면:
    std::vector<float> initial_gpu_data = {7.1f, 7.2f, 7.3f};
    CUDA_CHECK(cudaMemcpy(t.get_mutable_device_ptr(), initial_gpu_data.data(), num_elements * sizeof(float),
                          cudaMemcpyHostToDevice));
    ASSERT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::DEVICE);

    t.to_host();  // 데이터 이동 (DEVICE -> HOST), Device 메모리 해제
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    ASSERT_NE(t.get_host_ptr(), nullptr);
    EXPECT_EQ(t.get_device_ptr(), nullptr);  // Device 메모리는 해제되어야 함

    const float* t_host_ptr = static_cast<const float*>(t.get_host_ptr());
    ASSERT_NE(t_host_ptr, nullptr);
    EXPECT_TRUE(AreArraysEqual(t_host_ptr, initial_gpu_data.data(), num_elements));
}

TEST_F(TensorTest, TransferToDeviceAlreadyOnDevice)
{
    ushionn::Tensor t({1, 1});
    t.allocate_device_memory();
    const void* initial_d_ptr = t.get_device_ptr();
    t.to_device();  // 이미 Device에 있으므로 아무 작업도 안해야 함
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::DEVICE);
    EXPECT_EQ(t.get_device_ptr(), initial_d_ptr);  // 포인터가 변경되지 않았는지 확인
    EXPECT_EQ(t.get_host_ptr(), nullptr);
}

TEST_F(TensorTest, TransferToHostAlreadyOnHost)
{
    ushionn::Tensor t({1, 1});
    t.allocate_host_memory();
    const void* initial_h_ptr = t.get_host_ptr();
    t.to_host();  // 이미 Host에 있으므로 아무 작업도 안해야 함
    EXPECT_EQ(t.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    EXPECT_EQ(t.get_host_ptr(), initial_h_ptr);
    EXPECT_EQ(t.get_device_ptr(), nullptr);
}

// --- 복사 및 이동 생성자/대입 연산자 테스트 (단일 위치 강제 설계 기준) ---
TEST_F(TensorTest, CopyConstructorHost)
{
    std::vector<float> data = {1, 2, 3, 4};
    ushionn::Tensor original({2, 2}, data.data(), data.size() * sizeof(float));  // HOST 상태
    ushionn::Tensor copy = original;                                             // 복사 생성자

    EXPECT_NE(copy.get_uid(), original.get_uid());  // UID는 달라야 함
    EXPECT_EQ(copy.get_shape(), original.get_shape());
    EXPECT_EQ(copy.get_data_location(), ushionn::Tensor::DataLocation::HOST);  // 복사본도 HOST
    ASSERT_NE(copy.get_host_ptr(), nullptr);
    EXPECT_NE(copy.get_host_ptr(), original.get_host_ptr());  // 서로 다른 메모리
    EXPECT_EQ(copy.get_device_ptr(), nullptr);

    const float* orig_h = static_cast<const float*>(original.get_host_ptr());
    const float* copy_h = static_cast<const float*>(copy.get_host_ptr());
    EXPECT_TRUE(AreArraysEqual(orig_h, copy_h, data.size()));
}

TEST_F(TensorTest, CopyConstructorDevice)
{
    std::vector<float> data = {5, 6, 7, 8};
    ushionn::Tensor original_host({2, 2}, data.data(), data.size() * sizeof(float));
    original_host.to_device();  // DEVICE 상태로 만듦

    ushionn::Tensor copy = original_host;  // 복사 생성자

    EXPECT_NE(copy.get_uid(), original_host.get_uid());
    EXPECT_EQ(copy.get_shape(), original_host.get_shape());
    EXPECT_EQ(copy.get_data_location(), ushionn::Tensor::DataLocation::DEVICE);  // 복사본도 DEVICE
    EXPECT_EQ(copy.get_host_ptr(), nullptr);
    ASSERT_NE(copy.get_device_ptr(), nullptr);
    EXPECT_NE(copy.get_device_ptr(), original_host.get_device_ptr());  // 서로 다른 GPU 메모리

    // GPU 데이터 검증 (각각 CPU로 가져와서 비교)
    std::vector<float> orig_check(data.size());
    std::vector<float> copy_check(data.size());
    CUDA_CHECK(cudaMemcpy(orig_check.data(), original_host.get_device_ptr(), data.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(copy_check.data(), copy.get_device_ptr(), data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(AreArraysEqual(orig_check.data(), copy_check.data(), data.size()));
}

TEST_F(TensorTest, MoveConstructor)
{
    std::vector<float> data = {10, 20};
    ushionn::Tensor original({1, 2}, data.data(), data.size() * sizeof(float), cudnn_frontend::DataType_t::FLOAT,
                             "move_orig");
    int64_t orig_uid = original.get_uid();
    void* orig_h_ptr = original.get_mutable_host_ptr();  // get_mutable_host_ptr()은 void* 반환
    ASSERT_EQ(original.get_data_location(), ushionn::Tensor::DataLocation::HOST);

    ushionn::Tensor moved = std::move(original);  // 이동 생성자

    EXPECT_EQ(moved.get_uid(), orig_uid);  // UID는 유지
    EXPECT_EQ(moved.get_name(), "move_orig");
    EXPECT_EQ(moved.get_data_location(), ushionn::Tensor::DataLocation::HOST);
    EXPECT_EQ(moved.get_host_ptr(), orig_h_ptr);  // 메모리 포인터도 유지

    // 원본은 유효하지만 비어있는 상태가 되어야 함
    EXPECT_EQ(original.get_data_location(), ushionn::Tensor::DataLocation::NONE);
    EXPECT_EQ(original.get_host_ptr(), nullptr);
    EXPECT_EQ(original.get_device_ptr(), nullptr);
    EXPECT_TRUE(original.get_shape().empty());
}

// --- 가상 텐서 테스트 ---
TEST_F(TensorTest, VirtualTensor)
{
    ushionn::Tensor vt({1, 10}, cudnn_frontend::DataType_t::FLOAT, true, "virtual_t");
    EXPECT_TRUE(vt.is_virtual());
    EXPECT_EQ(vt.get_data_location(), ushionn::Tensor::DataLocation::NONE);
    // 가상 텐서는 메모리 할당 시도 시 경고 또는 무시되어야 함
    // EXPECT_NO_THROW(vt.allocate_host_memory()); // 경고만 출력하고 아무것도 안 함
    // EXPECT_NO_THROW(vt.allocate_device_memory());
    vt.allocate_host_memory();  // 내부에서 USHIONN_WARN으로 처리
    vt.allocate_device_memory();

    EXPECT_EQ(vt.get_host_ptr(), nullptr);
    EXPECT_EQ(vt.get_device_ptr(), nullptr);

    // to_device, to_host도 무시되어야 함
    // EXPECT_NO_THROW(vt.to_device());
    // EXPECT_NO_THROW(vt.to_host());
    vt.to_device();
    vt.to_host();
    EXPECT_EQ(vt.get_data_location(), ushionn::Tensor::DataLocation::NONE);
}

// --- 에러 상황 테스트 (선택적) ---
// USHIONN_ASSERT 나 USHIONN_LOG_FATAL 이 std::runtime_error 를 던진다고 가정
TEST_F(TensorTest, AccessNullData)
{
    ushionn::Tensor t({2, 2});  // DataLocation::NONE
    EXPECT_THROW(t.get_host_ptr(), std::runtime_error);
    EXPECT_THROW(t.get_device_ptr(), std::runtime_error);
    EXPECT_THROW(t.to_device(), std::runtime_error);  // Host에 데이터가 없으므로 Device로 옮길 수 없음
}

TEST_F(TensorTest, MutableAccessWrongLocation)
{
    ushionn::Tensor t({2, 2});
    t.allocate_device_memory();                                  // 현재 DEVICE 상태
    EXPECT_THROW(t.get_mutable_host_ptr(), std::runtime_error);  // Host에 없으므로 에러

    ushionn::Tensor t2({2, 2});
    t2.allocate_host_memory();                                      // 현재 HOST 상태
    EXPECT_THROW(t2.get_mutable_device_ptr(), std::runtime_error);  // Device에 없으므로 에러
}

// 더 많은 테스트 케이스 추가 가능:
// - 다른 데이터 타입 (HALF, INT8 등) 테스트
// - 비동기 데이터 전송 (스트림 사용) 테스트
// - create_graph_tensor_attributes 함수 테스트 (Graph 객체 mock 필요 가능성)
// - 매우 큰 텐서 할당 및 해제 테스트
// - 소멸자에서 메모리 해제 잘 되는지 (valgrind 등으로 확인)