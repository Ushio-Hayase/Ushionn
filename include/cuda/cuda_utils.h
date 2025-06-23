#pragma once

#include <cublas_v2.h>

#include <string>  // for std::string

#include "core/error_codes.h"  // cudaError_t, cudnnStatus_t 타입 정의

// 이 헤더는 CUDA API 함수를 직접 호출하지 않으므로,
// 일반 C++ 컴파일러도 처리 가능 (선언만 있기 때문)

namespace ushionn
{
namespace cuda
{
namespace internal
{  // CUDA 관련 내부 구현용 네임스페이스 (구현은 cuda_utils.cu에)

// CUDA API 에러 처리 함수 (선언)
void handleCudaError(cudaError_t err_code, const char* file, int line, const char* func_name);

// CUDA 커널 실행 후 에러 체크 함수 (선언)
void checkCudaKernelError(const char* message_prefix, const char* file, int line, const char* func_name);

// GPU 메모리 사용량 출력 함수 (선언)
void printGpuMemoryUsageImpl(const std::string& tag);

}  // namespace internal

// --- CUDA/cuDNN API 에러 체크 매크로 ---
// 이 매크로들은 위에 선언된 래퍼 함수들을 호출합니다.

#define CUDA_CHECK(cuda_call_result) \
    ushionn::cuda::internal::handleCudaError((cuda_call_result), __FILE__, __LINE__, __func__)

#define CUDNN_CHECK(cublas_call_result) \
    ushionn::cuda::internal::handleCudnnError((cublas_call_result), __FILE__, __LINE__, __func__)

#define USHIONN_KERNEL_CHECK_ERROR(message_prefix_str) \
    ushionn::cuda::internal::checkCudaKernelError((message_prefix_str), __FILE__, __LINE__, __func__)

#define IDX2F(i, j, ld) ((((j) - 1) * (ld)) + ((i) - 1))

// --- CUDA 관련 유틸리티 함수 (공개 인터페이스) ---
namespace utils
{

// GPU 메모리 사용량 출력 (구현은 cuda_utils.cu에)
inline void printGpuMemoryUsage(const std::string& tag = "")
{
#if USHIONN_DEBUG_MODE  // common.h에 정의된 ushionn_DEBUG_MODE 사용
    ushionn::cuda::internal::printGpuMemoryUsageImpl(tag);
#else
    (void)tag;  // unused parameter 경고 방지
#endif
}

}  // namespace utils
}  // namespace cuda
}  // namespace ushionn