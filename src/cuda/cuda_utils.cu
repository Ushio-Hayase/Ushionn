// src/cuda/cuda_utils.cu
#include "core/common.h"      // ushionn::internal::handleErrorInternal 사용 위함
#include "cuda/cuda_utils.h"  // 선언부

// 실제 CUDA/cuDNN 헤더는 여기서만 include
#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>  // 여기서 직접 std::cerr 사용 안함 (handleErrorInternal이 처리)
#include <sstream>   // 여기서 직접 std::ostringstream 사용 안함 (handleErrorInternal이 처리)

namespace ushionn
{
namespace cuda
{
namespace internal
{

void handleCudaError(cudaError_t err_code, const char* file, int line, const char* func_name)
{
    if (err_code != cudaSuccess)
    {
        std::ostringstream error_oss;
        error_oss << "CUDA API error: " << cudaGetErrorName(err_code) << " (" << cudaGetErrorString(err_code) << ")";
        // common.h 에 정의된 순수 C++ 에러 핸들러 호출
        ushionn::internal::handleErrorInternal(file, line, func_name, error_oss.str(), true);
    }
}

void handleCudnnError(cudnnStatus_t status_code, const char* file, int line, const char* func_name)
{
    if (status_code != CUDNN_STATUS_SUCCESS)
    {
        std::ostringstream error_oss;
        error_oss << "cuDNN API error: " << cudnnGetErrorString(status_code);
        ushionn::internal::handleErrorInternal(file, line, func_name, error_oss.str(), true);
    }
}

void checkCudaKernelError(const char* message_prefix, const char* file, int line, const char* func_name)
{
    // 커널 실행은 비동기이므로, 이전 모든 작업 완료를 기다리거나,
    // cudaGetLastError()를 사용하여 마지막 비동기 에러를 가져옴.
    // cudaDeviceSynchronize(); // 필요시 활성화 (성능 영향 고려)
    cudaError_t err_code = cudaGetLastError();
    if (err_code != cudaSuccess)
    {
        std::ostringstream error_oss;
        error_oss << "CUDA Kernel Execution error (" << message_prefix << "): " << cudaGetErrorName(err_code) << " ("
                  << cudaGetErrorString(err_code) << ")";
        ushionn::internal::handleErrorInternal(file, line, func_name, error_oss.str(), true);
    }
}

void printGpuMemoryUsageImpl(const std::string& tag)
{
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);  // CUDA API 직접 호출
    if (err == cudaSuccess)
    {
        // common.h의 formatBytes 사용 (utils 네임스페이스 명시)
        std::cout << "[" << (tag.empty() ? "GPU Memory" : tag) << "] "
                  << "Free: " << ushionn::utils::formatBytes(free_bytes)
                  << " / Total: " << ushionn::utils::formatBytes(total_bytes)
                  << " (Used: " << ushionn::utils::formatBytes(total_bytes - free_bytes) << ")" << std::endl;
    }
    else
    {
        std::cerr << "[" << (tag.empty() ? "GPU Memory" : tag) << "] "
                  << "Failed to get GPU memory info: " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace internal
}  // namespace cuda
}  // namespace ushionn