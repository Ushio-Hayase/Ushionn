#pragma once

// C++ 코드에서도 CUDA/cuDNN 에러 타입을 알 수 있도록 최소한의 include
// 실제 API 함수를 사용하지 않으므로 일반 C++ 컴파일러에서도 문제 없음
#include <cuda_runtime_api.h>  // for cudaError_t

// 필요하다면 여기에 라이브러리 자체 에러 코드 열거형 등을 추가할 수 있습니다.
namespace ushionn
{
enum class UshioNNError
{
    SUCCESS = 0,
    GENERAL_ERROR,
    CUDA_ERROR,
    CUDNN_ERROR,
};
}