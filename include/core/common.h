#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// 디버그 모드 정의
#ifndef NDEBUG
#define ushionn_DEBUG_MODE 1
#else
#define ushionn_DEBUG_MODE 0
#endif

namespace ushionn
{
namespace internal
{  // 내부 구현용 네임스페이스

// 에러 메시지 포맷팅 및 처리 (구현은 common.cpp에)
void handleErrorInternal(const char* file, int line, const char* func_name, const std::string& error_message,
                         bool is_fatal = true);

}  // namespace internal

// --- CPU 전용 매크로 및 유틸리티 ---

// 일반 조건 체크 매크로
#define USHIONN_ASSERT(condition, error_message_str)                                                            \
    do                                                                                                          \
    {                                                                                                           \
        if (!(condition))                                                                                       \
        {                                                                                                       \
            std::ostringstream error_oss_assert;                                                                \
            error_oss_assert << "Assertion failed: (" << #condition << "). Message: " << (error_message_str);   \
            ushionn::internal::handleErrorInternal(__FILE__, __LINE__, __func__, error_oss_assert.str(), true); \
        }                                                                                                       \
    } while (0)

// 경고 메시지 출력 매크로
#define USHIONN_WARN(warning_message_str)                                                                        \
    do                                                                                                           \
    {                                                                                                            \
        if (ushionn_DEBUG_MODE)                                                                                  \
        {                                                                                                        \
            std::ostringstream warning_oss_warn;                                                                 \
            warning_oss_warn << "[WARNING] " << (warning_message_str);                                           \
            ushionn::internal::handleErrorInternal(__FILE__, __LINE__, __func__, warning_oss_warn.str(), false); \
        }                                                                                                        \
    } while (0)

// 에러 로깅 및 즉시 종료 (복구 불가능한 치명적 오류 시)
#define USHIONN_LOG_FATAL(error_message_str)                                                               \
    do                                                                                                     \
    {                                                                                                      \
        std::ostringstream error_oss_fatal;                                                                \
        error_oss_fatal << "Fatal error: " << (error_message_str);                                         \
        ushionn::internal::handleErrorInternal(__FILE__, __LINE__, __func__, error_oss_fatal.str(), true); \
    } while (0)

enum class DataType
{
    FLOAT32,
    FLOAT64,
};

namespace utils
{  // 순수 C++ 유틸리티 함수 (선언)

// 바이트 크기를 읽기 쉬운 문자열로 변환 (구현은 common.cpp에)
std::string formatBytes(size_t bytes);

template <typename T>
DataType primitiveTypeToDataType()
{
    switch (typeid(T))
    {
        case typeid(float):
            return DataType::FLOAT32;
        case typeid(double):
            return DataType::FLOAT64;
        default:
            USHIONN_LOG_FATAL("Unkown type received");
    }
}

}  // namespace utils
}  // namespace ushionn
