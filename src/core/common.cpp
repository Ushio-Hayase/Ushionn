#include "core/common.h"  // src/core/common.cpp

namespace ushionn
{
namespace internal
{

void handleErrorInternal(const char* file, int line, const char* func_name, const std::string& error_message,
                         bool is_fatal)
{
    std::ostringstream oss;
    oss << "[" << (is_fatal ? "FATAL ERROR" : "ERROR") << "] " << error_message << std::endl;
    oss << "  Location: " << file << ":" << line << " (" << func_name << ")";

    std::cerr << oss.str() << std::endl;

    if (is_fatal)
    {
        throw std::runtime_error(oss.str());
        // 또는 std::abort();
    }
}

}  // namespace internal

namespace utils
{

std::string formatBytes(size_t bytes)
{
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double count = static_cast<double>(bytes);
    while (count >= 1024 && suffix_idx < 4)
    {
        count /= 1024;
        suffix_idx++;
    }
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << count << " " << suffixes[suffix_idx];
    return oss.str();
}

}  // namespace utils
}  // namespace ushionn