include(FetchContent)

FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip)
FetchContent_MakeAvailable(googletest)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE gtest_main gtest tensor)

target_include_directories(test PRIVATE ../tensor)

include(GoogleTest)
gtest_discover_tests(test)
