add_library(tensor tensor.cu)

add_executable(tensorTest test.cpp)
target_link_libraries(tensorTest PRIVATE gtest_main gtest tensor)

include(GoogleTest)
gtest_discover_tests(tensorTest)
