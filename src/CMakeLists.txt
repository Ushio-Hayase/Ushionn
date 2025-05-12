add_library(ushionn tensor.cu layers.cu model.cu)

target_include_directories(ushionn
    PRIVATE
    ${CUDNN_INCLUDE_DIR}
    ${CUDNN_FRONTEND_INCLUDE_DIR})

add_executable(tensorTest test.cpp)
target_link_libraries(tensorTest PRIVATE tensor ${CUDNN_LIBRARY} ${CUDA_LIBRARIES} gtest_main gtest)




include(GoogleTest)
gtest_discover_tests(tensorTest)
