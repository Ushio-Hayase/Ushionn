add_library(ushionn tensor.cu layers.cu activation.cu model.cu)
target_include_directories(ushionn
    PRIVATE
    ${CUDNN_INCLUDE_DIR}
    ${CUDNN_FRONTEND_INCLUDE_DIR})

add_executable(tensorTest tensor_test.cpp)

if(USE_CUDNN)
    target_link_libraries(tensorTest PRIVATE ushionn ${CUDNN_LIBRARY} ${CUDA_LIBRARIES} gtest_main gtest)
else()
    target_link_libraries(tensorTest PRIVATE ${CUDA_LIBRARIES} gtest_main gtest ushionn)
endif()

include(GoogleTest)
gtest_discover_tests(tensorTest)
