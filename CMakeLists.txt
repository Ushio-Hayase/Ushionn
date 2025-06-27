cmake_minimum_required(VERSION 3.20)
project(ushoinn_project LANGUAGES CXX CUDA)

set(BUILD_TESTING ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# CUDA 라이브러리 출력
message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}") # cudart 등
message(STATUS "CUDA Include DIRS: ${CUDA_INCLUDE_DIRS}")

add_subdirectory("lib")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(USHIONN_SOURCES
    src/core/tensor.cpp
    src/core/common.cpp
    src/layers/activation.cpp
    src/layers/layers.cpp
    src/model/model.cpp
    src/cuda/kernels.cu
    src/cuda/cuda_utils.cu
    src/cuda/tensor.cu
)

add_library(ushionn STATIC ${USHIONN_SOURCES})

target_include_directories(ushionn PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${CUDA_INCLUDE_DIRS})


include(FetchContent)

FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip)
FetchContent_MakeAvailable(googletest)

target_link_libraries(ushionn
    PUBLIC # 라이브러리를 사용하는 다른 타겟들도 이 라이브러리들을 링크해야 함
    ${CUDA_LIBRARIES} # cudart
    # CUDA::cublas 타겟 사용 (CMake 3.9+). 하위 버전은 ${CUDAToolkit_cublas_LIBRARY}
    $<$<TARGET_EXISTS:CUDA::cublas>:CUDA::cublas>
    $<$<NOT:$<TARGET_EXISTS:CUDA::cublas>>:${CUDAToolkit_cublas_LIBRARY}>
    PRIVATE
    # 라이브러리 내부적으로만 필요한 링크
)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.14.0 # 원하는 버전으로 변경 가능
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- 유닛 테스트 빌드 ---
if(BUILD_TESTING) # CMake 옵션으로 테스트 빌드 여부 제어 가능 (기본 ON)
    enable_testing() # CTest 사용 활성화

    # 테스트 실행 파일 이름
    set(UNIT_TEST_RUNNER_NAME ushionn_tests)

    # 테스트 소스 파일 목록 (tests/ 디렉토리의 파일들을 추가)
    # file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    set(USHIONN_TEST_SOURCES
        tests/main.cpp # 테스트를 실행하는 main 함수 (아래 예시 참고)
        tests/perceptron.cpp
        # ... 기타 테스트 파일 ...
    )

    add_executable(${UNIT_TEST_RUNNER_NAME} ${USHIONN_TEST_SOURCES})

    # cuda 헤더 불러오기
    target_include_directories(${UNIT_TEST_RUNNER_NAME}
        PRIVATE
        ${CUDA_INCLUDE_DIRS}
    )

    target_link_libraries(${UNIT_TEST_RUNNER_NAME}
        PRIVATE
        ushionn # 우리가 빌드한 라이브러리
        gtest_main # Google Test (main 함수 포함) 또는 gtest
        # 만약 gtest_main 대신 gtest를 사용하면, tests/main.cpp 에 직접 GTest 초기화 코드 작성 필요
    )

    # CTest에 테스트 추가
    include(GoogleTest)
    gtest_discover_tests(${UNIT_TEST_RUNNER_NAME})
endif()




message(STATUS "CMake configuration done. You can now build with 'cmake --build .'")

