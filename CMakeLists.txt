cmake_minimum_required(VERSION 3.22)

project(cuda_depth_pointcloud_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(pointcloud_cpu src/cpu_projector.cpp)
target_include_directories(pointcloud_cpu PUBLIC include)
target_compile_options(pointcloud_cpu PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(pointcloud_cuda src/cuda_projector.cu)
target_include_directories(pointcloud_cuda PUBLIC include)
target_link_libraries(pointcloud_cuda PUBLIC CUDA::cudart)
set_target_properties(pointcloud_cuda PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(pointcloud_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(depth_benchmark src/main.cpp)
target_link_libraries(depth_benchmark PRIVATE pointcloud_cpu pointcloud_cuda)
target_compile_options(depth_benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cpu_cuda_correctness COMMAND depth_benchmark --verify-only)
endif()
