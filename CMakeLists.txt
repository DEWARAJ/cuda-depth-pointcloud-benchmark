cmake_minimum_required(VERSION 3.22)

project(cuda_depth_pointcloud_benchmark LANGUAGES CXX CUDA)

option(BUILD_ROS2_COMPONENT "Build the ROS 2 composable depth projector" OFF)

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
  add_executable(cpu_unit_tests tests/cpu_tests.cpp)
  target_link_libraries(cpu_unit_tests PRIVATE pointcloud_cpu)
  target_compile_options(cpu_unit_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME cpu_unit_tests COMMAND cpu_unit_tests)
  add_test(NAME cpu_cuda_correctness COMMAND depth_benchmark --verify-only)
endif()

if(BUILD_ROS2_COMPONENT)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(sensor_msgs REQUIRED)

  add_library(cuda_depth_component SHARED ros2/cuda_depth_component.cpp)
  target_include_directories(cuda_depth_component PRIVATE include)
  target_link_libraries(cuda_depth_component pointcloud_cuda)
  target_compile_options(cuda_depth_component PRIVATE -Wall -Wextra -Wpedantic -Werror)
  ament_target_dependencies(cuda_depth_component
    rclcpp
    rclcpp_components
    sensor_msgs
  )
  rclcpp_components_register_nodes(cuda_depth_component
    "pointcloud::ros2::CudaDepthComponent"
  )

  add_executable(synthetic_depth_publisher ros2/synthetic_depth_publisher.cpp)
  target_compile_options(synthetic_depth_publisher PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  ament_target_dependencies(synthetic_depth_publisher rclcpp sensor_msgs)

  install(
    TARGETS cuda_depth_component synthetic_depth_publisher
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  ament_package()
endif()
