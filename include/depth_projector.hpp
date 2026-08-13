#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pointcloud {

struct CameraIntrinsics {
  float fx{};
  float fy{};
  float cx{};
  float cy{};
  int width{};
  int height{};
};

struct PointXYZ {
  float x{};
  float y{};
  float z{};
};

struct CudaTimings {
  std::vector<double> kernel_ms;
  std::vector<double> end_to_end_ms;
};

struct VoxelGridConfig {
  float min_x{};
  float min_y{};
  float min_z{};
  float voxel_size{};
  int dim_x{};
  int dim_y{};
  int dim_z{};

  [[nodiscard]] std::size_t voxel_count() const noexcept {
    return static_cast<std::size_t>(dim_x) * dim_y * dim_z;
  }
};

struct VoxelPipelineTimings {
  std::vector<double> gpu_ms;
  std::vector<double> end_to_end_ms;
  int output_count{};
};

void project_depth_cpu(const std::vector<float>& depth,
                       const CameraIntrinsics& intrinsics,
                       std::vector<PointXYZ>& points);

CudaTimings benchmark_depth_cuda(const std::vector<float>& depth,
                                 const CameraIntrinsics& intrinsics,
                                 std::vector<PointXYZ>& points,
                                 int warmup_iterations,
                                 int measured_iterations);

CudaTimings benchmark_depth_cuda_pinned(const std::vector<float>& depth,
                                        const CameraIntrinsics& intrinsics,
                                        std::vector<PointXYZ>& points,
                                        int warmup_iterations,
                                        int measured_iterations);

std::size_t count_occupied_voxels_cpu(const std::vector<PointXYZ>& points,
                                      const VoxelGridConfig& config);

VoxelPipelineTimings benchmark_depth_voxel_cuda(
    const std::vector<float>& depth,
    const CameraIntrinsics& intrinsics,
    const VoxelGridConfig& config,
    int warmup_iterations,
    int measured_iterations);

std::string cuda_device_name();

}  // namespace pointcloud
