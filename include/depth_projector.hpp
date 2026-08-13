#pragma once

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

std::string cuda_device_name();

}  // namespace pointcloud
