#include "depth_projector.hpp"

#include <cstddef>
#include <stdexcept>

namespace pointcloud {

void project_depth_cpu(const std::vector<float>& depth,
                       const CameraIntrinsics& intrinsics,
                       std::vector<PointXYZ>& points) {
  const std::size_t expected =
      static_cast<std::size_t>(intrinsics.width) * intrinsics.height;
  if (depth.size() != expected || points.size() != expected) {
    throw std::invalid_argument("Depth and point buffers must match image dimensions");
  }

  for (int row = 0; row < intrinsics.height; ++row) {
    for (int column = 0; column < intrinsics.width; ++column) {
      const std::size_t index =
          static_cast<std::size_t>(row) * intrinsics.width + column;
      const float z = depth[index];
      points[index] = {
          (static_cast<float>(column) - intrinsics.cx) * z / intrinsics.fx,
          (static_cast<float>(row) - intrinsics.cy) * z / intrinsics.fy,
          z,
      };
    }
  }
}

}  // namespace pointcloud
