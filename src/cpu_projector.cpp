#include "depth_projector.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

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
      if (!std::isfinite(z) || z <= 0.0F) {
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        points[index] = {invalid, invalid, invalid};
        continue;
      }
      points[index] = {
          (static_cast<float>(column) - intrinsics.cx) * z / intrinsics.fx,
          (static_cast<float>(row) - intrinsics.cy) * z / intrinsics.fy,
          z,
      };
    }
  }
}

std::size_t count_occupied_voxels_cpu(const std::vector<PointXYZ>& points,
                                      const VoxelGridConfig& config) {
  if (config.voxel_size <= 0.0F || config.dim_x <= 0 || config.dim_y <= 0 ||
      config.dim_z <= 0) {
    throw std::invalid_argument("Voxel grid dimensions and size must be positive");
  }

  std::vector<unsigned char> occupied(config.voxel_count(), 0);
  std::size_t occupied_count = 0;
  for (const PointXYZ& point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        !std::isfinite(point.z) || point.z <= 0.0F) {
      continue;
    }
    const int x = static_cast<int>(std::floor((point.x - config.min_x) /
                                              config.voxel_size));
    const int y = static_cast<int>(std::floor((point.y - config.min_y) /
                                              config.voxel_size));
    const int z = static_cast<int>(std::floor((point.z - config.min_z) /
                                              config.voxel_size));
    if (x < 0 || y < 0 || z < 0 || x >= config.dim_x || y >= config.dim_y ||
        z >= config.dim_z) {
      continue;
    }

    const std::size_t index =
        (static_cast<std::size_t>(z) * config.dim_y + y) * config.dim_x + x;
    if (occupied[index] == 0U) {
      occupied[index] = 1U;
      ++occupied_count;
    }
  }
  return occupied_count;
}

}  // namespace pointcloud
