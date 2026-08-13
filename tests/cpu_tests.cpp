#include "depth_projector.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_near(const float actual,
                  const float expected,
                  const float tolerance,
                  const std::string& message) {
  require(std::abs(actual - expected) <= tolerance, message);
}

void test_projection_geometry() {
  const pointcloud::CameraIntrinsics intrinsics{1.0F, 1.0F, 0.0F, 0.0F, 2, 2};
  const std::vector<float> depth{1.0F, 1.0F, 1.0F, 1.0F};
  std::vector<pointcloud::PointXYZ> points(4);
  pointcloud::project_depth_cpu(depth, intrinsics, points);

  require_near(points[0].x, 0.0F, 1.0e-6F, "Top-left x is incorrect");
  require_near(points[0].y, 0.0F, 1.0e-6F, "Top-left y is incorrect");
  require_near(points[1].x, 1.0F, 1.0e-6F, "Top-right x is incorrect");
  require_near(points[2].y, 1.0F, 1.0e-6F, "Bottom-left y is incorrect");
  require_near(points[3].z, 1.0F, 1.0e-6F, "Depth must be preserved");
}

void test_projection_rejects_mismatched_buffers() {
  const pointcloud::CameraIntrinsics intrinsics{1.0F, 1.0F, 0.0F, 0.0F, 2, 2};
  const std::vector<float> depth(3, 1.0F);
  std::vector<pointcloud::PointXYZ> points(4);
  bool threw = false;
  try {
    pointcloud::project_depth_cpu(depth, intrinsics, points);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "Projection must reject mismatched image buffers");
}

void test_projection_marks_invalid_depth() {
  const pointcloud::CameraIntrinsics intrinsics{1.0F, 1.0F, 0.0F, 0.0F, 4, 1};
  const std::vector<float> depth{
      2.0F,
      0.0F,
      -1.0F,
      std::numeric_limits<float>::quiet_NaN(),
  };
  std::vector<pointcloud::PointXYZ> points(4);
  pointcloud::project_depth_cpu(depth, intrinsics, points);

  require_near(points[0].z, 2.0F, 1.0e-6F, "Positive depth must be projected");
  for (std::size_t index = 1; index < points.size(); ++index) {
    require(std::isnan(points[index].x) && std::isnan(points[index].y) &&
                std::isnan(points[index].z),
            "Non-positive or non-finite depth must produce an invalid point");
  }
}

void test_voxel_count_deduplicates_and_crops() {
  const pointcloud::VoxelGridConfig config{0.0F, 0.0F, 0.0F, 0.1F, 4, 4, 4};
  const std::vector<pointcloud::PointXYZ> points{
      {0.01F, 0.01F, 0.01F},
      {0.09F, 0.09F, 0.09F},
      {0.21F, 0.01F, 0.01F},
      {0.50F, 0.50F, 0.50F},
  };
  require(pointcloud::count_occupied_voxels_cpu(points, config) == 2,
          "Voxel count must deduplicate points and crop out-of-bounds points");
}

void test_voxel_count_rejects_invalid_config() {
  const pointcloud::VoxelGridConfig config{0.0F, 0.0F, 0.0F, 0.0F, 1, 1, 1};
  bool threw = false;
  try {
    static_cast<void>(pointcloud::count_occupied_voxels_cpu({}, config));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "Voxel count must reject a non-positive voxel size");
}

}  // namespace

int main() {
  try {
    test_projection_geometry();
    test_projection_rejects_mismatched_buffers();
    test_projection_marks_invalid_depth();
    test_voxel_count_deduplicates_and_crops();
    test_voxel_count_rejects_invalid_config();
    std::cout << "CPU unit tests: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "CPU unit tests: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
