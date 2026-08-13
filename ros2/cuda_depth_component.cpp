#include "depth_projector.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pointcloud::ros2 {

class CudaDepthComponent final : public rclcpp::Node {
 public:
  explicit CudaDepthComponent(const rclcpp::NodeOptions& options)
      : Node("cuda_depth_component", options),
        fx_(declare_parameter<double>("fx", 0.0)),
        fy_(declare_parameter<double>("fy", 0.0)),
        cx_(declare_parameter<double>("cx", -1.0)),
        cy_(declare_parameter<double>("cy", -1.0)),
        publisher_(create_publisher<sensor_msgs::msg::PointCloud2>(
            "points", rclcpp::SensorDataQoS())) {
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
        "depth", rclcpp::SensorDataQoS(),
        std::bind(&CudaDepthComponent::on_depth, this, std::placeholders::_1));
  }

 private:
  static_assert(sizeof(PointXYZ) == 3U * sizeof(float));

  void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr message) {
    try {
      if (message->encoding != "32FC1") {
        throw std::invalid_argument("depth image encoding must be 32FC1");
      }
      if (message->is_bigendian != 0U) {
        throw std::invalid_argument("big-endian depth images are not supported");
      }

      const auto width = static_cast<int>(message->width);
      const auto height = static_cast<int>(message->height);
      const std::size_t row_bytes = static_cast<std::size_t>(width) * sizeof(float);
      const std::size_t point_count = static_cast<std::size_t>(width) * height;
      if (width <= 0 || height <= 0 || message->step < row_bytes ||
          message->data.size() < static_cast<std::size_t>(message->step) * height) {
        throw std::invalid_argument("depth image dimensions or row stride are invalid");
      }

      std::vector<float> depth(point_count);
      for (int row = 0; row < height; ++row) {
        std::memcpy(depth.data() + static_cast<std::size_t>(row) * width,
                    message->data.data() + static_cast<std::size_t>(row) * message->step,
                    row_bytes);
      }

      const CameraIntrinsics intrinsics{
          static_cast<float>(fx_ > 0.0 ? fx_ : width),
          static_cast<float>(fy_ > 0.0 ? fy_ : width),
          static_cast<float>(cx_ >= 0.0 ? cx_ : (width - 1) * 0.5),
          static_cast<float>(cy_ >= 0.0 ? cy_ : (height - 1) * 0.5),
          width,
          height,
      };
      std::vector<PointXYZ> points(point_count);
      const auto timings = benchmark_depth_cuda_pinned(depth, intrinsics, points, 0, 1);

      sensor_msgs::msg::PointCloud2 output;
      output.header = message->header;
      output.height = message->height;
      output.width = message->width;
      output.is_bigendian = false;
      output.is_dense = false;
      output.fields = {
          make_field("x", 0),
          make_field("y", sizeof(float)),
          make_field("z", 2U * sizeof(float)),
      };
      output.point_step = sizeof(PointXYZ);
      output.row_step = output.point_step * output.width;
      output.data.resize(points.size() * sizeof(PointXYZ));
      std::memcpy(output.data.data(), points.data(), output.data.size());
      publisher_->publish(output);

      RCLCPP_DEBUG(get_logger(), "Projected %zu points in %.3f ms", point_count,
                   timings.end_to_end_ms.front());
    } catch (const std::exception& error) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.what());
    }
  }

  static sensor_msgs::msg::PointField make_field(const std::string& name,
                                                  const std::size_t offset) {
    sensor_msgs::msg::PointField field;
    field.name = name;
    field.offset = static_cast<std::uint32_t>(offset);
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    return field;
  }

  double fx_;
  double fy_;
  double cx_;
  double cy_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

}  // namespace pointcloud::ros2

RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud::ros2::CudaDepthComponent)
