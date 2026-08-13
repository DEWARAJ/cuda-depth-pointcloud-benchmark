#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>

using namespace std::chrono_literals;

class SyntheticDepthPublisher final : public rclcpp::Node {
 public:
  SyntheticDepthPublisher()
      : Node("synthetic_depth_publisher"),
        width_(declare_parameter<int>("width", 640)),
        height_(declare_parameter<int>("height", 480)),
        publisher_(create_publisher<sensor_msgs::msg::Image>(
            "depth", rclcpp::SensorDataQoS())) {
    timer_ = create_wall_timer(500ms, [this] { publish_depth(); });
  }

 private:
  void publish_depth() {
    sensor_msgs::msg::Image message;
    message.header.stamp = now();
    message.header.frame_id = "synthetic_camera";
    message.height = static_cast<std::uint32_t>(height_);
    message.width = static_cast<std::uint32_t>(width_);
    message.encoding = "32FC1";
    message.is_bigendian = false;
    message.step = static_cast<std::uint32_t>(width_ * sizeof(float));
    message.data.resize(static_cast<std::size_t>(message.step) * height_);

    for (int row = 0; row < height_; ++row) {
      for (int column = 0; column < width_; ++column) {
        const float depth = 2.0F + 0.1F * std::sin(column * 0.02F) +
                            0.05F * std::cos(row * 0.03F);
        const std::size_t offset =
            (static_cast<std::size_t>(row) * width_ + column) * sizeof(float);
        std::memcpy(message.data.data() + offset, &depth, sizeof(depth));
      }
    }
    publisher_->publish(message);
  }

  int width_;
  int height_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SyntheticDepthPublisher>());
  rclcpp::shutdown();
  return 0;
}
