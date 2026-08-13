#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source /opt/ros/humble/setup.bash
source ros_install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-72}"
mkdir -p results

manager_pid=""
publisher_pid=""
cleanup() {
  [[ -z "${publisher_pid}" ]] || kill "${publisher_pid}" 2>/dev/null || true
  [[ -z "${manager_pid}" ]] || kill "${manager_pid}" 2>/dev/null || true
}
trap cleanup EXIT

ros2 run rclcpp_components component_container \
  --ros-args -r __node:=cuda_component_manager \
  >results/ros2_component_manager.log 2>&1 &
manager_pid=$!
sleep 2

ros2 component load /cuda_component_manager cuda_depth_pointcloud_benchmark \
  pointcloud::ros2::CudaDepthComponent \
  >results/ros2_component_load.txt

ros2 run cuda_depth_pointcloud_benchmark synthetic_depth_publisher \
  >results/ros2_synthetic_publisher.log 2>&1 &
publisher_pid=$!
sleep 2

if ! timeout 15 ros2 topic echo /points --once --field width \
  >results/ros2_smoke_test.txt 2>&1; then
  echo "ROS 2 smoke test failed; inspect results/ros2_*.log" >&2
  exit 1
fi

echo "ROS 2 smoke test passed: depth images produced PointCloud2 messages"
cat results/ros2_smoke_test.txt
