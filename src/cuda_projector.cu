#include "depth_projector.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace pointcloud {
namespace {

void check_cuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* data_{nullptr};
  std::size_t count_{};
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_{};
};

class CudaStream {
 public:
  CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  ~CudaStream() { cudaStreamDestroy(stream_); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{};
};

template <typename T>
class PinnedBuffer {
 public:
  explicit PinnedBuffer(const std::size_t count) : count_(count) {
    check_cuda(cudaMallocHost(&data_, count_ * sizeof(T)), "cudaMallocHost");
  }
  ~PinnedBuffer() { cudaFreeHost(data_); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

 private:
  T* data_{nullptr};
  std::size_t count_{};
};

__global__ void project_depth_kernel(const float* depth,
                                     PointXYZ* points,
                                     const CameraIntrinsics intrinsics,
                                     const int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const int row = index / intrinsics.width;
  const int column = index - row * intrinsics.width;
  const float z = depth[index];
  if (!isfinite(z) || z <= 0.0F) {
    const float invalid = nanf("");
    points[index] = {invalid, invalid, invalid};
    return;
  }
  points[index] = {
      (static_cast<float>(column) - intrinsics.cx) * z / intrinsics.fx,
      (static_cast<float>(row) - intrinsics.cy) * z / intrinsics.fy,
      z,
  };
}

__global__ void mark_voxel_owners_kernel(const PointXYZ* points,
                                         const int point_count,
                                         int* voxel_owners,
                                         const VoxelGridConfig config) {
  const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= point_count) {
    return;
  }

  const PointXYZ point = points[point_index];
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
      point.z <= 0.0F) {
    return;
  }
  const int x = static_cast<int>(floorf((point.x - config.min_x) /
                                        config.voxel_size));
  const int y = static_cast<int>(floorf((point.y - config.min_y) /
                                        config.voxel_size));
  const int z = static_cast<int>(floorf((point.z - config.min_z) /
                                        config.voxel_size));
  if (x < 0 || y < 0 || z < 0 || x >= config.dim_x || y >= config.dim_y ||
      z >= config.dim_z) {
    return;
  }

  const int voxel_index = (z * config.dim_y + y) * config.dim_x + x;
  atomicCAS(&voxel_owners[voxel_index], -1, point_index);
}

__global__ void compact_voxel_owners_kernel(const int* voxel_owners,
                                             const int voxel_count,
                                             const PointXYZ* points,
                                             PointXYZ* compacted_points,
                                             int* output_count) {
  const int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_index >= voxel_count) {
    return;
  }

  const int point_index = voxel_owners[voxel_index];
  if (point_index >= 0) {
    const int output_index = atomicAdd(output_count, 1);
    compacted_points[output_index] = points[point_index];
  }
}

void launch_kernel(const DeviceBuffer<float>& depth,
                   DeviceBuffer<PointXYZ>& points,
                   const CameraIntrinsics& intrinsics,
                   const int count,
                   const cudaStream_t stream = nullptr) {
  constexpr int threads_per_block = 256;
  const int blocks = (count + threads_per_block - 1) / threads_per_block;
  project_depth_kernel<<<blocks, threads_per_block, 0, stream>>>(
      depth.data(), points.data(), intrinsics, count);
  check_cuda(cudaGetLastError(), "project_depth_kernel launch");
}

void launch_voxel_kernels(const DeviceBuffer<PointXYZ>& points,
                          DeviceBuffer<int>& voxel_owners,
                          DeviceBuffer<PointXYZ>& compacted_points,
                          DeviceBuffer<int>& output_count,
                          const VoxelGridConfig& config,
                          const int point_count,
                          const int voxel_count,
                          const cudaStream_t stream) {
  constexpr int threads_per_block = 256;
  const int point_blocks =
      (point_count + threads_per_block - 1) / threads_per_block;
  const int voxel_blocks =
      (voxel_count + threads_per_block - 1) / threads_per_block;

  mark_voxel_owners_kernel<<<point_blocks, threads_per_block, 0, stream>>>(
      points.data(), point_count, voxel_owners.data(), config);
  check_cuda(cudaGetLastError(), "mark_voxel_owners_kernel launch");

  compact_voxel_owners_kernel<<<voxel_blocks, threads_per_block, 0, stream>>>(
      voxel_owners.data(), voxel_count, points.data(), compacted_points.data(),
      output_count.data());
  check_cuda(cudaGetLastError(), "compact_voxel_owners_kernel launch");
}

void validate_voxel_config(const VoxelGridConfig& config) {
  if (config.voxel_size <= 0.0F || config.dim_x <= 0 || config.dim_y <= 0 ||
      config.dim_z <= 0 ||
      config.voxel_count() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("Voxel grid configuration is invalid or too large");
  }
}

}  // namespace

CudaTimings benchmark_depth_cuda(const std::vector<float>& depth,
                                 const CameraIntrinsics& intrinsics,
                                 std::vector<PointXYZ>& points,
                                 const int warmup_iterations,
                                 const int measured_iterations) {
  const int count = intrinsics.width * intrinsics.height;
  if (count <= 0 || depth.size() != static_cast<std::size_t>(count) ||
      points.size() != static_cast<std::size_t>(count)) {
    throw std::invalid_argument("Depth and point buffers must match image dimensions");
  }
  if (warmup_iterations < 0 || measured_iterations <= 0) {
    throw std::invalid_argument("Iteration counts are invalid");
  }

  DeviceBuffer<float> device_depth(depth.size());
  DeviceBuffer<PointXYZ> device_points(points.size());

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    check_cuda(cudaMemcpy(device_depth.data(), depth.data(), device_depth.bytes(),
                          cudaMemcpyHostToDevice),
               "warmup host-to-device copy");
    launch_kernel(device_depth, device_points, intrinsics, count);
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");
  }

  CudaEvent start_event;
  CudaEvent stop_event;
  CudaTimings timings;
  timings.kernel_ms.reserve(measured_iterations);
  timings.end_to_end_ms.reserve(measured_iterations);

  for (int iteration = 0; iteration < measured_iterations; ++iteration) {
    const auto total_start = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(device_depth.data(), depth.data(), device_depth.bytes(),
                          cudaMemcpyHostToDevice),
               "host-to-device copy");

    check_cuda(cudaEventRecord(start_event.get()), "start event record");
    launch_kernel(device_depth, device_points, intrinsics, count);
    check_cuda(cudaEventRecord(stop_event.get()), "stop event record");
    check_cuda(cudaEventSynchronize(stop_event.get()), "stop event synchronization");

    float kernel_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event.get(), stop_event.get()),
               "kernel elapsed time");
    check_cuda(cudaMemcpy(points.data(), device_points.data(), device_points.bytes(),
                          cudaMemcpyDeviceToHost),
               "device-to-host copy");
    const auto total_stop = std::chrono::steady_clock::now();

    timings.kernel_ms.push_back(kernel_ms);
    timings.end_to_end_ms.push_back(
        std::chrono::duration<double, std::milli>(total_stop - total_start).count());
  }

  return timings;
}

CudaTimings benchmark_depth_cuda_pinned(const std::vector<float>& depth,
                                        const CameraIntrinsics& intrinsics,
                                        std::vector<PointXYZ>& points,
                                        const int warmup_iterations,
                                        const int measured_iterations) {
  const int count = intrinsics.width * intrinsics.height;
  if (count <= 0 || depth.size() != static_cast<std::size_t>(count) ||
      points.size() != static_cast<std::size_t>(count)) {
    throw std::invalid_argument("Depth and point buffers must match image dimensions");
  }
  if (warmup_iterations < 0 || measured_iterations <= 0) {
    throw std::invalid_argument("Iteration counts are invalid");
  }

  PinnedBuffer<float> pinned_depth(depth.size());
  PinnedBuffer<PointXYZ> pinned_points(points.size());
  std::copy(depth.begin(), depth.end(), pinned_depth.data());

  DeviceBuffer<float> device_depth(depth.size());
  DeviceBuffer<PointXYZ> device_points(points.size());
  CudaStream stream;

  const auto run_once = [&] {
    check_cuda(cudaMemcpyAsync(device_depth.data(), pinned_depth.data(),
                               device_depth.bytes(), cudaMemcpyHostToDevice,
                               stream.get()),
               "pinned host-to-device copy");
    launch_kernel(device_depth, device_points, intrinsics, count, stream.get());
    check_cuda(cudaMemcpyAsync(pinned_points.data(), device_points.data(),
                               device_points.bytes(), cudaMemcpyDeviceToHost,
                               stream.get()),
               "pinned device-to-host copy");
  };

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    run_once();
    check_cuda(cudaStreamSynchronize(stream.get()), "pinned warmup synchronization");
  }

  CudaEvent start_event;
  CudaEvent stop_event;
  CudaTimings timings;
  timings.kernel_ms.reserve(measured_iterations);
  timings.end_to_end_ms.reserve(measured_iterations);

  for (int iteration = 0; iteration < measured_iterations; ++iteration) {
    const auto total_start = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpyAsync(device_depth.data(), pinned_depth.data(),
                               device_depth.bytes(), cudaMemcpyHostToDevice,
                               stream.get()),
               "timed pinned host-to-device copy");
    check_cuda(cudaEventRecord(start_event.get(), stream.get()),
               "pinned start event record");
    launch_kernel(device_depth, device_points, intrinsics, count, stream.get());
    check_cuda(cudaEventRecord(stop_event.get(), stream.get()),
               "pinned stop event record");
    check_cuda(cudaMemcpyAsync(pinned_points.data(), device_points.data(),
                               device_points.bytes(), cudaMemcpyDeviceToHost,
                               stream.get()),
               "timed pinned device-to-host copy");
    check_cuda(cudaStreamSynchronize(stream.get()),
               "pinned measured synchronization");
    const auto total_stop = std::chrono::steady_clock::now();

    float kernel_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event.get(), stop_event.get()),
               "pinned kernel elapsed time");
    timings.kernel_ms.push_back(kernel_ms);
    timings.end_to_end_ms.push_back(
        std::chrono::duration<double, std::milli>(total_stop - total_start).count());
  }

  std::copy(pinned_points.data(), pinned_points.data() + points.size(), points.begin());
  return timings;
}

VoxelPipelineTimings benchmark_depth_voxel_cuda(
    const std::vector<float>& depth,
    const CameraIntrinsics& intrinsics,
    const VoxelGridConfig& config,
    const int warmup_iterations,
    const int measured_iterations) {
  const int point_count = intrinsics.width * intrinsics.height;
  if (point_count <= 0 || depth.size() != static_cast<std::size_t>(point_count)) {
    throw std::invalid_argument("Depth buffer must match image dimensions");
  }
  if (warmup_iterations < 0 || measured_iterations <= 0) {
    throw std::invalid_argument("Iteration counts are invalid");
  }
  validate_voxel_config(config);
  const int voxel_count = static_cast<int>(config.voxel_count());

  PinnedBuffer<float> pinned_depth(depth.size());
  PinnedBuffer<int> pinned_output_count(1);
  std::copy(depth.begin(), depth.end(), pinned_depth.data());

  DeviceBuffer<float> device_depth(depth.size());
  DeviceBuffer<PointXYZ> device_points(depth.size());
  DeviceBuffer<int> voxel_owners(config.voxel_count());
  DeviceBuffer<PointXYZ> compacted_points(depth.size());
  DeviceBuffer<int> device_output_count(1);
  CudaStream stream;

  const auto run_pipeline = [&](const cudaEvent_t start_event,
                                const cudaEvent_t stop_event) {
    check_cuda(cudaMemcpyAsync(device_depth.data(), pinned_depth.data(),
                               device_depth.bytes(), cudaMemcpyHostToDevice,
                               stream.get()),
               "voxel pipeline host-to-device copy");
    if (start_event != nullptr) {
      check_cuda(cudaEventRecord(start_event, stream.get()),
                 "voxel pipeline start event record");
    }
    check_cuda(cudaMemsetAsync(voxel_owners.data(), 0xFF, voxel_owners.bytes(),
                               stream.get()),
               "voxel owner reset");
    check_cuda(cudaMemsetAsync(device_output_count.data(), 0,
                               device_output_count.bytes(), stream.get()),
               "voxel output count reset");
    launch_kernel(device_depth, device_points, intrinsics, point_count,
                  stream.get());
    launch_voxel_kernels(device_points, voxel_owners, compacted_points,
                         device_output_count, config, point_count, voxel_count,
                         stream.get());
    if (stop_event != nullptr) {
      check_cuda(cudaEventRecord(stop_event, stream.get()),
                 "voxel pipeline stop event record");
    }
    check_cuda(cudaMemcpyAsync(pinned_output_count.data(),
                               device_output_count.data(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream.get()),
               "voxel pipeline count copy");
  };

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    run_pipeline(nullptr, nullptr);
    check_cuda(cudaStreamSynchronize(stream.get()),
               "voxel pipeline warmup synchronization");
  }

  CudaEvent start_event;
  CudaEvent stop_event;
  VoxelPipelineTimings timings;
  timings.gpu_ms.reserve(measured_iterations);
  timings.end_to_end_ms.reserve(measured_iterations);

  for (int iteration = 0; iteration < measured_iterations; ++iteration) {
    const auto total_start = std::chrono::steady_clock::now();
    run_pipeline(start_event.get(), stop_event.get());
    check_cuda(cudaStreamSynchronize(stream.get()),
               "voxel pipeline measured synchronization");
    const auto total_stop = std::chrono::steady_clock::now();

    float gpu_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&gpu_ms, start_event.get(), stop_event.get()),
               "voxel pipeline GPU elapsed time");
    timings.gpu_ms.push_back(gpu_ms);
    timings.end_to_end_ms.push_back(
        std::chrono::duration<double, std::milli>(total_stop - total_start).count());
    timings.output_count = *pinned_output_count.data();
  }

  return timings;
}

std::string cuda_device_name() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  return properties.name;
}

}  // namespace pointcloud
