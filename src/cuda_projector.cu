#include "depth_projector.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstddef>
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
  points[index] = {
      (static_cast<float>(column) - intrinsics.cx) * z / intrinsics.fx,
      (static_cast<float>(row) - intrinsics.cy) * z / intrinsics.fy,
      z,
  };
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

std::string cuda_device_name() {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  return properties.name;
}

}  // namespace pointcloud
