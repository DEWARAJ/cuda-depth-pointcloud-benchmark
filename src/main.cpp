#include "depth_projector.hpp"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  int width{1920};
  int height{1080};
  int warmup_iterations{5};
  int measured_iterations{30};
  bool verify_only{false};
};

struct Stats {
  double mean{};
  double p50{};
  double p95{};
  double minimum{};
  double maximum{};
};

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

int parse_integer(const char* value, const std::string_view option) {
  try {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
      throw std::invalid_argument("non-positive");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(option) + " requires a positive integer");
  }
}

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--verify-only") {
      options.verify_only = true;
      options.warmup_iterations = 1;
      options.measured_iterations = 1;
    } else if (argument == "--width" && index + 1 < argc) {
      options.width = parse_integer(argv[++index], argument);
    } else if (argument == "--height" && index + 1 < argc) {
      options.height = parse_integer(argv[++index], argument);
    } else if (argument == "--warmup" && index + 1 < argc) {
      options.warmup_iterations = parse_integer(argv[++index], argument);
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.measured_iterations = parse_integer(argv[++index], argument);
    } else {
      throw std::invalid_argument("Unknown or incomplete option: " + std::string(argument));
    }
  }
  return options;
}

std::vector<float> make_depth_image(const int width, const int height) {
  std::vector<float> depth(static_cast<std::size_t>(width) * height);
  for (int row = 0; row < height; ++row) {
    for (int column = 0; column < width; ++column) {
      const float wave_x = std::sin(static_cast<float>(column) * 0.013F);
      const float wave_y = std::cos(static_cast<float>(row) * 0.017F);
      depth[static_cast<std::size_t>(row) * width + column] =
          2.0F + 0.15F * wave_x + 0.10F * wave_y;
    }
  }
  return depth;
}

Stats summarize(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::invalid_argument("Cannot summarize an empty sample set");
  }
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&samples](const double fraction) {
    const auto position = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(samples.size())) - 1.0);
    return samples[std::min(position, samples.size() - 1)];
  };
  return {
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
      percentile(0.50),
      percentile(0.95),
      samples.front(),
      samples.back(),
  };
}

double maximum_error(const std::span<const pointcloud::PointXYZ> reference,
                     const std::span<const pointcloud::PointXYZ> candidate) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    maximum = std::max({maximum,
                        std::abs(static_cast<double>(reference[index].x) -
                                 candidate[index].x),
                        std::abs(static_cast<double>(reference[index].y) -
                                 candidate[index].y),
                        std::abs(static_cast<double>(reference[index].z) -
                                 candidate[index].z)});
  }
  return maximum;
}

std::vector<double> benchmark_cpu(const std::vector<float>& depth,
                                  const pointcloud::CameraIntrinsics& intrinsics,
                                  std::vector<pointcloud::PointXYZ>& points,
                                  const int warmup_iterations,
                                  const int measured_iterations) {
  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    pointcloud::project_depth_cpu(depth, intrinsics, points);
  }

  std::vector<double> timings;
  timings.reserve(measured_iterations);
  for (int iteration = 0; iteration < measured_iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    pointcloud::project_depth_cpu(depth, intrinsics, points);
    const auto stop = std::chrono::steady_clock::now();
    timings.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return timings;
}

void write_csv(const Stats& cpu,
               const Stats& kernel,
               const Stats& end_to_end,
               const Stats& pinned_kernel,
               const Stats& pinned_end_to_end,
               const Options& options,
               const double max_error) {
  std::filesystem::create_directories("results");
  std::ofstream output("results/latest.csv");
  output << "implementation,width,height,iterations,mean_ms,p50_ms,p95_ms,min_ms,max_ms,max_error\n";
  const auto row = [&](const std::string_view name, const Stats& stats) {
    output << name << ',' << options.width << ',' << options.height << ','
           << options.measured_iterations << ',' << stats.mean << ',' << stats.p50 << ','
           << stats.p95 << ',' << stats.minimum << ',' << stats.maximum << ','
           << max_error << '\n';
  };
  row("cpu", cpu);
  row("cuda_kernel", kernel);
  row("cuda_end_to_end", end_to_end);
  row("cuda_pinned_kernel", pinned_kernel);
  row("cuda_pinned_end_to_end", pinned_end_to_end);
}

void print_row(const std::string_view name,
               const Stats& stats,
               const std::size_t point_count) {
  const double throughput = static_cast<double>(point_count) / (stats.p50 * 1000.0);
  std::cout << std::left << std::setw(18) << name << std::right << std::setw(11)
            << stats.mean << std::setw(11) << stats.p50 << std::setw(11) << stats.p95
            << std::setw(15) << throughput << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::size_t point_count =
        static_cast<std::size_t>(options.width) * options.height;
    const pointcloud::CameraIntrinsics intrinsics{
        1000.0F,
        1000.0F,
        (static_cast<float>(options.width) - 1.0F) * 0.5F,
        (static_cast<float>(options.height) - 1.0F) * 0.5F,
        options.width,
        options.height,
    };

    const std::vector<float> depth = make_depth_image(options.width, options.height);
    std::vector<pointcloud::PointXYZ> cpu_points(point_count);
    std::vector<pointcloud::PointXYZ> cuda_points(point_count);
    std::vector<pointcloud::PointXYZ> pinned_cuda_points(point_count);

    const std::vector<double> cpu_samples =
        benchmark_cpu(depth, intrinsics, cpu_points, options.warmup_iterations,
                      options.measured_iterations);
    pointcloud::CudaTimings cuda_samples;
    {
      const NvtxRange range("cuda_pageable_benchmark");
      cuda_samples = pointcloud::benchmark_depth_cuda(
          depth, intrinsics, cuda_points, options.warmup_iterations,
          options.measured_iterations);
    }
    pointcloud::CudaTimings pinned_cuda_samples;
    {
      const NvtxRange range("cuda_pinned_async_benchmark");
      pinned_cuda_samples = pointcloud::benchmark_depth_cuda_pinned(
          depth, intrinsics, pinned_cuda_points, options.warmup_iterations,
          options.measured_iterations);
    }

    const double error = std::max(maximum_error(cpu_points, cuda_points),
                                  maximum_error(cpu_points, pinned_cuda_points));
    constexpr double tolerance = 1.0e-5;
    if (error > tolerance) {
      std::cerr << "Correctness check failed: maximum error " << error
                << " exceeds tolerance " << tolerance << '\n';
      return 2;
    }

    const Stats cpu_stats = summarize(cpu_samples);
    const Stats kernel_stats = summarize(cuda_samples.kernel_ms);
    const Stats end_to_end_stats = summarize(cuda_samples.end_to_end_ms);
    const Stats pinned_kernel_stats = summarize(pinned_cuda_samples.kernel_ms);
    const Stats pinned_end_to_end_stats =
        summarize(pinned_cuda_samples.end_to_end_ms);

    if (options.verify_only) {
      std::cout << "CPU/CUDA correctness: PASS (maximum error " << error << ")\n";
      return 0;
    }

    std::cout << std::fixed << std::setprecision(4)
              << "GPU: " << pointcloud::cuda_device_name() << '\n'
              << "Frame: " << options.width << 'x' << options.height << " ("
              << point_count << " points)\n"
              << "Measured iterations: " << options.measured_iterations << "\n\n"
              << std::left << std::setw(18) << "Implementation" << std::right
              << std::setw(11) << "Mean ms" << std::setw(11) << "P50 ms"
              << std::setw(11) << "P95 ms" << std::setw(15) << "Mpoints/s" << '\n';
    print_row("CPU", cpu_stats, point_count);
    print_row("CUDA kernel", kernel_stats, point_count);
    print_row("CUDA end-to-end", end_to_end_stats, point_count);
    print_row("Pinned kernel", pinned_kernel_stats, point_count);
    print_row("Pinned end-to-end", pinned_end_to_end_stats, point_count);
    std::cout << "\nMaximum CPU/CUDA error: " << error << '\n'
              << "Kernel speedup at p50: " << cpu_stats.p50 / kernel_stats.p50 << "x\n"
              << "End-to-end speedup at p50: "
              << cpu_stats.p50 / end_to_end_stats.p50 << "x\n"
              << "Pinned transfer improvement at p50: "
              << end_to_end_stats.p50 / pinned_end_to_end_stats.p50 << "x\n"
              << "Pinned end-to-end speedup vs CPU at p50: "
              << cpu_stats.p50 / pinned_end_to_end_stats.p50 << "x\n";

    write_csv(cpu_stats, kernel_stats, end_to_end_stats, pinned_kernel_stats,
              pinned_end_to_end_stats, options, error);
    std::cout << "Results written to results/latest.csv\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
