# CUDA Depth-to-Point-Cloud Benchmark

A reproducible modern C++20 application with a portable C++17 CUDA translation
unit that converts a synthetic depth image into an organized XYZ point cloud.
It compares a CPU reference implementation with a custom CUDA kernel, verifies
numerical agreement, and reports mean, p50, and p95 latency.

## Why this project exists

This repository is an initial evidence artifact for robotics-perception roles
that require CUDA and modern C++. It demonstrates a custom kernel, explicit GPU
memory management, RAII, CMake, repeatable measurements, correctness testing,
and profiler-ready line information. It does not claim production deployment
or Isaac Sim integration yet.

## Requirements

- Ubuntu 22.04 under WSL 2
- NVIDIA GPU visible through `nvidia-smi`
- CUDA Toolkit 13.3
- CMake 3.22 or newer
- Ninja

The default CUDA architecture is `sm_89`, matching the RTX 4050 Laptop GPU.

## Build and test

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run the benchmark

```bash
./build/depth_benchmark
```

Change the workload and sample count:

```bash
./build/depth_benchmark --width 1280 --height 720 --warmup 10 --iterations 100
```

Results are saved to `results/latest.csv`. The CUDA end-to-end measurement
includes host-to-device transfer, kernel execution, synchronization, and
device-to-host transfer. The CUDA kernel measurement isolates execution using
CUDA events.

## Verified baseline on this machine

Environment: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), CUDA Toolkit 13.3,
Ubuntu 22.04 under WSL 2. Workload: 1920 x 1080 depth frame, five warm-up runs,
and 30 measured runs.

| Implementation | Mean (ms) | P50 (ms) | P95 (ms) | P50 throughput |
|---|---:|---:|---:|---:|
| CPU | 2.8606 | 2.5514 | 4.2723 | 812.7 Mpoints/s |
| CUDA kernel | 0.2006 | 0.1965 | 0.2301 | 10,553.7 Mpoints/s |
| CUDA end-to-end | 4.7512 | 4.5975 | 6.8834 | 451.0 Mpoints/s |

The custom kernel achieved a 12.99x p50 compute speedup with zero numerical
error against the CPU reference. End-to-end execution was 0.56x the CPU speed
because synchronous host/device transfers dominated this small pipeline. This
is the baseline optimization problem, not a result to hide.

### First measured optimization

Pinned host buffers, a non-blocking CUDA stream, and asynchronous H2D/D2H copies
were added while preserving the original pageable implementation. A 100-run
1080p comparison measured:

| Path | P50 (ms) | P95 (ms) |
|---|---:|---:|
| Pageable CUDA end-to-end | 4.3058 | 7.4899 |
| Pinned async CUDA end-to-end | 2.8131 | 3.0784 |

Two independent 100-run sessions measured a **1.28x-1.53x p50 improvement** and
a **2.23x-2.43x p95 improvement**, with zero numerical error. Reporting the
range avoids selecting only the best run. The pinned path remains 0.81x-0.86x
CPU speed end to end because the benchmark copies a 24.9 MB XYZ point cloud
back to the host. Keeping downstream filtering and inference on the GPU is the
next architectural optimization.

Nsight Systems independently confirmed that device-to-host copies represented
64.8% of GPU memory-transfer time in the labeled trace. Inspect
`results/depth_pipeline_timeline.nsys-rep` in Nsight Systems and
`results/depth_to_pointcloud_baseline.ncu-rep` in Nsight Compute.

## Profile with Nsight Compute

```bash
/usr/local/cuda-13.3/bin/ncu \
  --set full \
  --target-processes all \
  --export results/depth_to_pointcloud_profile \
  ./build/depth_benchmark --width 1920 --height 1080 --warmup 1 --iterations 1
```

If Nsight reports `ERR_NVGPUCTRPERM`, open the NVIDIA App as an administrator,
then select **System > Advanced > Developer > Manage GPU Performance Counters**
and allow access for all users. Rerun the command after applying the setting.

The repository includes the baseline Nsight Compute report and the labeled
Nsight Systems timeline. Pinned memory and asynchronous copies are implemented;
use the reports to reproduce the measurements or evaluate later optimizations.

## Evidence roadmap

- [x] Enable GPU performance-counter access and preserve profiler reports.
- [x] Add pinned memory, a non-blocking stream, and asynchronous copies.
- [x] Add GoogleTest cases for camera intrinsics and invalid-depth handling.
- [x] Add a real RGB-D frame with documented provenance.
- [x] Integrate the projector as a ROS 2 C++ component.
- [x] Feed depth images from an Isaac Sim camera through the ROS 2 bridge.
- [x] Run randomized simulation trials and publish measured results.


