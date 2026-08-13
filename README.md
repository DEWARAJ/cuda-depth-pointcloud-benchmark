# CUDA Depth-to-Point-Cloud Benchmark

A reproducible modern C++20 application with a portable C++17 CUDA translation
unit that converts a synthetic depth image into an organized XYZ point cloud.
It compares a CPU reference implementation with custom CUDA kernels, verifies
numerical agreement, and reports mean, p50, and p95 latency. The second-stage
pipeline projects and voxel-compacts the cloud entirely on the GPU, returning
only the compacted count instead of a 24.9 MB organized cloud.

## Why this project exists

This repository is an evidence artifact for robotics-perception roles that
require CUDA, modern C++, ROS 2, and simulation/cloud awareness. It demonstrates
custom kernels, atomics, explicit GPU memory management, RAII, CMake, repeatable
measurements, correctness tests, a composable ROS 2 node, CI, profiler reports,
and a cost-controlled AWS/Isaac Sim workflow. Cloud results are intentionally
not claimed until an authenticated cloud run produces the JSON evidence.

Non-positive and non-finite depth samples are converted to invalid XYZ points
and excluded from voxel occupancy on both the CPU and CUDA paths.

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

GitHub Actions performs a clean CUDA compilation and runs the CPU tests in an
NVIDIA CUDA 12.8 build container. GPU correctness remains a local/self-hosted
test because GitHub-hosted runners do not expose NVIDIA GPUs.

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

### GPU-resident voxel pipeline

The optimized path keeps XYZ data on the device, atomically elects one point
per occupied 5 cm voxel, compacts those owners, and copies only the output
count to the host. A 100-run 1080p session produced:

| Path | Mean (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|
| CPU projection | 4.2109 | 4.0293 | 5.7841 |
| Pinned projection + full cloud readback | 2.7910 | 2.7573 | 2.8927 |
| GPU projection + voxel compaction | 1.2963 | 1.2659 | 1.4277 |

CPU and GPU both reported **7,385 occupied voxels**. The device-resident
pipeline was **2.18x faster at p50** than the pinned path that copied the full
cloud to the CPU. The exact measurements are preserved in
`results/voxel_pipeline_rtx4050_1080p.csv`.

Run a multi-session stability study and aggregate p50 variance with:

```bash
scripts/run_benchmark_trials.sh 10 30
```

The checked-in ten-session study measured a mean voxel-pipeline p50 of
**1.3791 ms** (0.0483 ms standard deviation) versus **3.0100 ms** for pinned
full-cloud readback. See `results/trials/summary.csv` and the ten raw trial
files; these are separate sessions, not selected iterations from one run.

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

## Compute Sanitizer

```bash
scripts/run_compute_sanitizer.sh
```

On Windows/WSL, NVIDIA requires a machine-wide WDDM debugger interface. This
machine currently records that setup block honestly in
`results/compute_sanitizer_memcheck.txt`; it is not a sanitizer pass. If you
explicitly accept the security impact, open Administrator PowerShell, run
`scripts/enable_wsl_compute_sanitizer.ps1 -IUnderstandSecurityRisk`, execute
`wsl --shutdown`, reopen Ubuntu, and rerun the sanitizer. See NVIDIA's current
[Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

## ROS 2 composable component

The optional component subscribes to a `32FC1` depth image on `/depth`, runs
the pinned asynchronous CUDA projection, and publishes an organized
`sensor_msgs/PointCloud2` on `/points`. A synthetic publisher supports a live
smoke test without a camera or simulator.

```bash
source /opt/ros/humble/setup.bash
colcon build --cmake-args \
  -DBUILD_ROS2_COMPONENT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
source install/setup.bash

# terminal 1
ros2 run rclcpp_components component_container
# terminal 2
ros2 component load /ComponentManager cuda_depth_pointcloud_benchmark \
  pointcloud::ros2::CudaDepthComponent
# terminal 3
ros2 run cuda_depth_pointcloud_benchmark synthetic_depth_publisher
ros2 topic hz /points
```

After building with the `ros_build`/`ros_install` paths used in this repository,
run the automated live check with `scripts/run_ros2_smoke_test.sh`. It launches
the component container and synthetic publisher, requires an observed
`PointCloud2` message, and saves the message/log evidence under `results/`.

Camera intrinsics are ROS parameters (`fx`, `fy`, `cx`, `cy`). Zero/negative
defaults derive a centered synthetic camera model from the input image.

## Isaac Sim cloud workflow

This laptop's RTX 4050 has 6 GB VRAM and the machine has 16 GB RAM, below Isaac
Sim 6.0's current 16 GB VRAM and 32 GB RAM minimum. The `cloud/aws` Terraform
configuration therefore targets a `g6.2xlarge` NVIDIA L4 instance, restricts
all inbound access to one CIDR, uses encrypted disposable storage, and
automatically terminates after 120 minutes. WebRTC ports are closed by default.

The headless smoke test uses NVIDIA's current `nvcr.io/nvidia/isaac-sim:6.0.1`
container and writes machine-readable timing evidence under `results/isaac/`:

```bash
export ACCEPT_NVIDIA_EULA=Y
isaac/run_cloud_smoke_test.sh
```

Follow the [official Isaac Sim AWS deployment guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html)
for the Marketplace subscription/AMI. No instance has been launched from this
repository because no authenticated AWS account, AMI subscription, key pair,
or billing authorization is present.

## Evidence roadmap

- [x] Enable GPU performance-counter access and preserve profiler reports.
- [x] Add pinned memory, a non-blocking stream, and asynchronous copies.
- [x] Add CPU unit tests and CPU/GPU correctness tests.
- [x] Add GPU-resident voxel filtering with a CPU validation reference.
- [x] Add a ROS 2 C++ composable component and synthetic input publisher.
- [x] Add CUDA build CI and an opt-in Compute Sanitizer workflow.
- [x] Add cost-controlled AWS IaC and an Isaac Sim smoke-test harness.
- [ ] Add a real RGB-D frame with documented provenance.
- [ ] Feed depth images from an Isaac Sim camera through the ROS 2 bridge.
- [ ] Run authenticated Isaac Sim trials and publish measured results/costs.

Do not place speedup numbers on a resume until they have been reproduced from
a clean build and preserved in version-controlled results.
