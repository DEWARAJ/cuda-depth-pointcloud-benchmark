# Nsight Compute baseline summary

Profile: `depth_to_pointcloud_baseline.ncu-rep`

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9
- Kernel grid: 8,100 blocks x 256 threads (2,073,600 threads)
- Profiled kernel duration: 96.59 microseconds average
- Theoretical occupancy: 100%
- Achieved occupancy: 79.11%
- Memory throughput utilization: 44.89%
- Compute throughput utilization: 27.61%
- Measured memory throughput: 57.39 GB/s
- Registers per thread: 24
- Local/shared memory spilling: 0 bytes
- Branch efficiency: 100%

Interpretation: the depth-projection kernel is compact, fully coalesced enough to
reach useful memory throughput, and not limited by register pressure or branch
divergence. Application-level measurements show that synchronous pageable
host/device transfers dominate the pipeline, so pinned asynchronous transfers
were selected as the first optimization.
