import json
import platform
import time
from pathlib import Path

from isaacsim import SimulationApp


output_path = Path("/workspace/output/isaac_smoke_metrics.json")
started = time.perf_counter()
simulation_app = SimulationApp({"headless": True})
startup_seconds = time.perf_counter() - started

frame_started = time.perf_counter()
frame_count = 300
for _ in range(frame_count):
    simulation_app.update()
frame_seconds = time.perf_counter() - frame_started

metrics = {
    "status": "pass",
    "platform": platform.platform(),
    "startup_seconds": startup_seconds,
    "frames": frame_count,
    "frame_loop_seconds": frame_seconds,
    "mean_frame_ms": frame_seconds * 1000.0 / frame_count,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
simulation_app.close()
