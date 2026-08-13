#!/usr/bin/env python3
import csv
import statistics
import sys
from pathlib import Path


def main() -> int:
    trial_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/trials")
    files = sorted(trial_dir.glob("trial_*.csv"))
    if not files:
        raise SystemExit(f"no trial CSV files found in {trial_dir}")

    samples: dict[str, list[dict[str, str]]] = {}
    for path in files:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                samples.setdefault(row["implementation"], []).append(row)

    output_path = trial_dir / "summary.csv"
    fields = [
        "implementation",
        "trials",
        "mean_of_p50_ms",
        "stdev_p50_ms",
        "min_p50_ms",
        "max_p50_ms",
        "mean_p95_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for implementation, rows in sorted(samples.items()):
            p50 = [float(row["p50_ms"]) for row in rows]
            p95 = [float(row["p95_ms"]) for row in rows]
            writer.writerow(
                {
                    "implementation": implementation,
                    "trials": len(rows),
                    "mean_of_p50_ms": f"{statistics.fmean(p50):.6f}",
                    "stdev_p50_ms": f"{statistics.stdev(p50) if len(p50) > 1 else 0.0:.6f}",
                    "min_p50_ms": f"{min(p50):.6f}",
                    "max_p50_ms": f"{max(p50):.6f}",
                    "mean_p95_ms": f"{statistics.fmean(p95):.6f}",
                }
            )
    print(f"wrote {output_path} from {len(files)} trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
