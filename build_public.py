#!/usr/bin/env python3
"""
Builds optimized public data files from a PsycheBench report.

Splits the full report into:
  - summary.json: metadata, scores, and metrics only (small, for overview charts)
  - traces/{model_id}.json: individual trace files per model (lazy-loaded on demand)
  - data.json: full report for backwards compatibility with existing frontend
"""
import json
import sys
from pathlib import Path


def build_public(report_path: str, output_dir: str):
    """Process a benchmark report into optimized public data files."""
    with open(report_path, 'r') as f:
        report = json.load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    traces_dir = output / "traces"
    traces_dir.mkdir(exist_ok=True)

    # Build summary (no traces — just scores and metrics)
    summary = {
        "meta": report["meta"],
        "models": {}
    }

    for model_id, model_data in report["models"].items():
        summary["models"][model_id] = {
            "scores": model_data["scores"],
            "metrics": model_data["metrics"]
        }

        # Write individual trace file for each model
        safe_name = model_id.replace("/", "_").replace("\\", "_")
        trace_file = traces_dir / f"{safe_name}.json"
        with open(trace_file, 'w') as f:
            json.dump({
                "model": model_id,
                "traces": model_data.get("traces", [])
            }, f, indent=2)

    # Write summary
    summary_path = output / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Write full data.json for backwards compatibility
    data_path = output / "data.json"
    with open(data_path, 'w') as f:
        json.dump(report, f)

    # Print results
    summary_size = summary_path.stat().st_size / 1024
    data_size = data_path.stat().st_size / 1024
    trace_count = len(list(traces_dir.glob("*.json")))

    print(f"✅ Built public data in {output}/")
    print(f"   📄 summary.json:  {summary_size:.1f} KB")
    print(f"   📁 traces/:       {trace_count} model files")
    print(f"   📦 data.json:     {data_size:.1f} KB (full, backwards compat)")
    print(f"   💾 Savings:       {((data_size - summary_size) / data_size * 100):.0f}% smaller initial load with summary.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_public.py <report.json> <output_dir>")
        print()
        print("Example:")
        print("  python build_public.py results/report_20240101_120000.json ../PsycheBenchPublic/")
        sys.exit(1)

    build_public(sys.argv[1], sys.argv[2])
