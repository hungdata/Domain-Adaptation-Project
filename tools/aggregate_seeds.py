#!/usr/bin/env python3
"""
aggregate_seeds.py — Tổng hợp kết quả 3 Seeds thành bảng Mean ± Std
=====================================================================
Dùng sau khi chạy xong run_3_seeds.sh.

Usage:
    python tools/aggregate_seeds.py --working-dir ./working --seeds 2026 2027 2028
"""
import argparse
import json
import os
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate 3-seed results into Mean ± Std table")
    p.add_argument("--working-dir", default="./working", help="Base working directory")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--output", default="results/aggregated_results.md", help="Output file")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Discover all methods
    methods = set()
    for seed in args.seeds:
        seed_dir = os.path.join(args.working_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        for d in os.listdir(seed_dir):
            if d.endswith("_logs"):
                methods.add(d.replace("_logs", ""))
    methods = sorted(methods)

    # Collect metrics
    results = {}
    for method in methods:
        results[method] = {"target_test_acc": [], "source_test_acc": [],
                           "target_test_f1": [], "target_test_macro_f1": []}
        for seed in args.seeds:
            json_path = os.path.join(args.working_dir, f"seed_{seed}", f"{method}_logs", "final_test_metrics.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                for key in results[method]:
                    if key in data:
                        results[method][key].append(float(data[key]))

    # Print and write table
    lines = []
    lines.append("# Kết Quả Tổng Hợp 3 Seeds\n")
    lines.append(f"Seeds: {args.seeds}\n")
    lines.append("")
    lines.append("| Method | Target Acc (%) | Source Acc (%) | Target F1 (%) | Target Macro-F1 (%) |")
    lines.append("|--------|---------------|---------------|--------------|-------------------|")

    for method in methods:
        row = [method]
        for metric in ["target_test_acc", "source_test_acc", "target_test_f1", "target_test_macro_f1"]:
            vals = results[method][metric]
            if len(vals) == len(args.seeds):
                mean = np.mean(vals) * 100
                std = np.std(vals) * 100
                row.append(f"{mean:.2f} ± {std:.2f}")
            elif len(vals) > 0:
                if metric == "target_test_acc":
                    print(f"[SKIP] Model {method} only has {len(vals)}/{len(args.seeds)} seeds completed. Skipping.")
                row.append("N/A")
            
                row.append(f"{vals[0]*100:.2f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated automatically by `aggregate_seeds.py`*")

    output_text = "\n".join(lines)
    print(output_text)

    with open(args.output, "w") as f:
        f.write(output_text)
    print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
