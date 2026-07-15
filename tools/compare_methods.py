#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_methods.py — Tạo bảng so sánh tổng hợp 8 Methods × 3 Seeds
=====================================================================
Đọc final_test_metrics.json từ working/seed_{seed}/{METHOD}_logs/,
tính Mean ± Std, xuất ra file .csv và .md.
Kết quả tốt nhất được in đậm (bold), kết quả tốt thứ 2 được gạch chân.

Usage:
    python tools/compare_methods.py
    python tools/compare_methods.py --working-dir ./working --seeds 2026 2027 2028
    python tools/compare_methods.py --output results/method_comparison
"""

import argparse
import json
import os
import csv
import numpy as np


# ──────────────────────────────────────────────────────────────
# Thứ tự chuẩn hiển thị trong bảng Paper (Baselines → Proposed)
# ──────────────────────────────────────────────────────────────
METHOD_ORDER = [
    "ERM", "DANN", "CDAN", "CDANE", "CDANC", "CDANE_MCC",
    "DSAN", "DSAN_SEP", "DSANE", "DSANE_SEP", 
    "CDANC", "CDANE_MCC", "CDANE_MCC_SEP",
    "CDAN_ENT", "CDANE_ENT", "DSANE_ENT"
]

# Các chỉ số đánh giá cần tổng hợp
METRICS = [
    "source_test_acc",
    "target_test_acc",
    "target_test_f1",
    "target_test_macro_f1",
]

# Tên cột hiển thị trên bảng
METRIC_LABELS = {
    "source_test_acc":      "Source Acc (%)",
    "target_test_acc":      "Target Acc (%)",
    "target_test_f1":       "Target F1 (%)",
    "target_test_macro_f1": "Target Macro-F1 (%)",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="So sánh tổng hợp 8 methods × 3 seeds → .csv + .md"
    )
    p.add_argument("--working-dir", default="./working",
                    help="Thư mục gốc chứa seed_* (default: ./working)")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028],
                    help="Danh sách seeds (default: 2026 2027 2028)")
    p.add_argument("--output", default="results/method_comparison",
                    help="Prefix đường dẫn output (sẽ tạo .csv và .md)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Thu thập dữ liệu
# ──────────────────────────────────────────────────────────────
def collect_results(working_dir, seeds):
    """Đọc toàn bộ final_test_metrics.json, trả về dict {method: {metric: [values]}}"""
    # Phát hiện tất cả methods có trong working dir
    discovered = set()
    for seed in seeds:
        seed_dir = os.path.join(working_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        for d in os.listdir(seed_dir):
            if d.endswith("_logs"):
                discovered.add(d.replace("_logs", ""))

    # Sắp xếp theo thứ tự chuẩn, các method mới phát hiện thêm xếp cuối
    methods = [m for m in METHOD_ORDER if m in discovered]
    extras = sorted(discovered - set(METHOD_ORDER))
    methods.extend(extras)

    # Đọc kết quả
    results = {}
    for method in methods:
        results[method] = {m: [] for m in METRICS}
        for seed in seeds:
            json_path = os.path.join(
                working_dir, f"seed_{seed}", f"{method}_logs",
                "final_test_metrics.json"
            )
            if not os.path.exists(json_path):
                print(f"  ⚠  Thiếu: {json_path}")
                continue
            with open(json_path) as f:
                data = json.load(f)
            for key in METRICS:
                if key in data:
                    results[method][key].append(float(data[key]))

    return methods, results


# ──────────────────────────────────────────────────────────────
# Tính Mean ± Std
# ──────────────────────────────────────────────────────────────
def compute_stats(results, methods, seeds):
    """Trả về dict {method: {metric: (mean, std, n_seeds)}}"""
    stats = {}
    for method in methods:
        stats[method] = {}
        for metric in METRICS:
            vals = results[method][metric]
            if len(vals) == len(seeds):
                mean = np.mean(vals) * 100
                std = np.std(vals, ddof=0) * 100  # population std (3 seeds)
                stats[method][metric] = (mean, std, len(vals))
            elif len(vals) > 0:
                print(f"[SKIP] Model {method} only has {len(vals)}/{len(seeds)} seeds completed. Skipping computation.")
                stats[method][metric] = (None, None, len(vals))
            

            else:
                stats[method][metric] = (None, None, 0)
    return stats


# ──────────────────────────────────────────────────────────────
# Xác định best / second-best cho mỗi cột
# ──────────────────────────────────────────────────────────────
def find_best_and_second(stats, methods, metric):
    """Trả về (best_method, second_method) dựa trên mean cao nhất."""
    valid = [(m, stats[m][metric][0]) for m in methods
             if stats[m][metric][0] is not None]
    if len(valid) < 1:
        return None, None
    valid.sort(key=lambda x: x[1], reverse=True)
    best = valid[0][0]
    second = valid[1][0] if len(valid) >= 2 else None
    return best, second


# ──────────────────────────────────────────────────────────────
# Xuất file .csv
# ──────────────────────────────────────────────────────────────
def write_csv(path, stats, methods, seeds):
    """Ghi bảng kết quả ra file CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["Method"]
        for metric in METRICS:
            header.append(f"{METRIC_LABELS[metric]} Mean")
            header.append(f"{METRIC_LABELS[metric]} Std")
            header.append(f"{METRIC_LABELS[metric]} N")
        writer.writerow(header)

        # Data
        for method in methods:
            row = [method]
            for metric in METRICS:
                mean, std, n = stats[method][metric]
                if mean is not None:
                    row.extend([f"{mean:.2f}", f"{std:.2f}", str(n)])
                else:
                    row.extend(["N/A", "N/A", "0"])
            writer.writerow(row)

    print(f"  📄 CSV saved: {path}")


# ──────────────────────────────────────────────────────────────
# Xuất file .md (Markdown)
# ──────────────────────────────────────────────────────────────
def format_cell_md(mean, std, is_best, is_second):
    """Format 1 ô trong bảng Markdown: bold best, underline second."""
    if mean is None:
        return "N/A"
    text = f"{mean:.2f} ± {std:.2f}"
    if is_best:
        return f"**{text}**"
    elif is_second:
        return f"<u>{text}</u>"
    return text


def write_markdown(path, stats, methods, seeds):
    """Ghi bảng kết quả ra file Markdown."""
    # Tìm best / second cho mỗi metric
    rankings = {}
    for metric in METRICS:
        best, second = find_best_and_second(stats, methods, metric)
        rankings[metric] = (best, second)

    lines = []
    lines.append("# Bảng So Sánh Tổng Hợp — 8 Methods × 3 Seeds\n")
    lines.append(f"**Seeds**: {seeds}\n")
    lines.append(f"**Backbone**: MobileNetV3-Small | **Epochs**: 150 | "
                 f"**Batch Size**: 64 | **LR**: 0.01\n")
    lines.append("")

    # Header
    cols = ["Method"] + [METRIC_LABELS[m] for m in METRICS]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    # Rows
    for method in methods:
        row = [f"`{method}`"]
        for metric in METRICS:
            mean, std, _ = stats[method][metric]
            best, second = rankings[metric]
            is_best = (method == best)
            is_second = (method == second)
            row.append(format_cell_md(mean, std, is_best, is_second))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("> **Bold** = Best, <u>Underline</u> = Second-best")
    lines.append("")
    lines.append("---")
    lines.append(f"*Tự động tạo bởi `tools/compare_methods.py`*")

    output_text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(output_text)
    print(f"  📝 Markdown saved: {path}")

    # In ra terminal
    print("\n" + output_text)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"📊 Compare Methods — Seeds: {args.seeds}")
    print(f"   Working dir: {args.working_dir}")
    print()

    # 1. Thu thập kết quả
    methods, results = collect_results(args.working_dir, args.seeds)
    if not methods:
        print("❌ Không tìm thấy method nào! Kiểm tra lại --working-dir.")
        return

    print(f"   Tìm thấy {len(methods)} methods: {methods}\n")

    # 2. Tính Mean ± Std
    stats = compute_stats(results, methods, args.seeds)

    # 3. Xuất CSV + Markdown
    csv_path = args.output + ".csv"
    md_path = args.output + ".md"
    write_csv(csv_path, stats, methods, args.seeds)
    write_markdown(md_path, stats, methods, args.seeds)

    print(f"\n✅ Hoàn tất! Output: {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
