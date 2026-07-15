#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_latex_table.py — Tạo bảng LaTeX chuẩn Paper UDA
==========================================================
Đọc final_test_metrics.json, tính Mean ± Std, xuất bảng LaTeX
sẵn sàng copy-paste vào file .tex.

Sử dụng booktabs, \\textbf{} cho best, \\underline{} cho second-best.

Usage:
    python tools/generate_latex_table.py
    python tools/generate_latex_table.py --working-dir ./working --seeds 2026 2027 2028
    python tools/generate_latex_table.py --output results/latex_table.tex
"""

import argparse
import json
import os
import numpy as np


# ──────────────────────────────────────────────────────────────
# Thứ tự chuẩn hiển thị trong bảng Paper
# ──────────────────────────────────────────────────────────────
METHOD_ORDER = [
    "ERM", "DANN", "CDAN", "CDANE",
    "DSAN", "DSANE", "DSANE_MCC", "CDANE_MCC",
]

# Tên hiển thị đẹp hơn trong bảng LaTeX
METHOD_DISPLAY = {
    "ERM":        "ERM (Source Only)",
    "DANN":       "DANN",
    "CDAN":       "CDAN",
    "CDANE":      "CDAN+E",
    "DSAN":       "DSAN",
    "DSANE":      "DSAN+E",
    "DSANE_MCC":  "DSAN+E+MCC (Ours)",
    "CDANE_MCC":  "CDAN+E+MCC (Ours)",
}

METRICS = [
    "source_test_acc",
    "target_test_acc",
    "target_test_f1",
    "target_test_macro_f1",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Tạo bảng LaTeX chuẩn Paper từ kết quả thực nghiệm"
    )
    p.add_argument("--working-dir", default="./working",
                    help="Thư mục gốc chứa seed_* (default: ./working)")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028],
                    help="Danh sách seeds (default: 2026 2027 2028)")
    p.add_argument("--output", default="results/latex_table.tex",
                    help="Đường dẫn file .tex output")
    p.add_argument("--caption", default="Comparison of UDA methods on Fruit Freshness dataset (MobileNetV3-Small). Best results in \\\\textbf{bold}, second-best \\\\underline{underlined}.",
                    help="Caption cho bảng LaTeX")
    p.add_argument("--label", default="tab:uda_comparison",
                    help="Label cho bảng LaTeX")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Thu thập dữ liệu (tương tự compare_methods.py)
# ──────────────────────────────────────────────────────────────
def collect_results(working_dir, seeds):
    """Đọc toàn bộ final_test_metrics.json."""
    discovered = set()
    for seed in seeds:
        seed_dir = os.path.join(working_dir, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        for d in os.listdir(seed_dir):
            if d.endswith("_logs"):
                discovered.add(d.replace("_logs", ""))

    # Sắp xếp theo thứ tự chuẩn
    methods = [m for m in METHOD_ORDER if m in discovered]
    extras = sorted(discovered - set(METHOD_ORDER))
    methods.extend(extras)

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


def compute_stats(results, methods, seeds):
    """Tính Mean ± Std cho mỗi method × metric."""
    stats = {}
    for method in methods:
        stats[method] = {}
        for metric in METRICS:
            vals = results[method][metric]
            if len(vals) == len(seeds):
                mean = np.mean(vals) * 100
                std = np.std(vals, ddof=0) * 100
                stats[method][metric] = (mean, std, len(vals))
            elif len(vals) > 0:
                print(f"[SKIP] Model {method} only has {len(vals)}/{len(seeds)} seeds completed. Skipping computation.")
                stats[method][metric] = (None, None, len(vals))
            

            else:
                stats[method][metric] = (None, None, 0)
    return stats


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
# Format LaTeX cell
# ──────────────────────────────────────────────────────────────
def format_cell_latex(mean, std, is_best, is_second):
    """Format 1 ô LaTeX: \\textbf{} cho best, \\underline{} cho second."""
    if mean is None:
        return "N/A"
    # Dùng $\\pm$ để có ký hiệu ± đẹp trong LaTeX
    text = f"{mean:.2f} $\\pm$ {std:.2f}"
    if is_best:
        return f"\\textbf{{{text}}}"
    elif is_second:
        return f"\\underline{{{text}}}"
    return text


# ──────────────────────────────────────────────────────────────
# Sinh bảng LaTeX hoàn chỉnh
# ──────────────────────────────────────────────────────────────
def generate_latex(stats, methods, seeds, caption, label):
    """Sinh nội dung bảng LaTeX đầy đủ với booktabs."""

    # Tìm best / second cho mỗi metric
    rankings = {}
    for metric in METRICS:
        best, second = find_best_and_second(stats, methods, metric)
        rankings[metric] = (best, second)

    lines = []
    lines.append("% ============================================================")
    lines.append("% Bảng tự động sinh bởi: tools/generate_latex_table.py")
    lines.append(f"% Seeds: {seeds}")
    lines.append("% ============================================================")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{lcccc}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Method} & \\textbf{Source Acc (\\%)} & "
                 "\\textbf{Target Acc (\\%)} & \\textbf{Target F1 (\\%)} & "
                 "\\textbf{Target Macro-F1 (\\%)} \\\\")
    lines.append("    \\midrule")

    for i, method in enumerate(methods):
        display_name = METHOD_DISPLAY.get(method, method)

        # Escape ký tự đặc biệt cho LaTeX
        display_name = display_name.replace("_", "\\_")

        cells = [display_name]
        for metric in METRICS:
            mean, std, _ = stats[method][metric]
            best, second = rankings[metric]
            is_best = (method == best)
            is_second = (method == second)
            cells.append(format_cell_latex(mean, std, is_best, is_second))

        row = "    " + " & ".join(cells) + " \\\\"

        # Thêm \\midrule giữa baselines và proposed methods
        if method == "DSANE" and i < len(methods) - 1:
            row += "\n    \\midrule"

        lines.append(row)

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"📊 Generate LaTeX Table — Seeds: {args.seeds}")
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

    # 3. Sinh LaTeX
    # Unescape caption (argparse sẽ double-escape backslash)
    caption = args.caption.replace("\\\\", "\\")
    latex_content = generate_latex(stats, methods, args.seeds, caption, args.label)

    # 4. Ghi file
    with open(args.output, "w") as f:
        f.write(latex_content)

    print(latex_content)
    print(f"\n✅ LaTeX table saved: {args.output}")


if __name__ == "__main__":
    main()
