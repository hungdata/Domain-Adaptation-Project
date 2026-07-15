#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
confusion_matrix_plot.py — Vẽ Confusion Matrix heatmap chất lượng Paper
=========================================================================
3 chế độ sử dụng:
  1. Từ checkpoint (.pth) → load model, evaluate trên target test set
  2. Từ final_test_metrics.json (nếu chứa confusion_matrix)
  3. Từ giá trị nhập tay: --tn --fp --fn --tp

Usage:
    # Chế độ 1: Load checkpoint và evaluate
    python tools/confusion_matrix_plot.py \\
        --model-path working/seed_2026/DSANE_MCC_logs/best_source_val_model.pth \\
        --data-root ./uda_fixed_folders

    # Chế độ 2: Đọc từ JSON
    python tools/confusion_matrix_plot.py \\
        --json working/seed_2026/DSANE_MCC_logs/final_test_metrics.json

    # Chế độ 3: Nhập tay
    python tools/confusion_matrix_plot.py --tn 45 --fp 5 --fn 3 --tp 47 --method DSANE_MCC
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thêm đường dẫn project root để import core.uda_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Tên class cho bài toán phân loại nhị phân Fresh/Rotten
CLASS_NAMES = ["Fresh", "Rotten"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Vẽ Confusion Matrix heatmap chất lượng Paper"
    )
    # ── Chế độ 1: Từ checkpoint ──
    p.add_argument("--model-path", type=str, default=None,
                    help="Đường dẫn file .pth checkpoint")
    p.add_argument("--data-root", type=str, default="./uda_fixed_folders",
                    help="Thư mục gốc dataset (cần cho chế độ checkpoint)")

    # ── Chế độ 2: Từ JSON ──
    p.add_argument("--json", type=str, default=None,
                    help="Đường dẫn final_test_metrics.json (nếu có confusion_matrix)")

    # ── Chế độ 3: Nhập tay ──
    p.add_argument("--tn", type=int, default=None, help="True Negatives (Fresh đúng)")
    p.add_argument("--fp", type=int, default=None, help="False Positives")
    p.add_argument("--fn", type=int, default=None, help="False Negatives")
    p.add_argument("--tp", type=int, default=None, help="True Positives (Rotten đúng)")

    # ── Chung ──
    p.add_argument("--method", type=str, default="",
                    help="Tên method hiển thị trên tiêu đề")
    p.add_argument("--output", type=str, default="results/confusion_matrix.png",
                    help="Đường dẫn file ảnh output")
    p.add_argument("--dpi", type=int, default=300,
                    help="Độ phân giải ảnh output (default: 300)")
    p.add_argument("--figsize", nargs=2, type=float, default=[6, 5],
                    help="Kích thước figure (width height)")
    p.add_argument("--cmap", type=str, default="Blues",
                    help="Colormap cho heatmap (default: Blues)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed cho reproducibility")
    p.add_argument("--batch-size", type=int, default=64,
                    help="Batch size khi evaluate (chế độ checkpoint)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# Chế độ 1: Load checkpoint và evaluate trên target test set
# ──────────────────────────────────────────────────────────────
def evaluate_from_checkpoint(checkpoint_path, data_root, batch_size, seed):
    """Load model từ checkpoint, evaluate trên target test, trả về confusion matrix."""
    try:
        from core.uda_utils import (
            set_seed, get_device, safe_torch_load,
            get_target_test_loader,
            FeatureExtractor, Bottleneck, ClassifierHead,
            NUM_CLASSES, FEATURE_DIM, BOTTLENECK_DIM,
        )
        import torch
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        print("   Chạy từ thư mục paper/ hoặc kiểm tra core/uda_utils.py")
        sys.exit(1)

    set_seed(seed)
    device = get_device()
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = safe_torch_load(checkpoint_path, map_location=device)

    feature_extractor = FeatureExtractor().to(device)
    bottleneck = Bottleneck(FEATURE_DIM, BOTTLENECK_DIM).to(device)
    classifier = ClassifierHead(BOTTLENECK_DIM, NUM_CLASSES).to(device)

    feature_extractor.load_state_dict(ckpt["model_state_dict"]["F_ext"])
    bottleneck.load_state_dict(ckpt["model_state_dict"]["bottleneck"])
    classifier.load_state_dict(ckpt["model_state_dict"]["classifier"])

    feature_extractor.eval()
    bottleneck.eval()
    classifier.eval()

    # Load target test data
    test_loader = get_target_test_loader(data_root, batch_size)

    # Evaluate
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = feature_extractor(images)
            bottleneck_out = bottleneck(features)
            logits = classifier(bottleneck_out)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    method_name = ckpt.get("method_name", "")

    # Accuracy
    acc = np.trace(cm) / cm.sum() * 100
    print(f"  Target Accuracy: {acc:.2f}%")

    return cm, method_name


# ──────────────────────────────────────────────────────────────
# Chế độ 2: Đọc từ JSON
# ──────────────────────────────────────────────────────────────
def load_from_json(json_path):
    """Đọc confusion matrix từ final_test_metrics.json (nếu có)."""
    with open(json_path) as f:
        data = json.load(f)

    method_name = data.get("method", "")

    if "confusion_matrix" in data:
        cm_d = data["confusion_matrix"]["target_test"]
        cm = np.array([[cm_d["tn"], cm_d["fp"]], [cm_d["fn"], cm_d["tp"]]])
        return cm, method_name
    else:
        print(f"  ⚠  JSON không chứa key 'confusion_matrix'.")
        print(f"     Các key có sẵn: {list(data.keys())}")
        return None, method_name


# ──────────────────────────────────────────────────────────────
# Chế độ 3: Từ giá trị nhập tay
# ──────────────────────────────────────────────────────────────
def build_from_manual(tn, fp, fn, tp):
    """Xây confusion matrix từ 4 giá trị TP/TN/FP/FN."""
    cm = np.array([[tn, fp],
                    [fn, tp]])
    return cm


# ──────────────────────────────────────────────────────────────
# Vẽ Confusion Matrix heatmap (paper-quality)
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, method_name, output_path, dpi, figsize, cmap):
    """Vẽ heatmap confusion matrix với styling chất lượng Paper."""
    try:
        import seaborn as sns
    except ImportError:
        print("❌ Cần cài seaborn: pip install seaborn")
        sys.exit(1)

    # ── Paper-quality styling ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Tính phần trăm
    cm_sum = cm.sum()
    cm_pct = cm / cm_sum * 100 if cm_sum > 0 else cm * 0

    # Tạo annotation: "count\n(xx.x%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_pct[i, j]
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    # Vẽ heatmap
    sns.heatmap(
        cm, annot=annot, fmt="",
        cmap=cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1.5,
        linecolor="white",
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Count"},
        ax=ax,
        annot_kws={"size": 16, "fontweight": "bold"},
    )

    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")

    # Tiêu đề
    title = "Confusion Matrix"
    if method_name:
        title = f"Confusion Matrix — {method_name}"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Thêm thông tin bổ sung bên dưới
    total = cm.sum()
    acc = np.trace(cm) / total * 100 if total > 0 else 0
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    info_text = (f"Accuracy: {acc:.1f}% | "
                 f"Precision: {precision:.1f}% | "
                 f"Recall: {recall:.1f}% | "
                 f"Total: {total}")
    fig.text(0.5, 0.01, info_text, ha="center", fontsize=10,
             style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  📊 Confusion matrix saved: {output_path}")

    # In bảng tóm tắt ra terminal
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Confusion Matrix Summary            │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  TN (Fresh→Fresh)   = {tn:>5}          │")
    print(f"  │  FP (Fresh→Rotten)  = {fp:>5}          │")
    print(f"  │  FN (Rotten→Fresh)  = {fn:>5}          │")
    print(f"  │  TP (Rotten→Rotten) = {tp:>5}          │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Accuracy  = {acc:>6.2f}%               │")
    print(f"  │  Precision = {precision:>6.2f}%               │")
    print(f"  │  Recall    = {recall:>6.2f}%               │")
    print(f"  └─────────────────────────────────────┘")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    cm = None
    method_name = args.method

    # ── Ưu tiên: checkpoint > json > manual ──

    if args.model_path:
        print(f"Chế độ 1: Đánh giá từ checkpoint")
        cm, detected_method = evaluate_from_checkpoint(
            args.model_path, args.data_root, args.batch_size, args.seed
        )
        if not method_name:
            method_name = detected_method

    elif args.json:
        print(f"📊 Chế độ 2: Đọc từ JSON — {args.json}")
        cm, detected_method = load_from_json(args.json)
        if not method_name:
            method_name = detected_method
        if cm is None:
            print("❌ Không thể lấy confusion matrix từ JSON.")
            print("   Thử dùng --model-path hoặc --tn --fp --fn --tp")
            return

    elif all(v is not None for v in [args.tn, args.fp, args.fn, args.tp]):
        print(f"📊 Chế độ 3: Nhập tay — TN={args.tn}, FP={args.fp}, "
              f"FN={args.fn}, TP={args.tp}")
        cm = build_from_manual(args.tn, args.fp, args.fn, args.tp)

    else:
        print("❌ Cần cung cấp 1 trong 3 nguồn dữ liệu:")
        print("   --model-path path/to/model.pth")
        print("   --json path/to/final_test_metrics.json")
        print("   --tn X --fp X --fn X --tp X")
        return

    # Vẽ
    plot_confusion_matrix(
        cm, method_name, args.output,
        args.dpi, tuple(args.figsize), args.cmap
    )

    print(f"\n✅ Hoàn tất!")


if __name__ == "__main__":
    main()
