#!/usr/bin/env python3
"""
plot_training_curves.py — Vẽ đồ thị Loss và Accuracy theo Epoch
================================================================
Usage:
    # Vẽ 1 method:
    python tools/plot_training_curves.py --csv working/seed_2026/DSAN_logs/epoch_log.csv

    # So sánh nhiều methods:
    python tools/plot_training_curves.py \
        --csv working/seed_2026/ERM_logs/epoch_log.csv \
             working/seed_2026/DSAN_logs/epoch_log.csv \
             working/seed_2026/DSANE_MCC_logs/epoch_log.csv \
        --output results/training_curves_comparison.png
"""
import argparse, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Plot training curves from epoch_log.csv")
    p.add_argument("--csv", nargs="+", required=True, help="Path(s) to epoch_log.csv")
    p.add_argument("--output", default="results/training_curves.png")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()

def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Curves Comparison", fontsize=16, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(args.csv)))

    for idx, csv_path in enumerate(args.csv):
        rows = read_csv(csv_path)
        if not rows:
            continue

        method = rows[0].get("method", f"Method_{idx}")
        epochs = [int(r["epoch"]) for r in rows]
        color = colors[idx]

        # Plot 1: Accuracy
        ax = axes[0, 0]
        src_val_acc = [safe_float(r.get("source_val_acc")) for r in rows]
        tgt_acc = [safe_float(r.get("target_monitor_acc")) for r in rows]
        ax.plot(epochs, src_val_acc, '-', color=color, alpha=0.5, label=f"{method} Src Val")
        ax.plot(epochs, tgt_acc, '-', color=color, linewidth=2, label=f"{method} Tgt")
        ax.set_title("Accuracy over Epochs")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)

        # Plot 2: Total Loss
        ax = axes[0, 1]
        total_loss = [safe_float(r.get("train_total_loss")) for r in rows]
        ax.plot(epochs, total_loss, '-', color=color, linewidth=2, label=method)
        ax.set_title("Total Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Plot 3: Domain / LMMD / MCC Loss
        ax = axes[1, 0]
        lmmd = [safe_float(r.get("lmmd_loss")) for r in rows]
        domain = [safe_float(r.get("domain_loss")) for r in rows]
        mcc = [safe_float(r.get("mcc_loss")) for r in rows]
        if any(v > 0 for v in lmmd):
            ax.plot(epochs, lmmd, '-', color=color, linewidth=2, label=f"{method} LMMD")
        if any(v > 0 for v in domain):
            ax.plot(epochs, domain, '--', color=color, linewidth=1.5, label=f"{method} Domain")
        if any(v > 0 for v in mcc):
            ax.plot(epochs, mcc, ':', color=color, linewidth=1.5, label=f"{method} MCC")
        ax.set_title("Domain Adaptation Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Plot 4: Target Entropy & Confidence
        ax = axes[1, 1]
        entropy = [safe_float(r.get("target_monitor_entropy")) for r in rows]
        confidence = [safe_float(r.get("target_monitor_confidence")) for r in rows]
        ax.plot(epochs, entropy, '-', color=color, linewidth=2, label=f"{method} Entropy")
        ax.plot(epochs, confidence, '--', color=color, linewidth=1.5, label=f"{method} Confidence")
        ax.set_title("Target Entropy & Confidence")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
