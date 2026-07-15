#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DSAN (Deep Subdomain Adaptation Network)
=====================================
Loss tổng: L = L_CE_source + λ · L_LMMD_target

Các quy tắc áp dụng:
1. GHÉP BATCH (Nối Ảnh): Source và Target được ghép chung rồi đi qua mạng 1 lần duy nhất để dùng chung thống kê BatchNorm.
2. KHÓA TARGET (Warmup Lock): Trong 10 epoch đầu, Target bị khóa hoàn toàn.
3. ZERO DATA LEAKAGE: Mốc test cuối cùng dùng "last_model.pth".
"""

import argparse
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.uda_utils import (
    safe_torch_load,
    build_base_ckpt,
    save_training_state,
    seed_epoch,
    configure_determinism,
    images_from_batch,
    set_rng_state,
    set_seed, get_device, get_gpu_memory_mb,
    get_source_train_loader, get_source_val_loader, get_source_test_loader,
    get_target_train_loader, get_target_test_loader,
    FeatureExtractor, Bottleneck, ClassifierHead, compute_lmmd,
    evaluate, init_csv, append_csv, save_json,
    build_param_groups, make_optimizer, adjust_lr,
    lambda_schedule, InfiniteIterator, print_epoch,
    EPOCH_LOG_FIELDS, TARGET_EVAL_FIELDS, SOURCE_EVAL_FIELDS,
    SEED, EPOCHS, WARMUP_EPOCHS, LR, BATCH_SIZE, BACKBONE_LR_FACTOR,
)

METHOD = "DSAN"

def parse_args():
    parser = argparse.ArgumentParser(description="DSAN")
    parser.add_argument("--data-root", type=str, default="./uda_fixed_folders")
    parser.add_argument("--output-dir", type=str, default="./working")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    configure_determinism(args.deterministic)
    device = get_device()

    model_dir = os.path.join(args.output_dir, f"{METHOD}_model")
    ckpt_dir  = os.path.join(model_dir, "epoch_checkpoints")
    log_dir   = os.path.join(args.output_dir, f"{METHOD}_logs")
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

    src_train_loader = get_source_train_loader(args.data_root, args.batch_size, args.num_workers)
    src_val_loader   = get_source_val_loader(args.data_root, args.batch_size, args.num_workers)
    src_test_loader  = get_source_test_loader(args.data_root, args.batch_size, args.num_workers)
    tgt_train_loader = get_target_train_loader(args.data_root, args.batch_size, args.num_workers)
    tgt_test_loader  = get_target_test_loader(args.data_root, args.batch_size, args.num_workers)

    F_ext       = FeatureExtractor().to(device)
    bottleneck  = Bottleneck().to(device)
    classifier  = ClassifierHead().to(device)
    modules     = {"F_ext": F_ext, "bottleneck": bottleneck, "classifier": classifier}

    param_groups = build_param_groups(F_ext, bottleneck, classifier, lr=args.lr)
    optimizer    = make_optimizer(param_groups, lr=args.lr)
    
    ce_loss_fn   = nn.CrossEntropyLoss()

    epoch_log_path  = os.path.join(log_dir, "epoch_log.csv")
    target_eval_path = os.path.join(log_dir, "target_eval_every_epoch.csv")
    source_eval_path = os.path.join(log_dir, "source_eval_every_epoch.csv")

    start_epoch = 1
    last_model_path = os.path.join(model_dir, "last_model.pth")
    if os.path.exists(last_model_path):
        print(f"[*] Found checkpoint {last_model_path}. Resuming...")
        ckpt = safe_torch_load(last_model_path, map_location=device)
        for name, module in modules.items():
            module.load_state_dict(ckpt["model_state_dict"][name])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        set_rng_state(ckpt.get("rng_state", None))
        print(f"[*] Resumed from epoch {ckpt['epoch']}, best acc: {0.0:.4f}")
    else:
        init_csv(epoch_log_path, EPOCH_LOG_FIELDS)
        init_csv(target_eval_path, TARGET_EVAL_FIELDS)
        init_csv(source_eval_path, SOURCE_EVAL_FIELDS)
        
    print("=" * 60)
    print(f"[{METHOD}] DSAN GỐC - GHÉP BATCH + WARMUP LOCK")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        seed_epoch(args.seed, epoch)
        lambda_p   = lambda_schedule(epoch, args.epochs, warmup=args.warmup)
        current_lr = adjust_lr(optimizer, epoch, args.epochs, lr0=args.lr, bb_factor=BACKBONE_LR_FACTOR)

        F_ext.train(); bottleneck.train(); classifier.train()
        tgt_iter = InfiniteIterator(tgt_train_loader)

        epoch_cls_loss = 0.0; epoch_lmmd_loss = 0.0; epoch_total_loss = 0.0
        correct = 0; total = 0

        for src_imgs, src_labels in tqdm(src_train_loader, desc=f'Epoch [{epoch:03d}/{args.epochs}]', leave=False, dynamic_ncols=True):
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)

            if lambda_p <= 0.0:
                # WARMUP LOCK
                src_feat   = F_ext(src_imgs)
                src_bn     = bottleneck(src_feat)
                src_logits = classifier(src_bn)
                
                cls_loss   = ce_loss_fn(src_logits, src_labels)
                lmmd_loss  = torch.zeros((), device=device)
                total_loss = cls_loss
            else:
                # DA PHASE - GHÉP BATCH
                tgt_batch = next(tgt_iter)
                tgt_imgs  = images_from_batch(tgt_batch).to(device)

                # Nối ảnh
                concat_imgs = torch.cat((src_imgs, tgt_imgs), dim=0)

                # Đi qua mạng 1 lần
                concat_feat   = F_ext(concat_imgs)
                concat_bn     = bottleneck(concat_feat)
                concat_logits = classifier(concat_bn)

                # Cắt kết quả
                bs_src = src_imgs.size(0)
                src_bn, tgt_bn = concat_bn[:bs_src], concat_bn[bs_src:]
                src_logits, tgt_logits = concat_logits[:bs_src], concat_logits[bs_src:]

                cls_loss   = ce_loss_fn(src_logits, src_labels)

                tgt_probs = F.softmax(tgt_logits, dim=1).detach()
                lmmd_loss = compute_lmmd(src_bn, tgt_bn, src_labels, tgt_probs)

                total_loss = cls_loss + lambda_p * lmmd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bs = src_labels.size(0)
            epoch_cls_loss    += cls_loss.item() * bs
            epoch_lmmd_loss   += lmmd_loss.item() * bs
            epoch_total_loss  += total_loss.item() * bs
            correct += (src_logits.argmax(1) == src_labels).sum().item()
            total   += bs

        epoch_cls_loss    /= max(total, 1)
        epoch_lmmd_loss   /= max(total, 1)
        epoch_total_loss  /= max(total, 1)
        src_train_acc      = correct / max(total, 1)
        elapsed            = time.time() - t0

        src_val = evaluate(F_ext, bottleneck, classifier, src_val_loader, device)
        tgt_mon = evaluate(F_ext, bottleneck, classifier, tgt_test_loader, device)

        ckpt_state = build_base_ckpt(METHOD, epoch, modules, optimizer, args, 0.0, args.lr, current_lr, args.epochs, args.warmup, extra={"lambda_lmmd": lambda_p})
        if epoch % 10 == 0:
            save_training_state(ckpt_state, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        save_training_state(ckpt_state, os.path.join(model_dir, "last_model.pth"))

        losses = {"total": epoch_total_loss, "cls": epoch_cls_loss, "domain": 0.0, "lmmd": epoch_lmmd_loss}
        print_epoch(epoch, args.epochs, METHOD, losses, src_val, tgt_mon, lam=lambda_p, elapsed=elapsed)

        append_csv(epoch_log_path, {
            "epoch": epoch, "method": METHOD, "seed": args.seed, "batch_size": args.batch_size, "learning_rate": f"{current_lr:.6f}",
            "warmup_status": "WARMUP_LOCK" if lambda_p <= 0 else "DA_ACTIVE",
            "lambda_adv": "0", "lambda_lmmd": f"{lambda_p:.6f}", "train_total_loss": f"{epoch_total_loss:.6f}",
            "source_cls_loss": f"{epoch_cls_loss:.6f}", "domain_loss": "0", "lmmd_loss": f"{epoch_lmmd_loss:.6f}", "mcc_loss": "0", "entropy_weight_mean": "NA",
            "source_train_acc": f"{src_train_acc:.6f}", "source_val_acc": f"{src_val['acc']:.6f}", "source_val_loss": f"{src_val['loss']:.6f}",
            "target_monitor_acc": f"{tgt_mon['acc']:.6f}", "target_monitor_precision": f"{tgt_mon['precision']:.6f}",
            "target_monitor_recall": f"{tgt_mon['recall']:.6f}", "target_monitor_f1": f"{tgt_mon['f1']:.6f}",
            "target_monitor_macro_f1": f"{tgt_mon['macro_f1']:.6f}", "target_monitor_entropy": f"{tgt_mon['entropy']:.6f}",
            "target_monitor_confidence": f"{tgt_mon['confidence']:.6f}",
            "target_monitor_fresh_acc": f"{tgt_mon['fresh_acc']:.6f}",
            "target_monitor_rotten_acc": f"{tgt_mon['rotten_acc']:.6f}",
            "gpu_memory_mb": f"{get_gpu_memory_mb():.1f}",
            "time_per_epoch_sec": f"{elapsed:.1f}", "checkpoint_path": os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth") if epoch % 10 == 0 else ""
        }, EPOCH_LOG_FIELDS)
        append_csv(target_eval_path, {
            "epoch": epoch, "method": METHOD, "target_acc": f"{tgt_mon['acc']:.6f}", "target_precision": f"{tgt_mon['precision']:.6f}",
            "target_recall": f"{tgt_mon['recall']:.6f}", "target_f1": f"{tgt_mon['f1']:.6f}", "target_macro_f1": f"{tgt_mon['macro_f1']:.6f}",
            "target_fresh_acc": f"{tgt_mon['fresh_acc']:.6f}", "target_rotten_acc": f"{tgt_mon['rotten_acc']:.6f}",
            "target_entropy": f"{tgt_mon['entropy']:.6f}", "target_confidence": f"{tgt_mon['confidence']:.6f}",
            "tn": tgt_mon["tn"], "fp": tgt_mon["fp"], "fn": tgt_mon["fn"], "tp": tgt_mon["tp"], "note": ""
        }, TARGET_EVAL_FIELDS)
        append_csv(source_eval_path, {
            "epoch": epoch, "method": METHOD, "source_train_acc": f"{src_train_acc:.6f}", "source_val_acc": f"{src_val['acc']:.6f}",
            "source_val_loss": f"{src_val['loss']:.6f}", "source_test_acc_optional": "NA", "source_precision": f"{src_val['precision']:.6f}",
            "source_recall": f"{src_val['recall']:.6f}", "source_f1": f"{src_val['f1']:.6f}", "source_macro_f1": f"{src_val['macro_f1']:.6f}"
        }, SOURCE_EVAL_FIELDS)

    # FINAL EVALUATION (LUÔN DÙNG LAST MODEL THEO YÊU CẦU CỦA USER)
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION — {METHOD}")
    print("=" * 60)
    last_path = os.path.join(model_dir, "last_model.pth")
    checkpoint_used = last_path
    checkpoint_type = "last_epoch"
    print(f"[*] Using last_model.pth for final evaluation.")
    
    ckpt = safe_torch_load(checkpoint_used, map_location=device)
    for name, module in modules.items(): module.load_state_dict(ckpt["model_state_dict"][name])

    src_test  = evaluate(F_ext, bottleneck, classifier, src_test_loader, device)
    tgt_final = evaluate(F_ext, bottleneck, classifier, tgt_test_loader, device)
    final_metrics = {
        "method": METHOD, "checkpoint_type": checkpoint_type,
        "selected_epoch": int(ckpt.get("epoch", -1)),
        "source_test_acc": src_test["acc"], "target_test_acc": tgt_final["acc"],
        "target_test_f1": tgt_final["f1"], "target_test_macro_f1": tgt_final["macro_f1"],
        "domain_gap_acc": src_test["acc"] - tgt_final["acc"],
        "confusion_matrix": {
            "target_test": {"tn": tgt_final["tn"], "fp": tgt_final["fp"],
                            "fn": tgt_final["fn"], "tp": tgt_final["tp"]},
        },
    }
    save_json(final_metrics, os.path.join(log_dir, "final_test_metrics.json"))
    print(f"[RESULT] epoch={final_metrics['selected_epoch']} | "
          f"Source Test Acc={src_test['acc']:.4f} | Target Test Acc={tgt_final['acc']:.4f}")

if __name__ == "__main__":
    main()
