#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import csv
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

print("============== [CORE] uda_utils LOADED WITH FIXED LMMD ==============")

SEED = 42
NUM_CLASSES = 2
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 150
WARMUP_EPOCHS = 10

BACKBONE_NAME = "mobilenet_v3_small"
FEATURE_DIM = 576
BOTTLENECK_DIM = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LR = 0.01
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 5e-4

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def configure_determinism(enabled: bool = True):
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

def seed_epoch(base_seed: int, epoch: int):
    seed = int(base_seed) + int(epoch) * 100_003
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_rng_state():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_random_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_random_all"] = None
    return state

def set_rng_state(state):
    if not state:
        return
    try: random.setstate(state["python_random"])
    except: pass
    try: np.random.set_state(state["numpy_random"])
    except: pass
    try: torch.set_rng_state(state["torch_random"])
    except: pass
    if torch.cuda.is_available() and state.get("torch_cuda_random_all") is not None:
        try: torch.cuda.set_rng_state_all(state["torch_cuda_random_all"])
        except: pass

def safe_torch_load(path, map_location):
    try: return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError: return torch.load(path, map_location=map_location)

def atomic_torch_save(obj, path: str):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
    return path

def save_training_state(state: dict, path: str):
    state["rng_state"] = get_rng_state()
    return atomic_torch_save(state, path)

def build_base_ckpt(method_name, epoch, modules, optimizer, args, best_source_val_acc,
                    base_lr, current_lr, schedule_total_epochs, schedule_warmup_epochs, extra=None):
    model_state = {name: module.state_dict() for name, module in modules.items()}
    state = {
        "method_name": method_name,
        "epoch": int(epoch),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_source_val_acc_so_far": float(best_source_val_acc),
        "schedule_total_epochs": int(schedule_total_epochs),
        "warmup_epochs": schedule_warmup_epochs,
        "learning_rate": float(base_lr),
        "deterministic": getattr(args, "deterministic", True),
    }
    if extra: state.update(extra)
    return state

def images_from_batch(batch):
    if isinstance(batch, (list, tuple)): return batch[0]
    return batch

def get_train_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

class TargetUnlabeledDataset(Dataset):

    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class TargetLabeledCSVDataset(Dataset):

    def __init__(self, df, transform=None):
        self.transform = transform
        self.image_paths = df['image_path'].tolist()
        self.labels = df['binary_label'].astype(int).tolist()

        if not self.image_paths:
            raise ValueError("No images found in the provided DataFrame")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_source_train_loader(data_root, batch_size=BATCH_SIZE, num_workers=0):
    ds = ImageFolder(os.path.join(data_root, "source", "train"),
                     transform=get_train_transform())
    print(f"[DATA] Source train : {len(ds)} imgs | classes: {ds.class_to_idx}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), drop_last=True)

def get_source_val_loader(data_root, batch_size=BATCH_SIZE, num_workers=0):
    ds = ImageFolder(os.path.join(data_root, "source", "val"),
                     transform=get_eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

def get_source_test_loader(data_root, batch_size=BATCH_SIZE, num_workers=0):
    ds = ImageFolder(os.path.join(data_root, "source", "test"),
                     transform=get_eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

def get_target_train_loader(data_root, batch_size=BATCH_SIZE, num_workers=0):
    ds = TargetUnlabeledDataset(
        os.path.join(data_root, "target", "train_unlabeled", "images"),
        transform=get_train_transform(),
    )
    print(f"[DATA] Target train (unlabeled): {len(ds)} imgs")
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), drop_last=True)

def get_target_test_loader(data_root, batch_size=BATCH_SIZE, num_workers=0):
    ds = ImageFolder(os.path.join(data_root, "target", "test"),
                     transform=get_eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

def get_target_labeled_train_val_loaders(csv_path, batch_size=BATCH_SIZE, num_workers=0, val_ratio=0.2):

    df = pd.read_csv(csv_path)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    val_size = int(len(df) * val_ratio)
    train_df = df.iloc[val_size:]
    val_df = df.iloc[:val_size]

    train_ds = TargetLabeledCSVDataset(train_df, transform=get_train_transform())
    val_ds = TargetLabeledCSVDataset(val_df, transform=get_eval_transform())

    print(f"[DATA] Oracle Target train: {len(train_ds)} imgs")
    print(f"[DATA] Oracle Target val  : {len(val_ds)} imgs")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return train_loader, val_loader

class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.output_dim = FEATURE_DIM

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

class Bottleneck(nn.Module):

    def __init__(self, in_dim=FEATURE_DIM, out_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.output_dim = out_dim

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ClassifierHead(nn.Module):

    def __init__(self, in_dim=BOTTLENECK_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)

class DomainDiscriminator(nn.Module):

    def __init__(self, in_dim=BOTTLENECK_DIM, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class _GRLFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None

class GRL(nn.Module):

    def forward(self, x, alpha=1.0):
        return _GRLFunc.apply(x, alpha)

def lambda_schedule(epoch, max_epoch, warmup=WARMUP_EPOCHS):

    if epoch <= warmup:
        return 0.0
    p = (epoch - warmup) / (max_epoch - warmup)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

def _lr_factor(epoch, max_epoch, alpha=10.0, beta=0.75):
    p = epoch / max_epoch
    return 1.0 / (1.0 + alpha * p) ** beta

def gaussian_kernel_matrix(source, target,
                           kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    L2 = torch.cdist(total, total, p=2.0) ** 2

    if fix_sigma is not None:
        bw = fix_sigma
    else:
        bw = torch.sum(L2.detach()) / (n * n - n)

    bw /= kernel_mul ** (kernel_num // 2)
    bws = [bw * (kernel_mul ** i) for i in range(kernel_num)]
    K = sum(torch.exp(-L2 / (b + 1e-8)) for b in bws)
    return K

def compute_lmmd(src_feat, tgt_feat, src_labels, tgt_softmax,
                 num_classes=NUM_CLASSES, kernel_mul=2.0, kernel_num=5,
                 fix_sigma=None, tgt_weight=None):

    bs = min(src_feat.size(0), tgt_feat.size(0))
    sf = src_feat[:bs]
    tf = tgt_feat[:bs]
    sl = src_labels[:bs]
    tp = tgt_softmax[:bs]
    if tgt_weight is not None:
        tw = tgt_weight[:bs]
    else:
        tw = None

    K = gaussian_kernel_matrix(sf, tf, kernel_mul, kernel_num, fix_sigma)
    Kss = K[:bs, :bs]
    Ktt = K[bs:, bs:]
    Kst = K[:bs, bs:]

    loss = sf.new_tensor(0.0)
    valid_classes = 0
    for c in range(num_classes):

        sm = (sl == c).float()
        ss = sm.sum()
        if float(ss) < 0.5:
            continue
        ss = ss.clamp(min=1e-8)
        sw = sm / ss

        tc = tp[:, c]
        if tw is not None:
            tc = tc * tw
        
        ts = tc.sum()
        if float(ts) < 0.05:
            continue
            
        ts = ts.clamp(min=1e-8)
        tc_n = tc / ts

        loss = loss + sw @ Kss @ sw
        loss = loss + tc_n @ Ktt @ tc_n
        loss = loss - 2.0 * (sw @ Kst @ tc_n)
        valid_classes += 1

    if valid_classes > 0:
        loss = loss / valid_classes
    else:
        loss = loss * 0.0

    return loss

def cdan_multilinear_map(features, softmax_out):
    B = features.size(0)
    # Match THUML's tensor dimensions: g (softmax) then f (features)
    # Output is torch.bmm(g.unsqueeze(2), f.unsqueeze(1)) -> [B, C, F] -> view [B, C*F]
    return torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1)).view(B, -1)

def entropy_of(softmax_out, eps=1e-8):

    return -(softmax_out * torch.log(softmax_out + eps)).sum(dim=1)

@torch.no_grad()
def evaluate(F_ext, bottleneck, classifier, loader, device):

    F_ext.eval(); bottleneck.eval(); classifier.eval()
    preds_all, labels_all, probs_all = [], [], []
    total_loss, n = 0.0, 0
    ce = nn.CrossEntropyLoss()

    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        feat = F_ext(imgs)
        bn   = bottleneck(feat)
        logits = classifier(bn)
        total_loss += ce(logits, labs).item() * imgs.size(0)
        n += imgs.size(0)
        prob = F.softmax(logits, dim=1)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labs.cpu().numpy())
        probs_all.extend(prob.cpu().numpy())

    y_true = np.array(labels_all)
    y_pred = np.array(preds_all)
    y_prob = np.array(probs_all)

    fm = y_true == 0
    rm = y_true == 1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "loss":       total_loss / max(n, 1),
        "acc":        float(accuracy_score(y_true, y_pred)),
        "fresh_acc":  float(accuracy_score(y_true[fm], y_pred[fm])) if fm.sum() else 0.0,
        "rotten_acc": float(accuracy_score(y_true[rm], y_pred[rm])) if rm.sum() else 0.0,
        "precision":  float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "entropy":    float(-(y_prob * np.log(y_prob + 1e-8)).sum(1).mean()),
        "confidence": float(y_prob.max(1).mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def save_checkpoint(state, directory, filename):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    torch.save(state, path)
    return path

def init_csv(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_csv(path, row, fieldnames):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, restval="").writerow(row)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def build_param_groups(F_ext, bottleneck, classifier,
                       lr=LR, bb_factor=BACKBONE_LR_FACTOR,
                       extra_modules=None):
    groups = [
        {"params": F_ext.parameters(),       "lr": lr * bb_factor, "tag": "backbone"},
        {"params": classifier.parameters(),   "lr": lr,            "tag": "classifier"},
    ]

    bottleneck_params = list(bottleneck.parameters())
    if len(bottleneck_params) > 0:
        groups.append({"params": bottleneck_params, "lr": lr, "tag": "bottleneck"})
    if extra_modules:
        for m in extra_modules:
            groups.append({"params": m.parameters(), "lr": lr,
                           "tag": type(m).__name__})
    return groups

def make_optimizer(param_groups, lr=LR, wd=WEIGHT_DECAY):
    return torch.optim.SGD(param_groups, lr=lr,
                           momentum=0.9, weight_decay=wd, nesterov=True)

def adjust_lr(optimizer, epoch, max_epoch,
              lr0=LR, bb_factor=BACKBONE_LR_FACTOR):
    factor = _lr_factor(epoch, max_epoch)
    for g in optimizer.param_groups:
        if g.get("tag") == "backbone":
            g["lr"] = lr0 * bb_factor * factor
        else:
            g["lr"] = lr0 * factor
    return lr0 * factor

class InfiniteIterator:

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

EPOCH_LOG_FIELDS = [
    "epoch", "method", "seed", "batch_size", "learning_rate",
    "lambda_adv", "lambda_lmmd", "warmup_status",
    "train_total_loss", "source_cls_loss", "domain_loss", "lmmd_loss", "mcc_loss",
    "entropy_weight_mean",
    "source_train_acc",
    "source_val_acc", "source_val_loss",
    "target_monitor_acc", "target_monitor_precision",
    "target_monitor_recall", "target_monitor_f1",
    "target_monitor_macro_f1", "target_monitor_entropy",
    "target_monitor_confidence",
    "target_monitor_fresh_acc", "target_monitor_rotten_acc",
    "gpu_memory_mb",
    "time_per_epoch_sec", "checkpoint_path",
]

TARGET_EVAL_FIELDS = [
    "epoch", "method",
    "target_acc", "target_precision", "target_recall",
    "target_f1", "target_macro_f1",
    "target_fresh_acc", "target_rotten_acc",
    "target_entropy", "target_confidence",
    "tn", "fp", "fn", "tp", "note",
]

SOURCE_EVAL_FIELDS = [
    "epoch", "method",
    "source_train_acc",
    "source_val_acc", "source_val_loss",
    "source_test_acc_optional",
    "source_precision", "source_recall",
    "source_f1", "source_macro_f1",
]

def print_epoch(epoch, mx, method, losses, src_val, tgt, lam=None, elapsed=None):
    print(f"\nEpoch [{epoch:03d}/{mx}] | Method: {method}")
    parts = [f"Total Loss: {losses['total']:.4f}",
             f"Src CE: {losses['cls']:.4f}"]
    for key, label in [("domain", "Domain"), ("lmmd", "LMMD")]:
        v = losses.get(key)
        parts.append(f"{label}: {v:.4f}" if isinstance(v, (int, float)) else f"{label}: NA")
    print(" | ".join(parts))
    if lam is not None:
        print(f"Lambda: {lam:.4f}")
    if src_val:
        print(f"Source Val Acc: {src_val['acc']:.4f}")
    if tgt:
        print(f"Target Test Monitor Acc: {tgt['acc']:.4f} | "
              f"Target Test F1: {tgt['f1']:.4f} | Target Test Entropy: {tgt['entropy']:.4f}")
    if elapsed:
        print(f"Time: {elapsed:.1f}s")
    print("Note: Target Test metrics are for monitoring only, NOT used for model selection (Zero Data Leakage).")

# ── Helper: get GPU memory usage ──
def get_gpu_memory_mb():
    """Lấy GPU memory đang sử dụng (MB). Trả về 0 nếu không có GPU."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


