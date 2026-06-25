


print("begin")
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models


# =========================================================
# Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# Config
# =========================================================
SOURCE_ROOT = "dataset"
TARGET_ROOT = "FruitsOriginal"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2000

LR = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 2

SUBDOMAIN_WEIGHT = 1.0
UNCERTAINTY_WEIGHT = 0.05

PATIENCE = 20
MIN_DELTA = 1e-6

TARGET_TEST_RATIO = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Dataset
# =========================================================
class SourceFruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for subdir, _, files in os.walk(root_dir):
            subdir_lower = subdir.lower()

            if "fresh" in subdir_lower:
                label = 0
            elif "rotten" in subdir_lower:
                label = 1
            else:
                continue

            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(subdir, file), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No source images found in {root_dir}")

        print(f"Source images found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


class TargetFruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        allowed_fruits = {"apple", "banana", "orange"}
        allowed_quality = {"fresh": 0, "rotten": 1}

        root_dir = Path(root_dir)

        for fruit_folder in sorted(root_dir.iterdir()):
            if not fruit_folder.is_dir():
                continue

            fruit_name = fruit_folder.name.lower()
            fruit_name = fruit_name.replace(" ", "").replace("_", "").replace("-", "")

            if fruit_name not in allowed_fruits:
                continue

            for quality_folder in sorted(fruit_folder.iterdir()):
                if not quality_folder.is_dir():
                    continue

                quality_name = quality_folder.name.lower()
                quality_name = quality_name.replace(" ", "").replace("_", "").replace("-", "")

                if quality_name not in allowed_quality:
                    continue

                label = allowed_quality[quality_name]

                for img_path in quality_folder.rglob("*"):
                    if img_path.suffix.lower() in {
                        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                    }:
                        self.samples.append((str(img_path), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No target images found in {root_dir}")

        print(f"Target images found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def split_dataset(dataset, test_ratio=0.2, seed=42):
    indices = list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_size = max(1, int(len(indices) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    if len(train_indices) == 0:
        raise RuntimeError("Target train split is empty. Reduce TARGET_TEST_RATIO.")

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


# =========================================================
# Transform + DataLoader
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

source_dataset = SourceFruitDataset(SOURCE_ROOT, transform=transform)
target_full_dataset = TargetFruitDataset(TARGET_ROOT, transform=transform)

target_train_dataset, target_test_dataset = split_dataset(
    target_full_dataset,
    test_ratio=TARGET_TEST_RATIO,
    seed=SEED
)

print(f"Target train/adaptation images: {len(target_train_dataset)}")
print(f"Target test/evaluation images : {len(target_test_dataset)}")

source_loader = DataLoader(
    source_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

target_train_loader = DataLoader(
    target_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

target_entropy_loader = DataLoader(
    target_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

target_test_loader = DataLoader(
    target_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0
)


# =========================================================
# Multi-kernel Gaussian Kernel
# =========================================================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))

    l2_distance = ((total0 - total1) ** 2).sum(dim=2)

    if fix_sigma is not None:
        bandwidth = torch.tensor(
            fix_sigma,
            device=source.device,
            dtype=source.dtype
        )
    else:
        denom = max(n_samples ** 2 - n_samples, 1)
        bandwidth = torch.sum(l2_distance.detach()) / denom

    bandwidth = torch.clamp(bandwidth, min=1e-8)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = 0.0
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        kernels = kernels + torch.exp(-l2_distance / bw)

    return kernels


# =========================================================
# LMMD Loss
# =========================================================
def lmmd_loss(source_feat, target_feat, source_label, target_logits, num_classes):
    batch_size = source_feat.size(0)

    kernels = gaussian_kernel(source_feat, target_feat)

    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    source_onehot = F.one_hot(source_label, num_classes=num_classes).float()

    target_prob = F.softmax(target_logits.detach(), dim=1)
    target_prob = torch.clamp(target_prob, min=1e-8, max=1.0)

    loss = torch.tensor(0.0, device=source_feat.device)
    valid_class_count = 0

    for c in range(num_classes):
        s_c = source_onehot[:, c]
        t_c = target_prob[:, c]

        if s_c.sum().item() < 1e-6 or t_c.sum().item() < 1e-6:
            continue

        s_c = s_c / (s_c.sum() + 1e-8)
        t_c = t_c / (t_c.sum() + 1e-8)

        ss = torch.sum(SS * torch.outer(s_c, s_c))
        tt = torch.sum(TT * torch.outer(t_c, t_c))
        st = torch.sum(ST * torch.outer(s_c, t_c))

        loss = loss + ss + tt - 2.0 * st
        valid_class_count += 1

    if valid_class_count > 0:
        loss = loss / valid_class_count

    return torch.nan_to_num(loss, nan=1e-8, posinf=1e-8, neginf=1e-8)


# =========================================================
# Entropy / Uncertainty
# =========================================================
def entropy_loss(logits):
    prob = F.softmax(logits, dim=1)
    prob = torch.clamp(prob, min=1e-8, max=1.0)

    entropy = -(prob * torch.log(prob)).sum(dim=1)
    entropy = torch.nan_to_num(
        entropy,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return entropy.mean()


def adaptation_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


# =========================================================
# Multi-Representation Module
# =========================================================
class MultiRepresentationModule(nn.Module):
    def __init__(self, in_channels=576, out_channels=128):
        super().__init__()

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        r1 = torch.flatten(self.avgpool(self.branch_1x1(x)), 1)
        r2 = torch.flatten(self.avgpool(self.branch_3x3(x)), 1)
        r3 = torch.flatten(self.avgpool(self.branch_5x5(x)), 1)
        r4 = torch.flatten(self.avgpool(self.branch_pool(x)), 1)

        concat = torch.cat([r1, r2, r3, r4], dim=1)

        return [r1, r2, r3, r4], concat


# =========================================================
# MSUN MobileNetV3 Small
# =========================================================
class MSUNMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features

        self.multi_rep = MultiRepresentationModule(
            in_channels=576,
            out_channels=128
        )

        feat_dim = 128 * 4

        self.label_predictor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat_map = self.feature_extractor(x)
        reps, concat_feat = self.multi_rep(feat_map)
        logits = self.label_predictor(concat_feat)

        return logits, reps, concat_feat


# =========================================================
# MSUN Loss
# =========================================================
def msun_loss(reps_s, reps_t, concat_s, concat_t, y_s, logits_t):
    multi_rep_lmmd = torch.tensor(0.0, device=concat_s.device)

    for rs, rt in zip(reps_s, reps_t):
        multi_rep_lmmd = multi_rep_lmmd + lmmd_loss(
            source_feat=rs,
            target_feat=rt,
            source_label=y_s,
            target_logits=logits_t,
            num_classes=NUM_CLASSES
        )

    multi_rep_lmmd = multi_rep_lmmd / len(reps_s)

    concat_lmmd = lmmd_loss(
        source_feat=concat_s,
        target_feat=concat_t,
        source_label=y_s,
        target_logits=logits_t,
        num_classes=NUM_CLASSES
    )

    subdomain_loss = multi_rep_lmmd + concat_lmmd
    uncertainty = entropy_loss(logits_t)

    total = SUBDOMAIN_WEIGHT * subdomain_loss + UNCERTAINTY_WEIGHT * uncertainty
    total = torch.nan_to_num(total, nan=1e-8, posinf=1e-8, neginf=1e-8)

    return total, subdomain_loss, uncertainty


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_entropy(model, loader, device):
    """
    Dùng để chọn model.
    Không dùng target label.
    """
    model.eval()

    total_entropy = 0.0
    total_samples = 0

    for batch in loader:
        x = batch[0].to(device)

        logits, _, _ = model(x)
        ent = entropy_loss(logits)

        total_entropy += ent.item() * x.size(0)
        total_samples += x.size(0)

    return total_entropy / max(total_samples, 1)


@torch.no_grad()
def evaluate_target_acc_for_print_only(model, loader, device):
    """
    Chỉ in ra để xem.
    KHÔNG dùng để train.
    KHÔNG dùng để save model.
    KHÔNG dùng để early stopping.
    """
    model.eval()

    correct = 0
    total = 0
    total_entropy = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _, _ = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        total_entropy += entropy_loss(logits).item() * x.size(0)

    acc = correct / max(total, 1)
    mean_entropy = total_entropy / max(total, 1)

    return mean_entropy, acc


# =========================================================
# Train
# =========================================================
print("Device:", DEVICE)

model = MSUNMobileV3Small(num_classes=NUM_CLASSES).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()

best_entropy = float("inf")
best_model_path = "MSUN_MobileV3Small_best_entropy.pth"
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_msun_loss = 0.0
    total_subdomain_loss = 0.0
    total_uncertainty_loss = 0.0
    total_train_loss = 0.0
    total_batches = 0

    lam = adaptation_lambda(epoch, EPOCHS)

    loop = tqdm(
        zip(source_loader, target_train_loader),
        total=min(len(source_loader), len(target_train_loader)),
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for (x_s, y_s), target_batch in loop:
        x_t = target_batch[0]

        x_s = x_s.to(DEVICE)
        y_s = y_s.to(DEVICE)
        x_t = x_t.to(DEVICE)

        logits_s, reps_s, concat_s = model(x_s)
        logits_t, reps_t, concat_t = model(x_t)

        class_loss = criterion_class(logits_s, y_s)

        adapt_loss, subdomain_loss, uncertainty = msun_loss(
            reps_s=reps_s,
            reps_t=reps_t,
            concat_s=concat_s,
            concat_t=concat_t,
            y_s=y_s,
            logits_t=logits_t
        )

        loss = class_loss + lam * adapt_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_class_loss += class_loss.item()
        total_msun_loss += adapt_loss.item()
        total_subdomain_loss += subdomain_loss.item()
        total_uncertainty_loss += uncertainty.item()
        total_train_loss += loss.item()
        total_batches += 1

        loop.set_postfix({
            "cls": f"{class_loss.item():.4f}",
            "msun": f"{adapt_loss.item():.4f}",
            "lambda": f"{lam:.4f}",
            "total": f"{loss.item():.4f}"
        })

    if total_batches == 0:
        raise RuntimeError(
            "No training batches. Reduce BATCH_SIZE or set drop_last=False."
        )

    avg_class = total_class_loss / total_batches
    avg_msun = total_msun_loss / total_batches
    avg_subdomain = total_subdomain_loss / total_batches
    avg_uncertainty = total_uncertainty_loss / total_batches
    avg_total = total_train_loss / total_batches

    # =====================================================
    # Unsupervised metric: dùng để save model
    # =====================================================
    target_entropy = evaluate_entropy(
        model,
        target_entropy_loader,
        DEVICE
    )

    # =====================================================
    # Target accuracy: chỉ in ra để xem
    # KHÔNG dùng để save / train / early stopping
    # =====================================================
    target_print_entropy, target_print_acc = evaluate_target_acc_for_print_only(
        model,
        target_test_loader,
        DEVICE
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss        : {avg_class:.6f}")
    print(f"Train MSUN loss         : {avg_msun:.6f}")
    print(f"Subdomain LMMD loss     : {avg_subdomain:.6f}")
    print(f"Uncertainty loss        : {avg_uncertainty:.6f}")
    print(f"Lambda adaptation       : {lam:.6f}")
    print(f"Train total loss        : {avg_total:.6f}")
    print(f"Target train entropy    : {target_entropy:.8f}  # used for saving, no labels")
    print(f"Target test entropy     : {target_print_entropy:.8f}  # print only")
    print(f"Target test accuracy    : {target_print_acc:.4f}  # print only, not used")
    print("=" * 70)

    # =====================================================
    # Save model: chỉ dựa vào target_entropy
    # KHÔNG dùng target_print_acc
    # =====================================================
    if target_entropy < best_entropy - MIN_DELTA:
        best_entropy = target_entropy
        wait = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_entropy": best_entropy,
                "config": {
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "subdomain_weight": SUBDOMAIN_WEIGHT,
                    "uncertainty_weight": UNCERTAINTY_WEIGHT,
                    "target_test_ratio": TARGET_TEST_RATIO,
                    "seed": SEED,
                }
            },
            best_model_path
        )

        print(f"Saved best model: epoch={epoch + 1}, entropy={best_entropy:.8f}")

    else:
        wait += 1
        print(f"No entropy improvement: {wait}/{PATIENCE}")

    if wait >= PATIENCE:
        print("\nEarly stopping triggered.")
        print(f"Stopped at epoch: {epoch + 1}")
        print(f"Best target train entropy: {best_entropy:.8f}")
        print(f"Best model file: {best_model_path}")
        break


# =========================================================
# Final target evaluation
# =========================================================
if Path(best_model_path).exists():
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_entropy, final_acc = evaluate_target_acc_for_print_only(
        model,
        target_test_loader,
        DEVICE
    )

    print("\nFinal Target Test Result")
    print(f"Best model file        : {best_model_path}")
    print(f"Selected epoch         : {checkpoint['epoch']}")
    print(f"Selection entropy      : {checkpoint['best_entropy']:.8f}")
    print(f"Target test entropy    : {final_entropy:.8f}")
    print(f"Target test accuracy   : {final_acc:.4f}  # final evaluation only")

else:
    print("No best model was saved.")






