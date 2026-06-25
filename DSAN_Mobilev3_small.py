


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np


# =========================================================
# Source Dataset
# =========================================================
class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    path = os.path.join(subdir, file)

                    if "fresh" in subdir.lower():
                        label = 0
                    elif "rotten" in subdir.lower():
                        label = 1
                    else:
                        continue

                    self.samples.append((path, label))

        print(f"Source images found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# =========================================================
# Target Dataset
# =========================================================
class FruitVisionSeenFruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        allowed_fruits = ["apple", "banana", "orange"]
        allowed_quality = {
            "fresh": 0,
            "rotten": 1
        }

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
                        self.samples.append((img_path, label))

        print(f"Target images found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


# =========================================================
# Config
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2000

LR = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 2
LMMD_WEIGHT = 1.0

patience = 20
min_delta = 1e-8


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

source_dataset = FruitDataset(
    root_dir="dataset",
    transform=transform
)

target_dataset = FruitVisionSeenFruitDataset(
    root_dir="FruitsOriginal",
    transform=transform
)

source_loader = DataLoader(
    source_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

target_loader = DataLoader(
    target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

target_test_loader = DataLoader(
    target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)


# =========================================================
# DSAN MobileNetV3 Small
# =========================================================
class DSANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.label_predictor = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat_flat = torch.flatten(feat, 1)

        class_out = self.label_predictor(feat_flat)

        return class_out, feat_flat


# =========================================================
# Gaussian Kernel
# =========================================================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(
        n_samples,
        n_samples,
        total.size(1)
    )

    total1 = total.unsqueeze(1).expand(
        n_samples,
        n_samples,
        total.size(1)
    )

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (
            n_samples ** 2 - n_samples
        )

    bandwidth = torch.clamp(bandwidth, min=1e-8)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    bandwidth_list = [
        bandwidth * (kernel_mul ** i)
        for i in range(kernel_num)
    ]

    kernel_val = [
        torch.exp(-l2_distance / bw)
        for bw in bandwidth_list
    ]

    return sum(kernel_val)


# =========================================================
# LMMD Loss for DSAN
# Source dùng label thật
# Target dùng pseudo-label probability
# KHÔNG dùng target label để train
# =========================================================
def lmmd_loss(
    source_feat,
    target_feat,
    source_label,
    target_logits,
    num_classes=2,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None
):
    batch_size = source_feat.size(0)

    kernels = gaussian_kernel(
        source_feat,
        target_feat,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    source_onehot = F.one_hot(
        source_label,
        num_classes=num_classes
    ).float()

    # DSAN: target dùng prediction làm soft pseudo-label
    # detach để pseudo-label chỉ làm weight, không dùng target label thật
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

        ss_loss = torch.sum(
            SS * torch.outer(s_c, s_c)
        )

        tt_loss = torch.sum(
            TT * torch.outer(t_c, t_c)
        )

        st_loss = torch.sum(
            ST * torch.outer(s_c, t_c)
        )

        class_lmmd = ss_loss + tt_loss - 2.0 * st_loss

        loss += class_lmmd
        valid_class_count += 1

    if valid_class_count > 0:
        loss = loss / valid_class_count

    loss = torch.nan_to_num(
        loss,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return loss


# =========================================================
# Lambda Schedule
# LMMD tăng dần trong quá trình train
# =========================================================
def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


# =========================================================
# Device + Model + Optimizer
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DSANMobileV3Small(num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()


# =========================================================
# Evaluate Target
# Không update gradient
# Target label chỉ dùng để test/debug
# =========================================================
def evaluate_target(model, target_loader, device):
    model.eval()

    correct = 0
    total = 0

    total_entropy = 0.0
    total_samples = 0

    eps = 1e-8

    with torch.no_grad():
        for x_t, y_t in target_loader:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

            class_pred, _ = model(x_t)

            probs = F.softmax(class_pred, dim=1)

            probs = torch.clamp(probs, min=eps, max=1.0)

            entropy = -(probs * torch.log(probs)).sum(dim=1)

            entropy = torch.nan_to_num(
                entropy,
                nan=1e-8,
                posinf=1e-8,
                neginf=1e-8
            )

            total_entropy += entropy.sum().item()
            total_samples += x_t.size(0)

            preds = class_pred.argmax(dim=1)
            correct += (preds == y_t).sum().item()
            total += y_t.size(0)

    if total_samples == 0:
        mean_entropy = 1e-8
    else:
        mean_entropy = total_entropy / total_samples

    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Training Loop + Early Stopping theo lowest entropy
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_lmmd_loss = 0.0
    total_loss = 0.0
    total_batches = 0

    lmmd_lambda = get_lambda(epoch, EPOCHS) * LMMD_WEIGHT

    loop = tqdm(
        zip(source_loader, target_loader),
        total=min(len(source_loader), len(target_loader)),
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for (x_s, y_s), (x_t, _) in loop:
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        # =========================
        # Source forward
        # =========================
        class_pred_s, feat_s = model(x_s)

        # =========================
        # Target forward
        # Không dùng target label
        # =========================
        class_pred_t, feat_t = model(x_t)

        # =========================
        # Source classification loss
        # =========================
        loss_class = criterion_class(class_pred_s, y_s)

        # =========================
        # DSAN LMMD loss
        # =========================
        loss_lmmd = lmmd_loss(
            source_feat=feat_s,
            target_feat=feat_t,
            source_label=y_s,
            target_logits=class_pred_t,
            num_classes=NUM_CLASSES
        )

        loss = loss_class + lmmd_lambda * loss_lmmd

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_lmmd_loss += loss_lmmd.item()
        total_loss += loss.item()
        total_batches += 1

        loop.set_postfix({
            "class_loss": loss_class.item(),
            "lmmd_loss": loss_lmmd.item(),
            "lambda": lmmd_lambda,
            "total_loss": loss.item()
        })

    if total_batches == 0:
        raise ValueError(
            "No training batches. Reduce BATCH_SIZE or set drop_last=False."
        )

    avg_class_loss = total_class_loss / total_batches
    avg_lmmd_loss = total_lmmd_loss / total_batches
    avg_total_loss = total_loss / total_batches

    # =====================================================
    # Test target sau mỗi epoch
    # Không backward, không update gradient
    # =====================================================
    mean_entropy, target_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train class loss : {avg_class_loss:.4f}")
    print(f"Train LMMD loss  : {avg_lmmd_loss:.8f}")
    print(f"LMMD lambda      : {lmmd_lambda:.6f}")
    print(f"Train total loss : {avg_total_loss:.4f}")
    print(f"Target entropy   : {mean_entropy:.8f}")
    print(f"Target accuracy  : {target_acc:.4f}  # only debug/evaluate")
    print("=" * 70)

    # =====================================================
    # Save best model theo entropy
    # Tên file có epoch + entropy giống code cũ
    # =====================================================
    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"DSAN_MobileV3Small_best_entropy_"
            f"epoch{epoch+1}_entropy{best_entropy:.8f}.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Saved best model at epoch {epoch+1} "
            f"with entropy {best_entropy:.8f}"
        )
        print(f"File: {best_model_path}")

    else:
        wait += 1
        print(f"No entropy improvement: {wait}/{patience}")

    if wait >= patience:
        print("\nEarly stopping triggered.")
        print(f"Stopped at epoch: {epoch+1}")
        print(f"Best entropy: {best_entropy:.8f}")
        print(f"Best model file: {best_model_path}")
        break


# =========================================================
# Load best model cuối cùng
# =========================================================
if best_model_path is not None:
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    final_entropy, final_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\nFinal Best Model Result")
    print(f"Best model file       : {best_model_path}")
    print(f"Best entropy          : {best_entropy:.8f}")
    print(f"Final target entropy  : {final_entropy:.8f}")
    print(f"Final target accuracy : {final_acc:.4f}  # only debug/evaluate")

else:
    print("No best model was saved.")






