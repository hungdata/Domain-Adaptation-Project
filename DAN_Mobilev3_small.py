







import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

# =========================================================
# Dataset Source
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
# Dataset Target
# =========================================================
class FruitVisionSeenFruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        allowed_fruits = ["apple", "banana", "orange"]
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
BATCH_SIZE = 16
EPOCHS = 2000

LR = 1e-4
WEIGHT_DECAY = 1e-5

MMD_WEIGHT = 1.0

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
# MK-MMD Loss for DAN
# =========================================================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth = torch.clamp(bandwidth, min=1e-8)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    bandwidth_list = [
        bandwidth * (kernel_mul ** i)
        for i in range(kernel_num)
    ]

    kernel_val = [
        torch.exp(-l2_distance / torch.clamp(bw, min=1e-8))
        for bw in bandwidth_list
    ]

    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)

    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)

    loss = torch.nan_to_num(
        loss,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return loss


# =========================================================
# DAN MobileNetV3 Small
# Logic:
# Backbone -> FC1 -> FC2 -> Classifier
# MK-MMD align FC1 and FC2
# =========================================================
class DANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_backbone_feature(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        backbone_feat = self.extract_backbone_feature(x)

        fc1_feat = self.fc1(backbone_feat)
        fc2_feat = self.fc2(fc1_feat)

        logits = self.classifier(fc2_feat)

        return logits, fc1_feat, fc2_feat


# =========================================================
# Device + Model + Optimizer
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DANMobileV3Small(num_classes=2).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()


# =========================================================
# Evaluate Target
# Chỉ để quan sát, không update gradient
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

            logits, _, _ = model(x_t)

            probs = F.softmax(logits, dim=1)
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

            preds = logits.argmax(dim=1)
            correct += (preds == y_t).sum().item()
            total += y_t.size(0)

    mean_entropy = total_entropy / total_samples if total_samples > 0 else 1e-8
    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Training Loop DAN
# Source: dùng label
# Target: không dùng label
# MMD: align fc1 + fc2
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_mmd_fc1_loss = 0.0
    total_mmd_fc2_loss = 0.0
    total_mmd_loss = 0.0
    total_loss_value = 0.0
    total_batches = 0

    loop = tqdm(
        zip(source_loader, target_loader),
        total=min(len(source_loader), len(target_loader)),
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for (x_s, y_s), (x_t, _) in loop:
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        logits_s, fc1_s, fc2_s = model(x_s)
        _, fc1_t, fc2_t = model(x_t)

        loss_class = criterion_class(logits_s, y_s)

        loss_mmd_fc1 = mmd_loss(fc1_s, fc1_t)
        loss_mmd_fc2 = mmd_loss(fc2_s, fc2_t)

        loss_mmd_total = loss_mmd_fc1 + loss_mmd_fc2

        loss = loss_class + MMD_WEIGHT * loss_mmd_total

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_mmd_fc1_loss += loss_mmd_fc1.item()
        total_mmd_fc2_loss += loss_mmd_fc2.item()
        total_mmd_loss += loss_mmd_total.item()
        total_loss_value += loss.item()
        total_batches += 1

        loop.set_postfix({
            "class": loss_class.item(),
            "mmd_fc1": loss_mmd_fc1.item(),
            "mmd_fc2": loss_mmd_fc2.item(),
            "total": loss.item()
        })

    avg_class_loss = total_class_loss / total_batches
    avg_mmd_fc1_loss = total_mmd_fc1_loss / total_batches
    avg_mmd_fc2_loss = total_mmd_fc2_loss / total_batches
    avg_mmd_loss = total_mmd_loss / total_batches
    avg_total_loss = total_loss_value / total_batches

    mean_entropy, target_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss   : {avg_class_loss:.4f}")
    print(f"Train MMD fc1 loss : {avg_mmd_fc1_loss:.8f}")
    print(f"Train MMD fc2 loss : {avg_mmd_fc2_loss:.8f}")
    print(f"Train MMD total    : {avg_mmd_loss:.8f}")
    print(f"Train total loss   : {avg_total_loss:.4f}")
    print(f"Target entropy     : {mean_entropy:.8f}")
    print(f"Target accuracy    : {target_acc:.4f}")
    print("=" * 70)

    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"DAN_MobileV3Small_best_entropy_"
            f"epoch{epoch + 1}_entropy{best_entropy:.8f}.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Saved best model at epoch {epoch + 1} "
            f"with entropy {best_entropy:.8f}"
        )
        print(f"File: {best_model_path}")

    else:
        wait += 1
        print(f"No entropy improvement: {wait}/{patience}")

    if wait >= patience:
        print("\nEarly stopping triggered.")
        print(f"Stopped at epoch: {epoch + 1}")
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
    print(f"Final target accuracy : {final_acc:.4f}")

else:
    print("No best model was saved.")


# # Update tieesp tuc chay



# =========================================================
# FULL RESUME CODE DAN - RUN ONLY THIS CELL
# Continue from saved checkpoint epoch 30
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

# =========================================================
# Dataset Source
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
# Dataset Target
# =========================================================
class FruitVisionSeenFruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        allowed_fruits = ["apple", "banana", "orange"]
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
BATCH_SIZE = 16
EPOCHS = 2000

LR = 1e-4
WEIGHT_DECAY = 1e-5
MMD_WEIGHT = 1.0

patience = 20
min_delta = 1e-8

RESUME_PATH = "DAN_MobileV3Small_best_entropy_epoch30_entropy0.23373981.pth"
START_EPOCH = 30
best_entropy = 0.23373981
best_model_path = RESUME_PATH
wait = 0


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
# MK-MMD Loss
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

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth = torch.clamp(bandwidth, min=1e-8)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    bandwidth_list = [
        bandwidth * (kernel_mul ** i)
        for i in range(kernel_num)
    ]

    kernel_val = [
        torch.exp(-l2_distance / torch.clamp(bw, min=1e-8))
        for bw in bandwidth_list
    ]

    return sum(kernel_val)


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)

    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)

    loss = torch.nan_to_num(
        loss,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return loss


# =========================================================
# DAN MobileNetV3 Small
# =========================================================
class DANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(256, num_classes)

    def extract_backbone_feature(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        backbone_feat = self.extract_backbone_feature(x)
        fc1_feat = self.fc1(backbone_feat)
        fc2_feat = self.fc2(fc1_feat)
        logits = self.classifier(fc2_feat)
        return logits, fc1_feat, fc2_feat


# =========================================================
# Evaluate Target
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

            logits, _, _ = model(x_t)

            probs = F.softmax(logits, dim=1)
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

            preds = logits.argmax(dim=1)
            correct += (preds == y_t).sum().item()
            total += y_t.size(0)

    mean_entropy = total_entropy / total_samples if total_samples > 0 else 1e-8
    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Device + Model + Resume
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DANMobileV3Small(num_classes=2).to(device)

model.load_state_dict(
    torch.load(RESUME_PATH, map_location=device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()

print(f"Loaded checkpoint: {RESUME_PATH}")
print(f"Continue from Epoch {START_EPOCH + 1}/{EPOCHS}")
print(f"Best entropy so far: {best_entropy:.8f}")


# =========================================================
# Training Loop Resume
# =========================================================
for epoch in range(START_EPOCH, EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_mmd_fc1_loss = 0.0
    total_mmd_fc2_loss = 0.0
    total_mmd_loss = 0.0
    total_loss_value = 0.0
    total_batches = 0

    loop = tqdm(
        zip(source_loader, target_loader),
        total=min(len(source_loader), len(target_loader)),
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for (x_s, y_s), (x_t, _) in loop:
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        logits_s, fc1_s, fc2_s = model(x_s)
        _, fc1_t, fc2_t = model(x_t)

        loss_class = criterion_class(logits_s, y_s)

        loss_mmd_fc1 = mmd_loss(fc1_s, fc1_t)
        loss_mmd_fc2 = mmd_loss(fc2_s, fc2_t)
        loss_mmd_total = loss_mmd_fc1 + loss_mmd_fc2

        loss = loss_class + MMD_WEIGHT * loss_mmd_total

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_mmd_fc1_loss += loss_mmd_fc1.item()
        total_mmd_fc2_loss += loss_mmd_fc2.item()
        total_mmd_loss += loss_mmd_total.item()
        total_loss_value += loss.item()
        total_batches += 1

        loop.set_postfix({
            "class": loss_class.item(),
            "mmd_fc1": loss_mmd_fc1.item(),
            "mmd_fc2": loss_mmd_fc2.item(),
            "total": loss.item()
        })

    avg_class_loss = total_class_loss / total_batches
    avg_mmd_fc1_loss = total_mmd_fc1_loss / total_batches
    avg_mmd_fc2_loss = total_mmd_fc2_loss / total_batches
    avg_mmd_loss = total_mmd_loss / total_batches
    avg_total_loss = total_loss_value / total_batches

    mean_entropy, target_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss   : {avg_class_loss:.4f}")
    print(f"Train MMD fc1 loss : {avg_mmd_fc1_loss:.8f}")
    print(f"Train MMD fc2 loss : {avg_mmd_fc2_loss:.8f}")
    print(f"Train MMD total    : {avg_mmd_loss:.8f}")
    print(f"Train total loss   : {avg_total_loss:.4f}")
    print(f"Target entropy     : {mean_entropy:.8f}")
    print(f"Target accuracy    : {target_acc:.4f}")
    print("=" * 70)

    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"DAN_MobileV3Small_best_entropy_"
            f"epoch{epoch + 1}_entropy{best_entropy:.8f}.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Saved best model at epoch {epoch + 1} "
            f"with entropy {best_entropy:.8f}"
        )
        print(f"File: {best_model_path}")

    else:
        wait += 1
        print(f"No entropy improvement: {wait}/{patience}")

    if wait >= patience:
        print("\nEarly stopping triggered.")
        print(f"Stopped at epoch: {epoch + 1}")
        print(f"Best entropy: {best_entropy:.8f}")
        print(f"Best model file: {best_model_path}")
        break


# =========================================================
# Load Best Final Model
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
    print(f"Final target accuracy : {final_acc:.4f}")

else:
    print("No best model was saved.")






