


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

# =========================================================
# Gradient Reversal Layer
# =========================================================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# =========================================================
# Source Dataset
# =========================================================
class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
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
# DAAN MobileNetV3 Small
# =========================================================
class DAANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.label_predictor = nn.Linear(feat_dim, num_classes)

        # Global domain classifier
        self.global_domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2)
        )

        # Local domain classifiers, mỗi class có 1 discriminator riêng
        self.local_domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 2)
            )
            for _ in range(num_classes)
        ])

    def extract_features(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        class_out = self.label_predictor(feat)
        return class_out, feat


# =========================================================
# DAAN Local Domain Loss
# =========================================================
def local_domain_loss(
    model,
    feat_s,
    feat_t,
    prob_s,
    prob_t,
    domain_labels_s,
    domain_labels_t,
    lambda_grl,
    criterion_domain
):
    total_local_loss = 0.0
    total_local_correct = 0.0
    total_local_weight = 0.0

    for c in range(NUM_CLASSES):
        local_classifier = model.local_domain_classifiers[c]

        weight_s = prob_s[:, c].detach()
        weight_t = prob_t[:, c].detach()

        pred_s = local_classifier(grad_reverse(feat_s, lambda_grl))
        pred_t = local_classifier(grad_reverse(feat_t, lambda_grl))

        loss_s = F.cross_entropy(
            pred_s,
            domain_labels_s,
            reduction="none"
        )

        loss_t = F.cross_entropy(
            pred_t,
            domain_labels_t,
            reduction="none"
        )

        weighted_loss_s = weight_s * loss_s
        weighted_loss_t = weight_t * loss_t

        class_loss = (
            weighted_loss_s.mean() +
            weighted_loss_t.mean()
        ) / 2.0

        total_local_loss += class_loss

        with torch.no_grad():
            pred_s_label = pred_s.argmax(dim=1)
            pred_t_label = pred_t.argmax(dim=1)

            correct_s = (pred_s_label == domain_labels_s).float() * weight_s
            correct_t = (pred_t_label == domain_labels_t).float() * weight_t

            total_local_correct += correct_s.sum().item()
            total_local_correct += correct_t.sum().item()

            total_local_weight += weight_s.sum().item()
            total_local_weight += weight_t.sum().item()

    total_local_loss = total_local_loss / NUM_CLASSES

    if total_local_weight > 0:
        local_acc = total_local_correct / total_local_weight
    else:
        local_acc = 0.5

    return total_local_loss, local_acc


# =========================================================
# Dynamic mu update
# =========================================================
def update_mu(global_acc, local_acc):
    global_distance = max(0.0, 2.0 * global_acc - 1.0)
    local_distance = max(0.0, 2.0 * local_acc - 1.0)

    total_distance = global_distance + local_distance

    if total_distance == 0:
        return 0.5

    mu = local_distance / total_distance
    mu = float(np.clip(mu, 0.0, 1.0))

    return mu


# =========================================================
# Evaluate Target
# Không update gradient
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

    mean_entropy = total_entropy / total_samples if total_samples > 0 else 1e-8
    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Device + Model + Optimizer
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DAANMobileV3Small(num_classes=NUM_CLASSES).to(device)

# Nếu có model train sẵn thì mở 2 dòng này
# pretrained_path = "your_model.pth"
# model.load_state_dict(torch.load(pretrained_path, map_location=device))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()


# =========================================================
# Training Loop + Early Stopping theo lowest entropy
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

mu = 0.5

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_global_loss = 0.0
    total_local_loss = 0.0
    total_adv_loss = 0.0
    total_loss = 0.0

    total_global_correct = 0
    total_global_samples = 0

    total_local_acc = 0.0
    total_batches = 0

    # GRL lambda schedule
    p = epoch / EPOCHS
    lambda_grl = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    loop = tqdm(
        zip(source_loader, target_loader),
        total=min(len(source_loader), len(target_loader)),
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for (x_s, y_s), (x_t, _) in loop:
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        bs_s = x_s.size(0)
        bs_t = x_t.size(0)

        domain_labels_s = torch.zeros(bs_s).long().to(device)
        domain_labels_t = torch.ones(bs_t).long().to(device)

        # =========================
        # Forward source + target
        # =========================
        class_pred_s, feat_s = model(x_s)
        class_pred_t, feat_t = model(x_t)

        # =========================
        # Source classification
        # =========================
        loss_class = criterion_class(class_pred_s, y_s)

        # =========================
        # Global domain adversarial loss
        # =========================
        feat_combined = torch.cat([feat_s, feat_t], dim=0)

        domain_labels = torch.cat(
            [domain_labels_s, domain_labels_t],
            dim=0
        )

        global_domain_pred = model.global_domain_classifier(
            grad_reverse(feat_combined, lambda_grl)
        )

        loss_global = criterion_domain(
            global_domain_pred,
            domain_labels
        )

        with torch.no_grad():
            global_pred_label = global_domain_pred.argmax(dim=1)
            total_global_correct += (global_pred_label == domain_labels).sum().item()
            total_global_samples += domain_labels.size(0)

        # =========================
        # Local domain adversarial loss
        # =========================
        prob_s = F.softmax(class_pred_s, dim=1)
        prob_t = F.softmax(class_pred_t, dim=1)

        loss_local, local_acc = local_domain_loss(
            model=model,
            feat_s=feat_s,
            feat_t=feat_t,
            prob_s=prob_s,
            prob_t=prob_t,
            domain_labels_s=domain_labels_s,
            domain_labels_t=domain_labels_t,
            lambda_grl=lambda_grl,
            criterion_domain=criterion_domain
        )

        # =========================
        # DAAN adversarial loss
        # mu càng cao => ưu tiên local domain loss
        # =========================
        loss_adv = (1.0 - mu) * loss_global + mu * loss_local

        loss = loss_class + loss_adv

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_global_loss += loss_global.item()
        total_local_loss += loss_local.item()
        total_adv_loss += loss_adv.item()
        total_loss += loss.item()
        total_local_acc += local_acc
        total_batches += 1

        loop.set_postfix({
            "class_loss": loss_class.item(),
            "global_loss": loss_global.item(),
            "local_loss": loss_local.item(),
            "mu": mu,
            "lambda": lambda_grl
        })

    avg_class_loss = total_class_loss / total_batches
    avg_global_loss = total_global_loss / total_batches
    avg_local_loss = total_local_loss / total_batches
    avg_adv_loss = total_adv_loss / total_batches
    avg_total_loss = total_loss / total_batches

    global_acc = total_global_correct / total_global_samples
    local_acc = total_local_acc / total_batches

    mu = update_mu(global_acc, local_acc)

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
    print(f"Train class loss  : {avg_class_loss:.4f}")
    print(f"Global loss       : {avg_global_loss:.4f}")
    print(f"Local loss        : {avg_local_loss:.4f}")
    print(f"DAAN adv loss     : {avg_adv_loss:.4f}")
    print(f"Train total loss  : {avg_total_loss:.4f}")
    print(f"Global domain acc : {global_acc:.4f}")
    print(f"Local domain acc  : {local_acc:.4f}")
    print(f"Dynamic mu        : {mu:.4f}")
    print(f"GRL lambda        : {lambda_grl:.4f}")
    print(f"Target entropy    : {mean_entropy:.8f}")
    print(f"Target accuracy   : {target_acc:.4f}")
    print("=" * 70)

    # =====================================================
    # Save best model theo entropy
    # =====================================================
    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"DAAN_MobileV3Small_best_entropy_"
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
    print(f"Final target accuracy : {final_acc:.4f}")

else:
    print("No best model was saved.")


# # chay tiep tuc



# =========================================================
# FULL RESUME DAAN - RUN ONLY THIS CELL AFTER RESTART KERNEL
# Auto find latest/best checkpoint and continue training
# =========================================================
print("begin")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os
import re
import glob
from tqdm import tqdm
import numpy as np

# =========================================================
# Gradient Reversal Layer
# =========================================================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


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
BATCH_SIZE = 8
EPOCHS = 2000

LR = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 2
patience = 20
min_delta = 1e-8

SOURCE_ROOT = "dataset"
TARGET_ROOT = "FruitsOriginal"


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

source_dataset = FruitDataset(SOURCE_ROOT, transform=transform)
target_dataset = FruitVisionSeenFruitDataset(TARGET_ROOT, transform=transform)

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
# DAAN MobileNetV3 Small
# =========================================================
class DAANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.label_predictor = nn.Linear(feat_dim, num_classes)

        self.global_domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2)
        )

        self.local_domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 2)
            )
            for _ in range(num_classes)
        ])

    def extract_features(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        class_out = self.label_predictor(feat)
        return class_out, feat


# =========================================================
# Local Domain Loss
# =========================================================
def local_domain_loss(
    model,
    feat_s,
    feat_t,
    prob_s,
    prob_t,
    domain_labels_s,
    domain_labels_t,
    lambda_grl,
    criterion_domain
):
    total_local_loss = 0.0
    total_local_correct = 0.0
    total_local_weight = 0.0

    for c in range(NUM_CLASSES):
        local_classifier = model.local_domain_classifiers[c]

        weight_s = prob_s[:, c].detach()
        weight_t = prob_t[:, c].detach()

        pred_s = local_classifier(grad_reverse(feat_s, lambda_grl))
        pred_t = local_classifier(grad_reverse(feat_t, lambda_grl))

        loss_s = F.cross_entropy(pred_s, domain_labels_s, reduction="none")
        loss_t = F.cross_entropy(pred_t, domain_labels_t, reduction="none")

        class_loss = ((weight_s * loss_s).mean() + (weight_t * loss_t).mean()) / 2.0

        total_local_loss += class_loss

        with torch.no_grad():
            pred_s_label = pred_s.argmax(dim=1)
            pred_t_label = pred_t.argmax(dim=1)

            correct_s = (pred_s_label == domain_labels_s).float() * weight_s
            correct_t = (pred_t_label == domain_labels_t).float() * weight_t

            total_local_correct += correct_s.sum().item()
            total_local_correct += correct_t.sum().item()

            total_local_weight += weight_s.sum().item()
            total_local_weight += weight_t.sum().item()

    total_local_loss = total_local_loss / NUM_CLASSES
    local_acc = total_local_correct / total_local_weight if total_local_weight > 0 else 0.5

    return total_local_loss, local_acc


def update_mu(global_acc, local_acc):
    global_distance = max(0.0, 2.0 * global_acc - 1.0)
    local_distance = max(0.0, 2.0 * local_acc - 1.0)

    total_distance = global_distance + local_distance

    if total_distance == 0:
        return 0.5

    mu = local_distance / total_distance
    return float(np.clip(mu, 0.0, 1.0))


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

    mean_entropy = total_entropy / total_samples if total_samples > 0 else 1e-8
    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Auto Find Checkpoint
# =========================================================
def find_best_checkpoint():
    files = glob.glob("DAAN_MobileV3Small_best_entropy_epoch57_entropy0.03262191.pth")

    if len(files) == 0:
        raise FileNotFoundError(
            "Không tìm thấy checkpoint DAAN dạng "
            "DAAN_MobileV3Small_best_entropy_epoch*_entropy*.pth"
        )

    parsed = []

    for f in files:
        m = re.search(r"epoch(\d+)_entropy([0-9.]+)\.pth", f)
        if m:
            epoch = int(m.group(1))
            entropy = float(m.group(2))
            parsed.append((entropy, epoch, f))

    if len(parsed) == 0:
        raise ValueError("Có file .pth nhưng không parse được epoch/entropy từ tên file.")

    parsed = sorted(parsed, key=lambda x: x[0])
    best_entropy, best_epoch, best_path = parsed[0]

    return best_path, best_epoch, best_entropy


# =========================================================
# Device + Model + Resume
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DAANMobileV3Small(num_classes=NUM_CLASSES).to(device)

best_model_path, loaded_epoch, best_entropy = find_best_checkpoint()

checkpoint = torch.load(best_model_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint, strict=True)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

START_EPOCH = loaded_epoch
wait = 0
mu = 0.5

print("=" * 70)
print("Resume checkpoint loaded")
print(f"File          : {best_model_path}")
print(f"Loaded epoch  : {loaded_epoch}")
print(f"Continue from : Epoch {START_EPOCH + 1}/{EPOCHS}")
print(f"Best entropy  : {best_entropy:.8f}")
print("=" * 70)


# =========================================================
# Training Loop Resume
# =========================================================
for epoch in range(START_EPOCH, EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_global_loss = 0.0
    total_local_loss = 0.0
    total_adv_loss = 0.0
    total_loss = 0.0

    total_global_correct = 0
    total_global_samples = 0

    total_local_acc = 0.0
    total_batches = 0

    p = epoch / EPOCHS
    lambda_grl = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    loop = tqdm(
        zip(source_loader, target_loader),
        total=min(len(source_loader), len(target_loader)),
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for (x_s, y_s), (x_t, _) in loop:
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        bs_s = x_s.size(0)
        bs_t = x_t.size(0)

        domain_labels_s = torch.zeros(bs_s).long().to(device)
        domain_labels_t = torch.ones(bs_t).long().to(device)

        class_pred_s, feat_s = model(x_s)
        class_pred_t, feat_t = model(x_t)

        loss_class = criterion_class(class_pred_s, y_s)

        feat_combined = torch.cat([feat_s, feat_t], dim=0)
        domain_labels = torch.cat([domain_labels_s, domain_labels_t], dim=0)

        global_domain_pred = model.global_domain_classifier(
            grad_reverse(feat_combined, lambda_grl)
        )

        loss_global = criterion_domain(global_domain_pred, domain_labels)

        with torch.no_grad():
            global_pred_label = global_domain_pred.argmax(dim=1)
            total_global_correct += (global_pred_label == domain_labels).sum().item()
            total_global_samples += domain_labels.size(0)

        prob_s = F.softmax(class_pred_s, dim=1)
        prob_t = F.softmax(class_pred_t, dim=1)

        loss_local, local_acc = local_domain_loss(
            model=model,
            feat_s=feat_s,
            feat_t=feat_t,
            prob_s=prob_s,
            prob_t=prob_t,
            domain_labels_s=domain_labels_s,
            domain_labels_t=domain_labels_t,
            lambda_grl=lambda_grl,
            criterion_domain=criterion_domain
        )

        loss_adv = (1.0 - mu) * loss_global + mu * loss_local
        loss = loss_class + loss_adv

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_global_loss += loss_global.item()
        total_local_loss += loss_local.item()
        total_adv_loss += loss_adv.item()
        total_loss += loss.item()
        total_local_acc += local_acc
        total_batches += 1

        loop.set_postfix({
            "class": loss_class.item(),
            "global": loss_global.item(),
            "local": loss_local.item(),
            "mu": mu,
            "lambda": lambda_grl
        })

    avg_class_loss = total_class_loss / total_batches
    avg_global_loss = total_global_loss / total_batches
    avg_local_loss = total_local_loss / total_batches
    avg_adv_loss = total_adv_loss / total_batches
    avg_total_loss = total_loss / total_batches

    global_acc = total_global_correct / total_global_samples
    local_acc = total_local_acc / total_batches

    mu = update_mu(global_acc, local_acc)

    mean_entropy, target_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss  : {avg_class_loss:.4f}")
    print(f"Global loss       : {avg_global_loss:.4f}")
    print(f"Local loss        : {avg_local_loss:.4f}")
    print(f"DAAN adv loss     : {avg_adv_loss:.4f}")
    print(f"Train total loss  : {avg_total_loss:.4f}")
    print(f"Global domain acc : {global_acc:.4f}")
    print(f"Local domain acc  : {local_acc:.4f}")
    print(f"Dynamic mu        : {mu:.4f}")
    print(f"GRL lambda        : {lambda_grl:.4f}")
    print(f"Target entropy    : {mean_entropy:.8f}")
    print(f"Target accuracy   : {target_acc:.4f}")
    print("=" * 70)

    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"DAAN_MobileV3Small_best_entropy_"
            f"epoch{epoch + 1}_entropy{best_entropy:.8f}.pth"
        )

        torch.save(model.state_dict(), best_model_path)

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
        torch.load(best_model_path, map_location=device)
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






