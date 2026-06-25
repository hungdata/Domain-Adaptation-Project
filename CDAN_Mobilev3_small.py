


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
# Dataset
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


# =========================================================
# DataLoader
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

source_dataset = FruitDataset("dataset", transform=transform)
target_dataset = FruitVisionSeenFruitDataset("FruitsOriginal", transform=transform)

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
# CDAN MobileNetV3 Small
# =========================================================
class CDANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576
        conditional_dim = feat_dim * num_classes

        self.label_predictor = nn.Linear(feat_dim, num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(conditional_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2)
        )

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
# CDAN Conditional Feature
# outer product: feature x class probability
# =========================================================
def conditional_feature(feature, logits):
    prob = F.softmax(logits, dim=1)
    prob = prob.detach()

    op = torch.bmm(
        prob.unsqueeze(2),
        feature.unsqueeze(1)
    )

    op = op.view(feature.size(0), -1)

    return op


# =========================================================
# Entropy Weight for CDAN-E style
# =========================================================
def entropy_weight(logits):
    prob = F.softmax(logits, dim=1)
    prob = torch.clamp(prob, min=1e-8, max=1.0)

    entropy = -(prob * torch.log(prob)).sum(dim=1)
    weight = 1.0 + torch.exp(-entropy)

    weight = weight / weight.mean().detach()

    weight = torch.nan_to_num(
        weight,
        nan=1.0,
        posinf=1.0,
        neginf=1.0
    )

    return weight.detach()


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

model = CDANMobileV3Small(num_classes=NUM_CLASSES).to(device)

# Nếu có model train sẵn thì mở 2 dòng này
# pretrained_path = "your_model.pth"
# model.load_state_dict(torch.load(pretrained_path, map_location=device))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss(reduction="none")


# =========================================================
# Training Loop + Early Stopping theo lowest entropy
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    total_batches = 0

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

        # =========================
        # Source + Target forward
        # =========================
        class_pred_s, feat_s = model(x_s)
        class_pred_t, feat_t = model(x_t)

        # =========================
        # Source classification loss
        # =========================
        loss_class = criterion_class(class_pred_s, y_s)

        # =========================
        # CDAN conditional feature
        # =========================
        cond_s = conditional_feature(feat_s, class_pred_s)
        cond_t = conditional_feature(feat_t, class_pred_t)

        cond_all = torch.cat([cond_s, cond_t], dim=0)

        domain_labels = torch.cat([
            torch.zeros(bs_s),
            torch.ones(bs_t)
        ]).long().to(device)

        domain_pred = model.domain_classifier(
            grad_reverse(cond_all, lambda_grl)
        )

        # =========================
        # CDAN-E entropy weighting
        # =========================
        weight_s = entropy_weight(class_pred_s)
        weight_t = entropy_weight(class_pred_t)

        weights = torch.cat([weight_s, weight_t], dim=0)

        domain_loss_each = criterion_domain(
            domain_pred,
            domain_labels
        )

        loss_domain = (weights * domain_loss_each).mean()

        loss = loss_class + loss_domain

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_domain_loss += loss_domain.item()
        total_loss += loss.item()
        total_batches += 1

        loop.set_postfix({
            "class_loss": loss_class.item(),
            "cdan_loss": loss_domain.item(),
            "lambda": lambda_grl,
            "total_loss": loss.item()
        })

    avg_class_loss = total_class_loss / total_batches
    avg_domain_loss = total_domain_loss / total_batches
    avg_total_loss = total_loss / total_batches

    mean_entropy, target_acc = evaluate_target(
        model,
        target_test_loader,
        device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train class loss : {avg_class_loss:.4f}")
    print(f"Train CDAN loss  : {avg_domain_loss:.4f}")
    print(f"Train total loss : {avg_total_loss:.4f}")
    print(f"GRL lambda       : {lambda_grl:.4f}")
    print(f"Target entropy   : {mean_entropy:.8f}")
    print(f"Target accuracy  : {target_acc:.4f}")
    print("=" * 70)

    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"CDAN_MobileV3Small_best_entropy_"
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
            map_location=device)
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






