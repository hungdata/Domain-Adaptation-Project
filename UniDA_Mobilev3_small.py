


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
# Bỏ mixed / unknown class
# Chỉ dùng fresh=0, rotten=1
# Label target chỉ dùng evaluate, KHÔNG dùng train
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

SOURCE_CE_WEIGHT = 1.0
SOURCE_OVA_WEIGHT = 1.0
TARGET_OVA_ENTROPY_WEIGHT = 0.1
TARGET_IM_WEIGHT = 0.05

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
# UniDA MobileNetV3 Small
# Closed classifier: fresh / rotten
# OVA classifier: mỗi class có [unknown, known]
# =========================================================
class UniDAMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.bottleneck = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.label_predictor = nn.Linear(256, num_classes)

        self.ova_predictor = nn.Linear(256, num_classes * 2)

        self.num_classes = num_classes

    def extract_features(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.bottleneck(feat)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)

        class_out = self.label_predictor(feat)

        ova_out = self.ova_predictor(feat)
        ova_out = ova_out.view(-1, self.num_classes, 2)

        return class_out, ova_out, feat


# =========================================================
# Source OVA Loss
# Class đúng -> known
# Class còn lại -> unknown
# =========================================================
def source_ova_loss(ova_out, labels):
    batch_size, num_classes, _ = ova_out.shape

    ova_labels = torch.zeros(
        batch_size,
        num_classes,
        dtype=torch.long,
        device=ova_out.device
    )

    ova_labels[torch.arange(batch_size), labels] = 1

    loss = F.cross_entropy(
        ova_out.view(-1, 2),
        ova_labels.view(-1)
    )

    return loss


# =========================================================
# Target OVA Entropy Loss
# Không dùng target label
# Làm target quyết định rõ known/unknown hơn
# =========================================================
def target_ova_entropy_loss(ova_out):
    probs = F.softmax(ova_out, dim=2)
    probs = torch.clamp(probs, min=1e-8, max=1.0)

    entropy = -(probs * torch.log(probs)).sum(dim=2)

    loss = entropy.mean()

    loss = torch.nan_to_num(
        loss,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return loss


# =========================================================
# Target Information Maximization Loss
# Không dùng target label
# Giúp target không collapse về 1 class
# =========================================================
def target_information_maximization_loss(class_out):
    probs = F.softmax(class_out, dim=1)
    probs = torch.clamp(probs, min=1e-8, max=1.0)

    sample_entropy = -(probs * torch.log(probs)).sum(dim=1).mean()

    mean_probs = probs.mean(dim=0)
    global_entropy = -(mean_probs * torch.log(mean_probs)).sum()

    loss = sample_entropy - global_entropy

    loss = torch.nan_to_num(
        loss,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return loss


# =========================================================
# Entropy
# =========================================================
def prediction_entropy(logits):
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-8, max=1.0)

    entropy = -(probs * torch.log(probs)).sum(dim=1)

    entropy = torch.nan_to_num(
        entropy,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return entropy


# =========================================================
# Evaluate Target
# Vì đã bỏ mixed/unknown nên evaluate binary fresh/rotten
# Không update gradient
# =========================================================
def evaluate_target(model, target_loader, device):
    model.eval()

    correct = 0
    total = 0

    total_entropy = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_t, y_t in target_loader:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

            class_out, ova_out, _ = model(x_t)

            entropy = prediction_entropy(class_out)

            total_entropy += entropy.sum().item()
            total_samples += x_t.size(0)

            preds = class_out.argmax(dim=1)

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

model = UniDAMobileV3Small(num_classes=NUM_CLASSES).to(device)

# Nếu có model train sẵn thì mở 2 dòng này
# pretrained_path = "your_model.pth"
# model.load_state_dict(torch.load(pretrained_path, map_location=device))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()


# =========================================================
# Training Loop + Early Stopping theo lowest entropy
# Giống style code DAAN của bạn
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_source_ova_loss = 0.0
    total_target_ova_entropy_loss = 0.0
    total_target_im_loss = 0.0
    total_loss = 0.0

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

        # =========================
        # Forward source + target
        # =========================
        class_s, ova_s, feat_s = model(x_s)
        class_t, ova_t, feat_t = model(x_t)

        # =========================
        # Source supervised classification
        # =========================
        loss_class = criterion_class(
            class_s,
            y_s
        )

        # =========================
        # Source OVA loss
        # UniDA-style open-set boundary learning
        # =========================
        loss_source_ova = source_ova_loss(
            ova_s,
            y_s
        )

        # =========================
        # Target unsupervised loss
        # Không dùng target label
        # =========================
        loss_target_ova_entropy = target_ova_entropy_loss(
            ova_t
        )

        loss_target_im = target_information_maximization_loss(
            class_t
        )

        # =========================
        # UniDA total loss
        # =========================
        loss = (
            SOURCE_CE_WEIGHT * loss_class
            + SOURCE_OVA_WEIGHT * loss_source_ova
            + TARGET_OVA_ENTROPY_WEIGHT * loss_target_ova_entropy
            + TARGET_IM_WEIGHT * loss_target_im
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_class_loss += loss_class.item()
        total_source_ova_loss += loss_source_ova.item()
        total_target_ova_entropy_loss += loss_target_ova_entropy.item()
        total_target_im_loss += loss_target_im.item()
        total_loss += loss.item()
        total_batches += 1

        loop.set_postfix({
            "class_loss": loss_class.item(),
            "src_ova": loss_source_ova.item(),
            "t_ova_ent": loss_target_ova_entropy.item(),
            "t_im": loss_target_im.item(),
            "total": loss.item()
        })

    avg_class_loss = total_class_loss / total_batches
    avg_source_ova_loss = total_source_ova_loss / total_batches
    avg_target_ova_entropy_loss = total_target_ova_entropy_loss / total_batches
    avg_target_im_loss = total_target_im_loss / total_batches
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
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss        : {avg_class_loss:.4f}")
    print(f"Source OVA loss         : {avg_source_ova_loss:.4f}")
    print(f"Target OVA entropy loss : {avg_target_ova_entropy_loss:.4f}")
    print(f"Target IM loss          : {avg_target_im_loss:.4f}")
    print(f"Train total loss        : {avg_total_loss:.4f}")
    print(f"Target entropy          : {mean_entropy:.8f}")
    print(f"Target accuracy         : {target_acc:.4f}")
    print("=" * 70)

    # =====================================================
    # Save best model theo entropy
    # Giữ giống logic code cũ của bạn
    # =====================================================
    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"UniDA_MobileV3Small_best_entropy_"
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






