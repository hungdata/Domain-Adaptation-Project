


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
# Target label chỉ dùng evaluate, KHÔNG dùng train
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

LR_G = 1e-4
LR_C = 1e-4
WEIGHT_DECAY = 1e-5

NUM_CLASSES = 2
MCD_STEPS = 4

SOURCE_STEP_C_WEIGHT = 0.0

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
# MCD Model: Feature Extractor + 2 Classifiers
# =========================================================
class MCDMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.classifier1 = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        feat = self.feature_extractor(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        out1 = self.classifier1(feat)
        out2 = self.classifier2(feat)
        return out1, out2, feat


# =========================================================
# MCD Discrepancy
# L1 distance giữa softmax outputs của C1 và C2
# =========================================================
def classifier_discrepancy(out1, out2):
    prob1 = F.softmax(out1, dim=1)
    prob2 = F.softmax(out2, dim=1)

    discrepancy = torch.mean(torch.abs(prob1 - prob2))

    discrepancy = torch.nan_to_num(
        discrepancy,
        nan=1e-8,
        posinf=1e-8,
        neginf=1e-8
    )

    return discrepancy


# =========================================================
# Freeze / Unfreeze
# =========================================================
def set_requires_grad(module, flag=True):
    for p in module.parameters():
        p.requires_grad = flag


# =========================================================
# Evaluate Target
# Không update gradient
# Dùng trung bình output của 2 classifiers
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

            out1, out2, _ = model(x_t)

            prob1 = F.softmax(out1, dim=1)
            prob2 = F.softmax(out2, dim=1)

            probs = (prob1 + prob2) / 2.0
            probs = torch.clamp(probs, min=1e-8, max=1.0)

            entropy = -(probs * torch.log(probs)).sum(dim=1)

            entropy = torch.nan_to_num(
                entropy,
                nan=1e-8,
                posinf=1e-8,
                neginf=1e-8
            )

            preds = probs.argmax(dim=1)

            correct += (preds == y_t).sum().item()
            total += y_t.size(0)

            total_entropy += entropy.sum().item()
            total_samples += x_t.size(0)

    mean_entropy = total_entropy / total_samples if total_samples > 0 else 1e-8
    target_acc = correct / total if total > 0 else 0.0

    return mean_entropy, target_acc


# =========================================================
# Device + Model + Optimizers
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = MCDMobileV3Small(
    num_classes=NUM_CLASSES
).to(device)

# Nếu có model train sẵn thì mở 2 dòng này
# pretrained_path = "your_model.pth"
# model.load_state_dict(torch.load(pretrained_path, map_location=device))

optimizer_g = torch.optim.Adam(
    model.feature_extractor.parameters(),
    lr=LR_G,
    weight_decay=WEIGHT_DECAY
)

optimizer_c = torch.optim.Adam(
    list(model.classifier1.parameters()) +
    list(model.classifier2.parameters()),
    lr=LR_C,
    weight_decay=WEIGHT_DECAY
)

criterion_class = nn.CrossEntropyLoss()


# =========================================================
# Training Loop MCD
# Step A: train G + C1 + C2 on source
# Step B: freeze G, train C1/C2:
#         minimize source CE, maximize target discrepancy
# Step C: freeze C1/C2, train G:
#         minimize target discrepancy
# =========================================================
best_entropy = float("inf")
best_model_path = None
wait = 0

for epoch in range(EPOCHS):
    model.train()

    total_step_a_loss = 0.0
    total_step_b_loss = 0.0
    total_step_c_loss = 0.0
    total_discrepancy_b = 0.0
    total_discrepancy_c = 0.0
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

        # =================================================
        # Step A: Train G + C1 + C2 on source
        # =================================================
        set_requires_grad(model.feature_extractor, True)
        set_requires_grad(model.classifier1, True)
        set_requires_grad(model.classifier2, True)

        optimizer_g.zero_grad()
        optimizer_c.zero_grad()

        out1_s, out2_s, _ = model(x_s)

        loss_s1 = criterion_class(out1_s, y_s)
        loss_s2 = criterion_class(out2_s, y_s)

        loss_step_a = loss_s1 + loss_s2

        loss_step_a.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer_g.step()
        optimizer_c.step()

        # =================================================
        # Step B: Freeze G, train C1/C2
        # maximize discrepancy target
        # minimize CE source - discrepancy target
        # =================================================
        set_requires_grad(model.feature_extractor, False)
        set_requires_grad(model.classifier1, True)
        set_requires_grad(model.classifier2, True)

        optimizer_c.zero_grad()

        with torch.no_grad():
            feat_s = model.extract_features(x_s)
            feat_t = model.extract_features(x_t)

        out1_s = model.classifier1(feat_s)
        out2_s = model.classifier2(feat_s)

        out1_t = model.classifier1(feat_t)
        out2_t = model.classifier2(feat_t)

        loss_s1 = criterion_class(out1_s, y_s)
        loss_s2 = criterion_class(out2_s, y_s)

        discrepancy_b = classifier_discrepancy(out1_t, out2_t)

        loss_step_b = loss_s1 + loss_s2 - discrepancy_b

        loss_step_b.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.classifier1.parameters()) +
            list(model.classifier2.parameters()),
            max_norm=5.0
        )

        optimizer_c.step()

        # =================================================
        # Step C: Freeze C1/C2, train G
        # minimize target discrepancy
        # =================================================
        set_requires_grad(model.feature_extractor, True)
        set_requires_grad(model.classifier1, False)
        set_requires_grad(model.classifier2, False)

        loss_step_c_value = 0.0
        discrepancy_c_value = 0.0

        for _ in range(MCD_STEPS):
            optimizer_g.zero_grad()

            out1_t, out2_t, _ = model(x_t)

            discrepancy_c = classifier_discrepancy(out1_t, out2_t)

            if SOURCE_STEP_C_WEIGHT > 0:
                out1_s, out2_s, _ = model(x_s)

                source_loss_c = (
                    criterion_class(out1_s, y_s) +
                    criterion_class(out2_s, y_s)
                )

                loss_step_c = discrepancy_c + SOURCE_STEP_C_WEIGHT * source_loss_c
            else:
                loss_step_c = discrepancy_c

            loss_step_c.backward()

            torch.nn.utils.clip_grad_norm_(
                model.feature_extractor.parameters(),
                max_norm=5.0
            )

            optimizer_g.step()

            loss_step_c_value = loss_step_c.item()
            discrepancy_c_value = discrepancy_c.item()

        # mở lại grad cho vòng sau
        set_requires_grad(model.feature_extractor, True)
        set_requires_grad(model.classifier1, True)
        set_requires_grad(model.classifier2, True)

        total_step_a_loss += loss_step_a.item()
        total_step_b_loss += loss_step_b.item()
        total_step_c_loss += loss_step_c_value
        total_discrepancy_b += discrepancy_b.item()
        total_discrepancy_c += discrepancy_c_value
        total_batches += 1

        loop.set_postfix({
            "stepA_src": loss_step_a.item(),
            "stepB": loss_step_b.item(),
            "disc_B": discrepancy_b.item(),
            "stepC": loss_step_c_value,
            "disc_C": discrepancy_c_value
        })

    avg_step_a_loss = total_step_a_loss / total_batches
    avg_step_b_loss = total_step_b_loss / total_batches
    avg_step_c_loss = total_step_c_loss / total_batches
    avg_discrepancy_b = total_discrepancy_b / total_batches
    avg_discrepancy_c = total_discrepancy_c / total_batches

    # =====================================================
    # Evaluate target sau mỗi epoch
    # Target label chỉ dùng evaluate
    # =====================================================
    mean_entropy, target_acc = evaluate_target(
        model=model,
        target_loader=target_test_loader,
        device=device
    )

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Step A source loss       : {avg_step_a_loss:.4f}")
    print(f"Step B classifier loss   : {avg_step_b_loss:.4f}")
    print(f"Step C generator loss    : {avg_step_c_loss:.8f}")
    print(f"Target discrepancy StepB : {avg_discrepancy_b:.8f}")
    print(f"Target discrepancy StepC : {avg_discrepancy_c:.8f}")
    print(f"Target entropy           : {mean_entropy:.8f}")
    print(f"Target accuracy          : {target_acc:.4f}")
    print("=" * 70)

    # =====================================================
    # Save best model theo entropy
    # =====================================================
    if mean_entropy < best_entropy - min_delta:
        best_entropy = mean_entropy
        wait = 0

        best_model_path = (
            f"MCD_MobileV3Small_best_entropy_"
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
        model=model,
        target_loader=target_test_loader,
        device=device
    )

    print("\nFinal Best Model Result")
    print(f"Best model file       : {best_model_path}")
    print(f"Best entropy          : {best_entropy:.8f}")
    print(f"Final target entropy  : {final_entropy:.8f}")
    print(f"Final target accuracy : {final_acc:.4f}")

else:
    print("No best model was saved.")






