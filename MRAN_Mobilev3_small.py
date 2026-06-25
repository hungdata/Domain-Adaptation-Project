


print("begin")
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from itertools import cycle

from torch.utils.data import DataLoader
from torchvision import transforms, models


# =========================================================
# Dataset
# =========================================================
class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    path = os.path.join(subdir, file)
                    subdir_lower = subdir.lower()

                    if "fresh" in subdir_lower:
                        label = 0
                    elif "rotten" in subdir_lower:
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
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
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

# MRAN paper logic: multi-representation conditional distribution alignment
CLASS_WEIGHT = 1.0
CONDITIONAL_WEIGHT = 1.0

# để 0.0 nếu muốn sát MRAN paper hơn
MARGINAL_WEIGHT = 0.0

PSEUDO_CONF_THRESHOLD = 0.70
WARMUP_EPOCHS = 5

patience = 20
min_delta = 1e-8


# =========================================================
# Transform + Loader
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
    drop_last=True,
    num_workers=0
)

target_loader = DataLoader(
    target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

target_test_loader = DataLoader(
    target_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0
)


# =========================================================
# Multi-kernel Gaussian MMD
# =========================================================
def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([source, target], dim=0)
    total = F.normalize(total, p=2, dim=1)

    n_samples = total.size(0)

    total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / max(n_samples ** 2 - n_samples, 1)

    bandwidth = torch.clamp(bandwidth, min=1e-8)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = 0.0
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        kernels += torch.exp(-l2_distance / bw)

    return kernels


def mmd_loss(source, target):
    if source.size(0) < 1 or target.size(0) < 1:
        return torch.tensor(0.0, device=source.device)

    kernels = gaussian_kernel(source, target)

    ns = source.size(0)
    nt = target.size(0)

    K_ss = kernels[:ns, :ns]
    K_tt = kernels[ns:, ns:]
    K_st = kernels[:ns, ns:]

    loss = K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()

    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def conditional_mmd_loss(feat_s, feat_t, y_s, pseudo_y_t, pseudo_conf_t, num_classes, conf_threshold):
    device = feat_s.device

    total_loss = torch.tensor(0.0, device=device)
    valid_classes = 0

    for c in range(num_classes):
        source_mask = y_s == c
        target_mask = (pseudo_y_t == c) & (pseudo_conf_t >= conf_threshold)

        if source_mask.sum() > 0 and target_mask.sum() > 0:
            total_loss += mmd_loss(feat_s[source_mask], feat_t[target_mask])
            valid_classes += 1

    if valid_classes > 0:
        total_loss = total_loss / valid_classes

    return torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0), valid_classes


# =========================================================
# IAM - Inception Adaptation Module
# =========================================================
class InceptionAdaptationModule(nn.Module):
    def __init__(self, in_channels=576, branch_channels=128):
        super().__init__()

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        r1 = torch.flatten(self.avgpool(self.branch_1x1(x)), 1)
        r2 = torch.flatten(self.avgpool(self.branch_3x3(x)), 1)
        r3 = torch.flatten(self.avgpool(self.branch_5x5(x)), 1)
        r4 = torch.flatten(self.avgpool(self.branch_pool(x)), 1)

        reps = [r1, r2, r3, r4]
        concat = torch.cat(reps, dim=1)

        return reps, concat


# =========================================================
# MRAN MobileNetV3 Small
# =========================================================
class MRANMobileV3Small(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = mobilenet.features

        self.iam = InceptionAdaptationModule(
            in_channels=576,
            branch_channels=128
        )

        self.branch_classifiers = nn.ModuleList([
            nn.Linear(128, num_classes),
            nn.Linear(128, num_classes),
            nn.Linear(128, num_classes),
            nn.Linear(128, num_classes)
        ])

        self.final_classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat_map = self.feature_extractor(x)
        reps, concat_feat = self.iam(feat_map)

        branch_logits = []
        for rep, clf in zip(reps, self.branch_classifiers):
            branch_logits.append(clf(rep))

        final_logits = self.final_classifier(concat_feat)

        return final_logits, branch_logits, reps, concat_feat


# =========================================================
# MRAN Loss
# =========================================================
def mran_loss(
    reps_s,
    reps_t,
    concat_s,
    concat_t,
    logits_t,
    y_s,
    num_classes=2,
    conf_threshold=0.70,
    use_marginal=False
):
    probs_t = F.softmax(logits_t.detach(), dim=1)
    pseudo_conf_t, pseudo_y_t = probs_t.max(dim=1)

    all_s = reps_s + [concat_s]
    all_t = reps_t + [concat_t]

    total_cond = torch.tensor(0.0, device=concat_s.device)
    total_marginal = torch.tensor(0.0, device=concat_s.device)
    total_valid = 0

    for fs, ft in zip(all_s, all_t):
        cond_loss, valid_classes = conditional_mmd_loss(
            feat_s=fs,
            feat_t=ft,
            y_s=y_s,
            pseudo_y_t=pseudo_y_t,
            pseudo_conf_t=pseudo_conf_t,
            num_classes=num_classes,
            conf_threshold=conf_threshold
        )

        total_cond += cond_loss
        total_valid += valid_classes

        if use_marginal:
            total_marginal += mmd_loss(fs, ft)

    num_reps = len(all_s)

    total_cond = total_cond / num_reps
    total_marginal = total_marginal / num_reps

    total = CONDITIONAL_WEIGHT * total_cond

    if use_marginal:
        total = total + MARGINAL_WEIGHT * total_marginal

    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

    return total, total_cond, total_marginal, total_valid


def prediction_entropy(logits):
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    entropy = -(probs * torch.log(probs)).sum(dim=1)
    return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_target(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    entropy_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _, _, _ = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            entropy = prediction_entropy(logits)
            entropy_sum += entropy.sum().item()
            sample_count += x.size(0)

    mean_entropy = entropy_sum / max(sample_count, 1)
    acc = correct / max(total, 1)

    return mean_entropy, acc


def evaluate_source(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _, _, _ = model(x)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item()

    return loss_sum / max(len(loader), 1), correct / max(total, 1)


# =========================================================
# Train
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = MRANMobileV3Small(num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss()

best_entropy = float("inf")
best_model_path = None
wait = 0

steps_per_epoch = max(len(source_loader), len(target_loader))

for epoch in range(EPOCHS):
    model.train()

    total_class_loss = 0.0
    total_branch_loss = 0.0
    total_adapt_loss = 0.0
    total_cond_loss = 0.0
    total_marginal_loss = 0.0
    total_valid = 0
    total_loss_meter = 0.0

    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for _ in loop:
        x_s, y_s = next(source_iter)
        x_t, _ = next(target_iter)

        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)

        logits_s, branch_logits_s, reps_s, concat_s = model(x_s)
        logits_t, _, reps_t, concat_t = model(x_t)

        main_class_loss = criterion(logits_s, y_s)

        branch_class_loss = 0.0
        for b_logits in branch_logits_s:
            branch_class_loss = branch_class_loss + criterion(b_logits, y_s)
        branch_class_loss = branch_class_loss / len(branch_logits_s)

        if epoch < WARMUP_EPOCHS:
            adapt_loss = torch.tensor(0.0, device=device)
            cond_loss = torch.tensor(0.0, device=device)
            marginal_loss = torch.tensor(0.0, device=device)
            valid_classes = 0
        else:
            adapt_loss, cond_loss, marginal_loss, valid_classes = mran_loss(
                reps_s=reps_s,
                reps_t=reps_t,
                concat_s=concat_s,
                concat_t=concat_t,
                logits_t=logits_t,
                y_s=y_s,
                num_classes=NUM_CLASSES,
                conf_threshold=PSEUDO_CONF_THRESHOLD,
                use_marginal=(MARGINAL_WEIGHT > 0)
            )

        loss = (
            CLASS_WEIGHT * main_class_loss
            + 0.5 * branch_class_loss
            + adapt_loss
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_class_loss += main_class_loss.item()
        total_branch_loss += float(branch_class_loss.item())
        total_adapt_loss += adapt_loss.item()
        total_cond_loss += cond_loss.item()
        total_marginal_loss += marginal_loss.item()
        total_valid += valid_classes
        total_loss_meter += loss.item()

        loop.set_postfix({
            "class": main_class_loss.item(),
            "branch": float(branch_class_loss.item()),
            "adapt": adapt_loss.item(),
            "cond": cond_loss.item(),
            "total": loss.item()
        })

    avg_class_loss = total_class_loss / steps_per_epoch
    avg_branch_loss = total_branch_loss / steps_per_epoch
    avg_adapt_loss = total_adapt_loss / steps_per_epoch
    avg_cond_loss = total_cond_loss / steps_per_epoch
    avg_marginal_loss = total_marginal_loss / steps_per_epoch
    avg_valid = total_valid / steps_per_epoch
    avg_total_loss = total_loss_meter / steps_per_epoch

    source_eval_loss, source_eval_acc = evaluate_source(model, source_loader, device)
    target_entropy, target_acc = evaluate_target(model, target_test_loader, device)

    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train class loss       : {avg_class_loss:.4f}")
    print(f"Train branch loss      : {avg_branch_loss:.4f}")
    print(f"Train MRAN adapt loss  : {avg_adapt_loss:.8f}")
    print(f"Conditional MMD loss   : {avg_cond_loss:.8f}")
    print(f"Marginal MMD loss      : {avg_marginal_loss:.8f}")
    print(f"Valid pseudo classes   : {avg_valid:.2f}")
    print(f"Train total loss       : {avg_total_loss:.4f}")
    print(f"Source eval loss       : {source_eval_loss:.4f}")
    print(f"Source eval acc        : {source_eval_acc:.4f}")
    print(f"Target entropy         : {target_entropy:.8f}")
    print(f"Target accuracy        : {target_acc:.4f}")
    print("=" * 70)

    # Không dùng target label để save model.
    # Target accuracy chỉ để xem thử, không dùng chọn model.
    if target_entropy < best_entropy - min_delta and source_eval_acc > 0.80:
        best_entropy = target_entropy
        wait = 0

        best_model_path = (
            f"MRAN_MobileV3Small_paper_logic_"
            f"epoch{epoch + 1}_entropy{best_entropy:.8f}.pth"
        )

        torch.save(model.state_dict(), best_model_path)

        print(f"Saved best model: {best_model_path}")
        print(f"Best entropy: {best_entropy:.8f}")

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
# Load best model
# =========================================================
if best_model_path is not None:
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_entropy, final_acc = evaluate_target(model, target_test_loader, device)

    print("\nFinal Best Model Result")
    print(f"Best model file       : {best_model_path}")
    print(f"Best entropy          : {best_entropy:.8f}")
    print(f"Final target entropy  : {final_entropy:.8f}")
    print(f"Final target accuracy : {final_acc:.4f}")
else:
    print("No best model was saved.")






