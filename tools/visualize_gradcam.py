import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.uda_utils import (
    FeatureExtractor,
    Bottleneck,
    ClassifierHead,
    get_device,
    get_eval_transform
)

# Hỗ trợ nhận diện MRM Feature Extractor của MSUN
class MRMFeatureExtractor(nn.Module):
    def __init__(self, base_extractor):
        super().__init__()
        self.features = base_extractor.features
        self.avgpool = base_extractor.avgpool
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.output_dim = base_extractor.output_dim
        
        self.mrm_fc = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        f = self.features(x)
        avg_f = torch.flatten(self.avgpool(f), 1)
        max_f = torch.flatten(self.maxpool(f), 1)
        concat_f = torch.cat([avg_f, max_f], dim=1)
        return self.mrm_fc(concat_f)

class FullModel(nn.Module):
    def __init__(self, F_ext, bottleneck, classifier):
        super().__init__()
        self.F_ext = F_ext
        self.bottleneck = bottleneck
        self.classifier = classifier
        
    def forward(self, x):
        f = self.F_ext(x)
        b = self.bottleneck(f)
        c = self.classifier(b)
        return c

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        score = logits[0, class_idx]
        score.backward()
        
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        activations = self.activations
        
        cam = (gradients * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0, 0].detach().cpu().numpy(), class_idx

def get_cam_overlay(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = 0.5
    overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return overlay

def overlay_cam_on_image(img_path, cam, out_path):
    overlay = get_cam_overlay(img_path, cam)
    # Đọc ảnh gốc
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize cam cho vừa với kích thước ảnh gốc
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Tạo heatmap màu đỏ/vàng
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    alpha = 0.5
    overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    
    # Lưu lại
    plt.figure(figsize=(5,5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image-path", type=str, default=None, help="Path to single input image")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory containing images (will pick 10 random)")
    parser.add_argument("--output", type=str, default="gradcam_output.png")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    method_name = ckpt.get("method_name", "Unknown")
    
    base_ext = FeatureExtractor()
    if method_name.startswith("MSUN"):
        print("Detected MSUN Method. Using MRMFeatureExtractor.")
        F_ext = MRMFeatureExtractor(base_ext).to(device)
    else:
        F_ext = base_ext.to(device)
        
    bottleneck = Bottleneck().to(device)
    
    cls_state = ckpt["model_state_dict"]["classifier"]
    if "fc.bias" in cls_state or "bias" in cls_state:
        classifier = ClassifierHead().to(device)
    else:
        print("Detected ArcMarginProduct Head - Warning: ArcMarginProduct is deprecated, falling back to Linear.")
        classifier = ClassifierHead().to(device)
        
    F_ext.load_state_dict(ckpt["model_state_dict"]["F_ext"])
    bottleneck.load_state_dict(ckpt["model_state_dict"]["bottleneck"])
    classifier.load_state_dict(cls_state)
    
    model = FullModel(F_ext, bottleneck, classifier).to(device)
    
    # Lấy target layer (layer Conv cuối cùng của features)
    if hasattr(F_ext, 'features'):
        target_layer = F_ext.features[-1]
    else:
        target_layer = F_ext[-1] # Fallback
    
    grad_cam = GradCAM(model, target_layer)
    transform = get_eval_transform()
    
    if args.image_dir and os.path.isdir(args.image_dir):
        # Pick 10 random images
        import glob
        import random
        
        # Lấy tất cả ảnh
        valid_exts = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
        all_imgs = []
        for ext in valid_exts:
            all_imgs.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))
            
        fresh_imgs = [p for p in all_imgs if "fresh" in p.lower()]
        rotten_imgs = [p for p in all_imgs if "rotten" in p.lower()]

        # Nếu không phân loại được theo tên thư mục/file, lấy 50-50 dựa trên nửa đầu và nửa cuối (fallback an toàn)
        if len(fresh_imgs) == 0 and len(rotten_imgs) == 0 and len(all_imgs) > 0:
            print("[WARN] Không tìm thấy từ khóa 'fresh' hay 'rotten' trong đường dẫn ảnh. Đang lấy chia đôi danh sách ảnh.")
            half = len(all_imgs) // 2
            fresh_imgs = all_imgs[:half]
            rotten_imgs = all_imgs[half:]

        # Lọc unique
        fresh_imgs = list(set(fresh_imgs))
        rotten_imgs = list(set(rotten_imgs))

        random.shuffle(fresh_imgs)
        random.shuffle(rotten_imgs)
        
        selected_imgs = fresh_imgs[:50] + rotten_imgs[:50]
        
        if len(selected_imgs) == 0:
            print(f"[ERROR] No images found in {args.image_dir}")
            return
            
        fig, axes = plt.subplots(10, 10, figsize=(30, 30))
        axes = axes.flatten()
        class_names = ["Fresh", "Rotten"]
        
        for i, img_p in enumerate(selected_imgs):
            try:
                pil_img = Image.open(img_p).convert('RGB')
            except:
                continue
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            cam, pred_class = grad_cam(input_tensor)
            overlay = get_cam_overlay(img_p, cam)
            
            ax = axes[i]
            ax.imshow(overlay)
            
            # Suy luận label thực tế từ tên đường dẫn (thư mục chứa ảnh)
            true_class = "Rotten" if "rotten" in img_p.lower() else "Fresh"
            
            # Đổi màu text để dễ nhìn: Xanh lá nếu đoán đúng, Đỏ nếu đoán sai
            color = "green" if true_class == class_names[pred_class] else "red"
            
            ax.set_title(f"True: {true_class} | Pred: {class_names[pred_class]}\n{os.path.basename(img_p)}", 
                         fontsize=8, color=color, fontweight='bold')
            ax.axis('off')
            
        # Hide empty subplots if fewer than 100 images
        for j in range(len(selected_imgs), 100):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Đã lưu {len(selected_imgs)} hình ảnh Grad-CAM tại: {args.output}")
        
    elif args.image_path:
        pil_img = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        cam, pred_class = grad_cam(input_tensor)
        class_names = ["Fresh", "Rotten"]
        
        print(f"Predicted Class: {class_names[pred_class]}")
        overlay_cam_on_image(args.image_path, cam, args.output)
        print(f"[SUCCESS] Đã lưu hình ảnh Grad-CAM tại: {args.output}")
    else:
        print("[ERROR] Please provide either --image-path or --image-dir")

if __name__ == "__main__":
    main()
