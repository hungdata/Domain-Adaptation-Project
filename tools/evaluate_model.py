import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.uda_utils import (
    FeatureExtractor,
    Bottleneck,
    ClassifierHead,
    get_device,
    get_target_test_loader,
    evaluate
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on Target dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-root", type=str, default="./uda_fixed_folders")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    print(f"Loading checkpoint from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    method_name = ckpt.get("method_name", "Unknown")
    epoch = ckpt.get("epoch", "?")
    
    print(f"Model Method: {method_name} | Saved at Epoch: {epoch}")
    
    base_ext = FeatureExtractor()
    if method_name.startswith("MSUN"):
        print("Detected MSUN Method. Using MRMFeatureExtractor.")
        # Local definition to avoid circular imports
        import torch.nn as nn
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
                return self.mrm_fc(torch.cat([avg_f, max_f], dim=1))
                
        F_ext = MRMFeatureExtractor(base_ext).to(device)
    else:
        F_ext = base_ext.to(device)
        
    bottleneck = Bottleneck().to(device)
    
    # Tự động nhận diện loại Classifier (Linear hay ArcFace)
    cls_state = ckpt["model_state_dict"]["classifier"]
    if "fc.bias" in cls_state or "bias" in cls_state:
        print("Detected Standard Linear ClassifierHead.")
        classifier = ClassifierHead().to(device)
    else:
        print("Detected ArcMarginProduct Head.")
        classifier = ArcMarginProduct().to(device)
        
    F_ext.load_state_dict(ckpt["model_state_dict"]["F_ext"])
    bottleneck.load_state_dict(ckpt["model_state_dict"]["bottleneck"])
    classifier.load_state_dict(cls_state)
    
    print("\nLoading Target Test Data...")
    tgt_test_loader = get_target_test_loader(args.data_root, batch_size=args.batch_size)
    
    print("Evaluating on Target Test...")
    tgt_res = evaluate(F_ext, bottleneck, classifier, tgt_test_loader, device)
    
    tn, fp = tgt_res.get('tn', 0), tgt_res.get('fp', 0)
    fn, tp = tgt_res.get('fn', 0), tgt_res.get('tp', 0)
    
    total = tn + fp + fn + tp
    errors = fp + fn
    accuracy = (tn + tp) / total * 100 if total > 0 else 0
    
    import numpy as np
    from sklearn.metrics import classification_report
    
    # Tái tạo mảng y_true và y_pred từ confusion matrix
    y_true = np.array([0]*(tn + fp) + [1]*(fn + tp))
    y_pred = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
    
    report = classification_report(y_true, y_pred, target_names=["Fresh", "Rotten"], digits=4)
    
    print("\n" + "=" * 50)
    print(f"TARGET REPORT - {method_name.upper()}")
    print(f"Test set: {total} images | Errors: {errors} | Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(f"[[{tn:>3}  {fp:>3}]")
    print(f" [{fn:>3}  {tp:>3}]]")
    print("=" * 50)
    
    # ── Plot Confusion Matrix ──
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = [[tgt_res.get('tn', 0), tgt_res.get('fp', 0)],
              [tgt_res.get('fn', 0), tgt_res.get('tp', 0)]]
              
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fresh (0)", "Rotten (1)"],
                    yticklabels=["Fresh (0)", "Rotten (1)"])
        plt.title(f"Target Test Confusion Matrix\nAccuracy: {tgt_res['acc']*100:.2f}%")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        out_name = f"confusion_matrix_{method_name}.png"
        plt.savefig(out_name, dpi=300)
        print(f"\n[SUCCESS] Đã lưu hình ảnh Confusion Matrix Heatmap tại: {out_name}")
        
        # ── Save Text Report as Image ──
        text = f"""TARGET REPORT - {method_name.upper()}
Test set: {total} images | Errors: {errors} | Accuracy: {accuracy:.2f}%

Classification Report:
{report}
Confusion Matrix:
[[{tn:>3}  {fp:>3}]
 [{fn:>3}  {tp:>3}]]"""
 
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.axis('off')
        ax2.text(0.05, 0.95, text, 
                family='monospace', 
                fontsize=12, 
                va='top', 
                ha='left',
                color='black')
        
        text_out_name = f"report_text_{method_name}.png"
        plt.savefig(text_out_name, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
        print(f"[SUCCESS] Đã lưu hình ảnh Text Report tại: {text_out_name}")
        
    except ImportError:
        print("\n[INFO] Không tìm thấy thư viện matplotlib/seaborn để vẽ biểu đồ Confusion Matrix.")
        print("Hãy cài đặt bằng: pip install matplotlib seaborn")

if __name__ == '__main__':
    main()
