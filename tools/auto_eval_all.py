import os
import subprocess
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Auto run evaluation and visualizations for all models and seeds")
    parser.add_argument("--working-dir", type=str, default="./working", help="Directory containing seed_xxxx folders")
    parser.add_argument("--output-dir", type=str, default="./results/vis", help="Directory to save visualizations")
    parser.add_argument("--data-root", type=str, default="./uda_fixed_folders")
    args = parser.parse_args()

    seeds = [2026, 2027, 2028]
    methods = [
        "ERM", "DANN", "CDAN", "CDANE", 
        "DSAN", "DSANE", "DSANE_MCC", "CDANE_MCC"
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("🚀 BẮT ĐẦU AUTO EVAL VÀ VISUALIZE")
    print("="*60)

    for seed in seeds:
        print(f"\n{'='*20} SEED {seed} {'='*20}")
        for method in methods:
            model_path = os.path.join(args.working_dir, f"seed_{seed}", f"{method}_model", "last_model.pth")
            
            if not os.path.exists(model_path):
                print(f"[SKIP] Model {method} ở seed {seed} vẫn chưa có (không tìm thấy last_model.pth)")
                continue
                
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                epoch = int(ckpt.get("epoch", -1))
            except Exception as e:
                print(f"[ERROR] Không thể đọc {model_path}: {e}")
                continue
                
            if epoch != 150:
                print(f"[SKIP] Model {method} ở seed {seed} đang ở epoch {epoch} (Yêu cầu: epoch 150). Bỏ qua Visualize!")
                continue
            
            print(f"\n▶ Đang xử lý: {method} - Seed {seed}")
            print(f"  Model: {model_path}")

            # 1. Evaluate Model (Luôn chạy để xem report text)
            print(f"  [1/5] Chạy evaluate_model.py...")
            subprocess.run(["python", "tools/evaluate_model.py", "--model-path", model_path, "--data-root", args.data_root], check=True)

            # Định nghĩa các đường dẫn output
            tsne_out = os.path.join(args.output_dir, f"tsne_{method}_seed{seed}.png")
            pca_out = os.path.join(args.output_dir, f"pca_{method}_seed{seed}.png")
            gradcam_out = os.path.join(args.output_dir, f"gradcam_{method}_seed{seed}.png")
            cm_out = os.path.join(args.output_dir, f"cm_{method}_seed{seed}.png")

            # 2. t-SNE
            if os.path.exists(tsne_out):
                print(f"  [2/5] t-SNE của {method} (Seed {seed} - Epoch 150) đã có sẵn tại: {tsne_out}")
            else:
                print(f"  [2/5] Đang vẽ t-SNE...")
                subprocess.run(["python", "tools/visualize_tsne.py", "--model-path", model_path, "--data-root", args.data_root, "--output", tsne_out], check=True)

            # 3. PCA
            if os.path.exists(pca_out):
                print(f"  [3/5] PCA của {method} (Seed {seed} - Epoch 150) đã có sẵn tại: {pca_out}")
            else:
                print(f"  [3/5] Đang vẽ PCA...")
                subprocess.run(["python", "tools/visualize_pca.py", "--model-path", model_path, "--data-root", args.data_root, "--output", pca_out], check=True)

            # 4. GradCAM
            if os.path.exists(gradcam_out):
                print(f"  [4/5] GradCAM của {method} (Seed {seed} - Epoch 150) đã có sẵn tại: {gradcam_out}")
            else:
                print(f"  [4/5] Đang vẽ GradCAM...")
                subprocess.run(["python", "tools/visualize_gradcam.py", "--model-path", model_path, "--image-dir", os.path.join(args.data_root, "target", "test"), "--output", gradcam_out], check=True)
            
            # 5. Confusion Matrix
            if os.path.exists(cm_out):
                print(f"  [5/5] Confusion Matrix của {method} (Seed {seed} - Epoch 150) đã có sẵn tại: {cm_out}")
            else:
                print(f"  [5/5] Đang vẽ Confusion Matrix...")
                subprocess.run(["python", "tools/confusion_matrix_plot.py", "--model-path", model_path, "--data-root", args.data_root, "--output", cm_out], check=True)
    
    print("\n" + "="*60)
    print("🎉 HOÀN TẤT TẤT CẢ!")
    print(f"Tất cả ảnh trực quan hoá được lưu tại: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
