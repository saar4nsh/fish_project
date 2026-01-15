import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

def run_multi_class_predictions(base_dir, yolo_repo_path, model_weights):
    base_path = Path(base_dir).resolve()
    yolo_repo = Path(yolo_repo_path).resolve()
    detect_script = yolo_repo / "detect.py"
    
    if not detect_script.exists():
        print(f"❌ Error: detect.py not found at {detect_script}")
        return

    case_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    target_channels = ['FITC', 'ORANGE', 'AQUA']

    for model_idx, weights_path in enumerate(model_weights, 1):
        print(f"\n--- Model {model_idx} | Starting Batch Inference ---")
        weights_path = Path(weights_path).resolve()
        
        for case_dir in tqdm(case_dirs, desc=f"Model {model_idx}"):
            pred_dir = case_dir / f"predictions-{model_idx}"
            
            # Skip if already processed
            if pred_dir.exists() and (pred_dir / "labels").exists():
                continue

            patch_dir = case_dir / "cell_patches"
            if not patch_dir.exists():
                continue

            # 1. Create temporary symlink folder
            filter_tmp = case_dir / "inference_tmp"
            filter_tmp.mkdir(exist_ok=True)
            
            valid_patches = [
                p for p in patch_dir.glob("*.png") 
                if any(ch in p.name for ch in target_channels)
            ]

            if not valid_patches:
                shutil.rmtree(filter_tmp)
                continue

            for patch in valid_patches:
                link_path = filter_tmp / patch.name
                if not link_path.exists():
                    os.symlink(patch, link_path)

            # 2. Setup Prediction Directory
            pred_dir.mkdir(exist_ok=True)

            # 3. Call YOLOv5 with specific visual parameters
            cmd = [
                "python", str(detect_script),
                "--weights", str(weights_path),
                "--source", str(filter_tmp),
                "--device", "1",            # Still using GPU 1 for speed
                "--project", str(pred_dir),
                "--name", "temp",
                "--save-txt",
                "--save-conf",
                "--exist-ok",
                "--classes", "0", "1", "2",
                "--line-thickness", "1",    # Thinner edges & smaller text
                "--hide-conf"               # Optional: hides confidence % for a cleaner look
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # 4. Organize into 'labels' and 'visualisations'
            temp_path = pred_dir / "temp"
            if temp_path.exists():
                # Create the two sub-folders
                final_labels_dir = pred_dir / "labels"
                final_vis_dir = pred_dir / "visualisations"
                final_labels_dir.mkdir(exist_ok=True)
                final_vis_dir.mkdir(exist_ok=True)

                # Move label files (.txt)
                label_source = temp_path / "labels"
                if label_source.exists():
                    for txt_file in label_source.glob("*.txt"):
                        shutil.move(str(txt_file), str(final_labels_dir / txt_file.name))
                
                # Move visualization images (.png)
                for img_file in temp_path.glob("*.png"):
                    shutil.move(str(img_file), str(final_vis_dir / img_file.name))
                
                shutil.rmtree(temp_path)

            shutil.rmtree(filter_tmp)

    print("\n✅ Inference complete and organized.")

if __name__ == "__main__":
    BASE_DIR = "FISH-Sample-Standardized"
    YOLO_PATH = "yolov5"
    WEIGHTS_LIST = ['Yolov5-5Fold/fold1/weights/best.pt',
                    'Yolov5-5Fold/fold2/weights/best.pt',
                    'Yolov5-5Fold/fold3/weights/best.pt',
                    'Yolov5-5Fold/fold4/weights/best.pt',
                    'Yolov5-5Fold/fold5/weights/best.pt']
    
    run_multi_class_predictions(BASE_DIR, YOLO_PATH, WEIGHTS_LIST)