import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

def run_multi_class_predictions(base_dir, yolo_repo_path, model_weights):
    """
    Robustly runs multi-object/multi-class inference across 5 models.
    Filters for FITC, ORANGE, and AQUA channels only.
    """
    base_path = Path(base_dir)
    yolo_repo = Path(yolo_repo_path)
    detect_script = yolo_repo / "detect.py"
    
    if not detect_script.exists():
        print(f"❌ Error: detect.py not found at {detect_script}")
        return

    # 1. Identify all target cases
    case_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    target_channels = ['FITC', 'ORANGE', 'AQUA']

    # 2. Iterate through each of the 5 models
    for model_idx, weights_path in enumerate(model_weights, 1):
        print(f"\n--- Model {model_idx} | Multi-Class Inference Start ---")
        
        for case_dir in tqdm(case_dirs, desc=f"Model {model_idx}"):
            patch_dir = case_dir / "cell_patches"
            if not patch_dir.exists():
                continue

            # Output folder for this specific model run
            pred_dir = case_dir / f"predictions-{model_idx}"
            pred_dir.mkdir(exist_ok=True)

            # Filter patches to process only FITC, ORANGE, and AQUA
            valid_patches = [
                p for p in patch_dir.glob("*.png") 
                if any(ch in p.name for ch in target_channels)
            ]

            if not valid_patches:
                continue

            # 3. Call YOLOv5 detect.py
            # Using --save-txt and --save-conf for robust multi-object data
            # We pass the list of valid patches to the --source argument
            for patch in valid_patches:
                cmd = [
                    "python", str(detect_script),
                    "--weights", str(weights_path),
                    "--source", str(patch),
                    "--project", str(pred_dir),
                    "--name", "temp",
                    "--save-txt",        # Extract all objects to .txt
                    "--save-conf",       # Include confidence scores
                    "--exist-ok",        # Overwrite temp if needed
                    "--classes", "0", "1", "2" # Explicitly target your FISH classes
                ]
                
                # Run the command silently
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

            # 4. Robust Data Organization (Flattening 'temp' folder)
            temp_path = pred_dir / "temp"
            if temp_path.exists():
                # Move label files (.txt)
                label_folder = temp_path / "labels"
                if label_folder.exists():
                    for txt_file in label_folder.glob("*.txt"):
                        shutil.move(str(txt_file), str(pred_dir / txt_file.name))
                
                # Move visualization files (.png)
                for img_file in temp_path.glob("*.png"):
                    shutil.move(str(img_file), str(pred_dir / img_file.name))
                
                # Remove the empty temp structure
                shutil.rmtree(temp_path)

    print("\n✅ Inference complete. Each case folder now contains predictions-1 through predictions-5.")

if __name__ == "__main__":
    # --- UPDATE PATHS ---
    BASE_DIR = "temp"
    YOLO_PATH = "yolov5"
    WEIGHTS_LIST = [
        'Yolov5-5Fold/fold1/weights/best.pt'
    ]
    
    run_multi_class_predictions(BASE_DIR, YOLO_PATH, WEIGHTS_LIST)