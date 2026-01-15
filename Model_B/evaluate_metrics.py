import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("temp") 
FUSION_CLASS_ID = 3

def read_yolo_labels(path, target_class):
    labels = []
    if not path.exists():
        return labels
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) == target_class:
                labels.append(list(map(float, parts[1:5])))
    return labels

def evaluate_fold(fold_idx):
    # Standard counters (Spatial/IoU based)
    tp, fp, fn = 0, 0, 0
    
    # Modified counters (Count-based)
    mod_tp, mod_fp, mod_fn = 0, 0, 0
    
    total_cells = 0
    total_instances_gt = 0

    case_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])

    for case_dir in case_dirs:
        gt_dir = case_dir / "yolo_ground_truth"
        pred_dir = case_dir / f"predictions-{fold_idx}" / "labels"
        
        if not gt_dir.exists() or not pred_dir.exists():
            continue

        sky_gt_files = list(gt_dir.glob("*_SKY_cell*.txt"))
        
        for gt_file in sky_gt_files:
            total_cells += 1
            pred_file = pred_dir / gt_file.name
            
            gt_boxes = read_yolo_labels(gt_file, FUSION_CLASS_ID)
            pred_boxes = read_yolo_labels(pred_file, FUSION_CLASS_ID)
            
            num_gt = len(gt_boxes)
            num_pred = len(pred_boxes)
            total_instances_gt += num_gt

            # --- UPDATED: Multi-Instance Modified Logic ---
            # If Pred Count == GT Count, all are TPs
            if num_pred == num_gt:
                mod_tp += num_gt
            # If model over-predicts (e.g., Pred=3, GT=2)
            elif num_pred > num_gt:
                mod_tp += num_gt           # The ones that exist in GT
                mod_fp += (num_pred - num_gt) # The extra ones
            # If model under-predicts (e.g., Pred=1, GT=3)
            elif num_pred < num_gt:
                mod_tp += num_pred         # The ones found
                mod_fn += (num_gt - num_pred) # The ones missed

    # Modified Metrics Calculation
    mod_p = mod_tp / (mod_tp + mod_fp) if (mod_tp + mod_fp) > 0 else 0
    mod_r = mod_tp / (mod_tp + mod_fn) if (mod_tp + mod_fn) > 0 else 0

    return {
        "Fold": fold_idx,
        "Images": total_cells,
        "Instances": total_instances_gt,
        "Mod_P": round(mod_p, 3),
        "Mod_R": round(mod_r, 3)
    }

if __name__ == "__main__":
    final_stats = []
    for i in range(1, 6):
        final_stats.append(evaluate_fold(i))
    
    df = pd.DataFrame(final_stats)
    df.to_csv("fusion_count_metrics.csv", index=False)
    print(df.to_string(index=False))