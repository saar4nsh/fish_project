import cv2
import numpy as np
import math
import re
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = "FISH-Sample-Standardized" 
CHANNELS = ['FITC', 'ORANGE', 'AQUA']
COLORS = {
    0: (0, 255, 0),   # Green (FITC)
    1: (0, 0, 255),   # Red (ORANGE)
    2: (255, 255, 0), # Aqua (AQUA)
    3: (255, 0, 255)  # Fusion (Magenta)
}

# --- HELPER FUNCTIONS ---
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def get_distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def yolo_to_cv2(yolo_box, img_shape):
    h, w = img_shape
    cx_n, cy_n, w_n, h_n = yolo_box
    x1 = int((cx_n - w_n/2) * w)
    y1 = int((cy_n - h_n/2) * h)
    x2 = int((cx_n + w_n/2) * w)
    y2 = int((cy_n + h_n/2) * h)
    return (x1, y1, x2, y2)

def cv2_to_yolo(cv2_box, img_shape, cid):
    h, w = img_shape
    x1, y1, x2, y2 = cv2_box
    bw, bh = x2 - x1, y2 - y1
    # Ensure values stay within 0-1 range
    return (cid, max(0, min(1, (x1 + bw/2)/w)), max(0, min(1, (y1 + bh/2)/h)), max(0, min(1, bw/w)), max(0, min(1, bh/h)))

def process_fusion_logic(cell_data, sky_path, output_label_path, output_vis_path):
    sky_img = cv2.imread(str(sky_path))
    if sky_img is None: 
        return
    img_h, img_w, _ = sky_img.shape
    
    all_dets = []
    for ch_idx, txt_path in cell_data.items():
        if not txt_path.exists(): continue
        with open(txt_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    # parts[0] is class, but we use ch_idx for consistency
                    cv2_box = yolo_to_cv2(parts[1:5], (img_h, img_w))
                    all_dets.append({'cid': ch_idx, 'box': cv2_box, 'yolo': parts[1:5]})

    greens = [d for d in all_dets if d['cid'] == 0]
    reds = [d for d in all_dets if d['cid'] == 1]
    aquas = [d for d in all_dets if d['cid'] == 2]

    used_g, used_r = set(), set()
    final_labels = []
    
    # 1. Matching Logic (Sorted by distance to prioritize closest pairs)
    pairs = []
    for i, g in enumerate(greens):
        for j, r in enumerate(reds):
            dist = get_distance(get_center(g['box']), get_center(r['box']))
            thresh = (get_diag(g['box']) + get_diag(r['box'])) / 2.0
            if dist <= thresh:
                pairs.append((dist, i, j))
    
    pairs.sort()
    for dist, gi, rj in pairs:
        if gi in used_g or rj in used_r: continue
        used_g.add(gi); used_r.add(rj)
        gc, rc = get_center(greens[gi]['box']), get_center(reds[rj]['box'])
        fc = ((gc[0]+rc[0])/2, (gc[1]+rc[1])/2)
        side = (get_diag(greens[gi]['box']) + get_diag(reds[rj]['box'])) / 2.828
        f_box = (int(fc[0]-side), int(fc[1]-side), int(fc[0]+side), int(fc[1]+side))
        
        final_labels.append(cv2_to_yolo(f_box, (img_h, img_w), 3))
        cv2.rectangle(sky_img, (f_box[0], f_box[1]), (f_box[2], f_box[3]), COLORS[3], 1)

    # 2. Add residuals (Unfused signals)
    for i, g in enumerate(greens):
        if i not in used_g:
            final_labels.append((0, *g['yolo']))
            cv2.rectangle(sky_img, (g['box'][0], g['box'][1]), (g['box'][2], g['box'][3]), COLORS[0], 1)
    for i, r in enumerate(reds):
        if i not in used_r:
            final_labels.append((1, *r['yolo']))
            cv2.rectangle(sky_img, (r['box'][0], r['box'][1]), (r['box'][2], r['box'][3]), COLORS[1], 1)
    for a in aquas:
        final_labels.append((2, *a['yolo']))
        cv2.rectangle(sky_img, (a['box'][0], a['box'][1]), (a['box'][2], a['box'][3]), COLORS[2], 1)

    # 3. Robust Saving (Forcing file write)
    try:
        with open(output_label_path, 'w', encoding='utf-8') as f:
            for lab in final_labels:
                f.write(f"{int(lab[0])} {' '.join(f'{x:.6f}' for x in lab[1:])}\n")
            f.flush()
            os.fsync(f.fileno()) # Forces OS to write to disk immediately
        
        cv2.imwrite(str(output_vis_path), sky_img)
    except Exception as e:
        print(f"Error saving {output_label_path.name}: {e}")

def run_multi_model_fusion():
    base_path = Path(BASE_DIR).resolve()
    # Sort case directories to ensure consistent processing order
    case_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for case_dir in tqdm(case_dirs, desc="Processing Cases"):
        patch_dir = case_dir / "cell_patches"
        if not patch_dir.exists(): continue

        # Refresh sky files dictionary for every case
        sky_files = {f.name: f for f in patch_dir.glob("*_SKY_cell*.png")}

        for m_idx in range(1, 6):
            pred_dir = case_dir / f"predictions-{m_idx}"
            label_dir = pred_dir / "labels"
            vis_dir = pred_dir / "visualisations"
            
            if not label_dir.exists(): continue
            vis_dir.mkdir(exist_ok=True, parents=True)

            # Group prediction TXT files
            cell_groups = defaultdict(dict)
            for txt_file in label_dir.glob("*.txt"):
                # Avoid processing a file we just created if the script is re-run
                if "_SKY_cell" in txt_file.name: continue 
                
                # Regex matches prefix, channel, and cell index
                match = re.search(r'^(.*)_(FITC|ORANGE|AQUA)_(cell\d+)', txt_file.stem)
                if match:
                    prefix, channel, cell_id = match.groups()
                    unique_key = f"{prefix}_{cell_id}"
                    ch_idx = CHANNELS.index(channel)
                    cell_groups[unique_key][ch_idx] = txt_file

            for unique_key, data in cell_groups.items():
                # Correctly map back to the SKY image name found in cell_patches
                expected_sky_name = unique_key.replace("_cell", "_SKY_cell") + ".png"
                
                if expected_sky_name in sky_files:
                    sky_path = sky_files[expected_sky_name]
                    
                    # Target filenames
                    out_label_name = sky_path.with_suffix('.txt').name
                    out_vis_name = sky_path.name
                    
                    # Full paths
                    target_label_path = label_dir / out_label_name
                    target_vis_path = vis_dir / out_vis_name
                    
                    process_fusion_logic(
                        data, 
                        sky_path, 
                        target_label_path, 
                        target_vis_path
                    )

if __name__ == "__main__":
    run_multi_model_fusion()
    print("\n✅ Multi-model fusion finalized.")