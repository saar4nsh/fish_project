import cv2
import numpy as np
import math
from pathlib import Path
from collections import defaultdict
import shutil
import os
import re  # <-- Import regular expressions

# --- Configuration ---

# **MODIFIED**: Path to the ROOT directory containing the .txt labels
LABEL_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/temp/labels"

# **MODIFIED**: Path to the ROOT directory containing the images
IMAGE_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/13.09.2025/cell_patches"

# **MODIFIED**: Path to the ROOT directory for the final visualizations
VISUALIZATION_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/temp/visualizations"


# Visualization colors
# 0: Green, 1: Red, 2: Aqua, 3: Fusion
COLORS = {
    0: (0, 255, 0),   # Green
    1: (0, 0, 255),   # Red
    2: (255, 255, 0), # Aqua (Cyan/Yellow)
    3: (255, 0, 255)  # **NEW** Fusion (Magenta)
}

# --- Helper Functions ---
# (Helper functions remain unchanged)
def get_center(box):
    """Calculates the center (cx, cy) of a cv2 bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def get_diag(box):
    """Calculates the diagonal length of a cv2 bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def get_distance(p1, p2):
    """Calculates the Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def yolo_to_cv2(yolo_box, img_shape):
    """Converts a YOLO tuple (cx_n, cy_n, w_n, h_n) to a cv2 box (x1, y1, x2, y2)."""
    img_h, img_w = img_shape
    cx_n, cy_n, w_n, h_n = yolo_box
    
    box_w = w_n * img_w
    box_h = h_n * img_h
    cx = cx_n * img_w
    cy = cy_n * img_h
    
    x1 = int(cx - box_w / 2)
    y1 = int(cy - box_h / 2)
    x2 = int(cx + box_w / 2)
    y2 = int(cy + box_h / 2)
    
    return (x1, y1, x2, y2)

def cv2_to_yolo(cv2_box, img_shape, class_id):
    """Converts a cv2 box (x1, y1, x2, y2) to a YOLO tuple (cid, cx_n, cy_n, w_n, h_n)."""
    img_h, img_w = img_shape
    x1, y1, x2, y2 = cv2_box
    
    box_w = x2 - x1
    box_h = y2 - y1
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = box_w / img_w
    h_n = box_h / img_h
    
    return (int(class_id), cx_n, cy_n, w_n, h_n)

def draw_labels(image, labels_to_draw, colors):
    """Draws colored bounding boxes on an image without text."""
    for class_id, box in labels_to_draw:
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(class_id, (255, 255, 255)) # Default to white
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1) # Thickness of 1
    return image

# --- Main Processing Function ---

def process_fused_labels(root_label_dir, root_image_dir, root_vis_dir):
    """
    Main function to read .txt labels, perform fusion logic,
    and save visualizations and new labels back into their
    respective subdirectories.
    """
    
    print("--- Starting Gene Fusion Processing (Multi-Directory) ---")
    
    # === 1. Setup Root Paths ===
    root_label_dir = Path(root_label_dir)
    root_image_dir = Path(root_image_dir)
    root_vis_dir = Path(root_vis_dir)
    
    root_vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading labels from:    {root_label_dir}")
    print(f"Reading images from:    {root_image_dir}")
    print(f"Saving visuals to:      {root_vis_dir}")

    # === 2. Find and Group Image & Label Files (ROBUST LOGIC) ===
    cells = defaultdict(dict)
    
    # Map of channel keywords to their file types
    channel_map = {
        'FITC': 'FITC_label',
        'ORANGE': 'ORANGE_label',
        'AQUA': 'AQUA_label',
        'SKY': 'SKY'
    }
    # Create a simple list of keywords for searching
    channel_keywords = list(channel_map.keys())
    
    print(f"Scanning for all files in: {root_image_dir} and {root_label_dir}")
    all_files = list(root_image_dir.rglob('*.png')) + list(root_label_dir.rglob('*.txt'))
    
    for file_path in all_files:
        if '_cell_' not in file_path.stem:
            continue
            
        try:
            # e.g., "A-25-119 BCRABL CELL1  FITC_cell_001"
            prefix, cell_num = file_path.stem.split('_cell_')
            # prefix = "A-25-119 BCRABL CELL1  FITC"
            # cell_num = "001"
        except ValueError:
            print(f"  Warning: Skipping malformed file name: {file_path.name}")
            continue
            
        file_type = None
        base_prefix = prefix # Start with the full prefix
        
        # Find which channel this file is
        for keyword in channel_keywords:
            if keyword in prefix:
                file_type = channel_map[keyword]
                
                # **ROBUST PART**: Remove the keyword AND any surrounding spaces
                base_prefix = re.sub(r'\s+' + re.escape(keyword) + r'\s*', ' ', prefix).strip()
                
                # We found the channel, no need to check others for this file
                break 
                
        # If we didn't find a channel keyword, skip this file
        if not file_type:
            continue
            
        # Reconstruct the unique, normalized cell name
        # e.g., "A-25-119 BCRABL CELL1_cell_001"
        unique_cell_name = f"{base_prefix}_cell_{cell_num}"
        
        # The final unique key is (parent_directory, unique_cell_name)
        # e.g., ('A-25-119', 'A-25-119 BCRABL CELL1_cell_001')
        cell_key = (file_path.parent.name, unique_cell_name)
        
        # --- Store the file path based on its type and extension ---
        if file_type == 'SKY' and file_path.suffix == '.png':
            cells[cell_key]['SKY'] = file_path
        elif file_type.endswith('_label') and file_path.suffix == '.txt':
            cells[cell_key][file_type] = file_path
        
            
    print(f"Found {len(cells)} unique cells to process across all subdirectories.")

    # === 3. Process Each Cell ===
    for cell_key, file_paths in cells.items():
        parent_name, unique_cell_name = cell_key
        print(f"\n--- Processing Cell: {unique_cell_name} in {parent_name} ---")
        
        # --- 3a. Check for SKY image (ESSENTIAL) ---
        if 'SKY' not in file_paths:
            print(f"   Skipping... Missing SKY image for overlay.")
            print(f"   (Looking for a file that normalizes to: {unique_cell_name} with 'SKY')")
            continue
            
        sky_path = file_paths['SKY'] 
        sky_img = cv2.imread(str(sky_path))
        if sky_img is None:
            print(f"   Error: Could not read SKY image at {sky_path}, skipping cell.")
            continue
        vis_img = sky_img.copy()
        img_h, img_w, _ = sky_img.shape

        # --- 3b. Read Detections from .txt Files ---
        all_detections = []
        label_key_map = {'FITC_label': 0, 'ORANGE_label': 1, 'AQUA_label': 2}
        
        # Check that we have the labels we need
        if 'FITC_label' not in file_paths:
            print(f"   Warning: Missing FITC label for this cell. Green signals will be absent.")
        if 'ORANGE_label' not in file_paths:
            print(f"   Warning: Missing ORANGE label for this cell. Red signals will be absent.")
            
        for label_key, class_id in label_key_map.items():
            if label_key in file_paths:
                txt_path = file_paths[label_key]
                try:
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cx_n, cy_n, w_n, h_n = map(float, parts[1:])
                            yolo_box = (cx_n, cy_n, w_n, h_n)
                            cv2_box = yolo_to_cv2(yolo_box, (img_h, img_w))
                            
                            all_detections.append({
                                'class_id': class_id,
                                'cv2_box': cv2_box,
                                'yolo_tuple': (class_id, cx_n, cy_n, w_n, h_n),
                                'idx': f"{class_id}_{i}"
                            })
                except Exception as e:
                    print(f"   Warning: Could not read or parse {txt_path}. Error: {e}")

        greens = [d for d in all_detections if d['class_id'] == 0]
        reds = [d for d in all_detections if d['class_id'] == 1]
        aquas = [d for d in all_detections if d['class_id'] == 2]
        
        print(f"   Initial signals read: {len(greens)} Green, {len(reds)} Red, {len(aquas)} Aqua")

        # --- 3c. Implement Fusion Logic ---
        pairs = []
        for g in greens:
            for r in reds:
                g_center = get_center(g['cv2_box'])
                r_center = get_center(r['cv2_box'])
                dist = get_distance(g_center, r_center)
                g_diag = get_diag(g['cv2_box'])
                r_diag = get_diag(r['cv2_box'])
                avg_diag = (g_diag + r_diag) / 2.0
                
                pairs.append({
                    'green': g, 'red': r, 'dist': dist, 'avg_diag': avg_diag
                })
        
        sorted_pairs = sorted(pairs, key=lambda p: p['dist'])
        
        final_labels_to_save = [] # List of YOLO tuples
        final_boxes_to_draw = []  # List of (class_id, cv2_box) tuples
        used_signal_indices = set()
        
        for pair in sorted_pairs:
            g, r = pair['green'], pair['red']
            if g['idx'] in used_signal_indices or r['idx'] in used_signal_indices:
                continue
                
            if pair['dist'] <= pair['avg_diag']:
                used_signal_indices.add(g['idx'])
                used_signal_indices.add(r['idx'])
                
                g_center, r_center = get_center(g['cv2_box']), get_center(r['cv2_box'])
                fusion_cx = (g_center[0] + r_center[0]) / 2.0
                fusion_cy = (g_center[1] + r_center[1]) / 2.0
                side_length = pair['avg_diag'] / math.sqrt(2)
                half_side = side_length / 2.0
                
                f_x1, f_y1 = int(fusion_cx - half_side), int(fusion_cy - half_side)
                f_x2, f_y2 = int(fusion_cx + half_side), int(fusion_cy + half_side)
                fusion_cv2_box = (f_x1, f_y1, f_x2, f_y2)
                
                # --- **MODIFIED**: Assign new FUSION class ID 3 ---
                fusion_yolo_tuple = cv2_to_yolo(fusion_cv2_box, (img_h, img_w), class_id=3)
                final_labels_to_save.append(fusion_yolo_tuple)
                final_boxes_to_draw.append((3, fusion_cv2_box))

        # --- 3d. Add all unused/original signals ---
        for g in greens:
            if g['idx'] not in used_signal_indices:
                final_labels_to_save.append(g['yolo_tuple'])
                final_boxes_to_draw.append((0, g['cv2_box']))
        for r in reds:
            if r['idx'] not in used_signal_indices:
                final_labels_to_save.append(r['yolo_tuple'])
                final_boxes_to_draw.append((1, r['cv2_box']))
        # Add original AQUA signals (Class 2)
        for a in aquas:
            final_labels_to_save.append(a['yolo_tuple'])
            final_boxes_to_draw.append((2, a['cv2_box']))
            
        print(f"   Final signals: {len(final_labels_to_save)} total.")

        # --- 3e. Save Visualization ---
        vis_img_final = draw_labels(vis_img, final_boxes_to_draw, COLORS)
        
        # Get the original SKY filename (e.g., "A-25-119...SKY_cell_001.png")
        output_filename_png = sky_path.name
        
        vis_save_dir = root_vis_dir / parent_name
        vis_save_dir.mkdir(parents=True, exist_ok=True)
        
        vis_save_path = vis_save_dir / output_filename_png
        cv2.imwrite(str(vis_save_path), vis_img_final)

        # --- 3f. Save YOLO Labels ---
        
        # Get the original SKY filename and change its extension to .txt
        output_filename_txt = sky_path.with_suffix('.txt').name
        
        label_save_dir = root_label_dir / parent_name
        label_save_dir.mkdir(parents=True, exist_ok=True) 
        
        label_save_path = label_save_dir / output_filename_txt
        with open(label_save_path, 'w') as f:
            for (cid, cx_n, cy_n, w_n, h_n) in final_labels_to_save:
                f.write(f"{cid} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")
                
        print(f"   Successfully saved outputs to:")
        print(f"     Label: {label_save_path}")
        print(f"     Visual: {vis_save_path}")

    print("\n--- Processing Complete ---")


# --- Main execution ---
if __name__ == "__main__":
    
    # *** IMPORTANT ***
    # --- PLEASE UPDATE THESE 3 ROOT PATHS ---
    
    # Path to the parent directory containing label subfolders (A-25-102, etc.)
    INPUT_LABEL_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/temp/labels"
    
    # Path to the parent directory containing image subfolders (A-25-102, etc.)
    INPUT_IMAGE_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025/cell_patches"
    
    # Path to the parent directory where visualization subfolders will be created
    FINAL_VIS_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/temp/visualizations"
    
    # ------------------------------------
    
    # Run the processing
    process_fused_labels(
        root_label_dir=INPUT_LABEL_DIR,
        root_image_dir=INPUT_IMAGE_DIR,
        root_vis_dir=FINAL_VIS_DIR
    )