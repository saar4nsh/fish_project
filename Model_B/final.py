import cv2

import numpy as np

import math

from pathlib import Path

from collections import defaultdict

import shutil

import os



# --- Configuration ---



# **NEW**: Path to the directory containing the pre-generated .txt labels

LABEL_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/13.09.2025/yolo_labels"



# **NEW**: Path to the directory containing the images (we only need SKY)

IMAGE_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/13.09.2025/cell_patches"



# **NEW**: Path for the final "fused" outputs

OUTPUT_DIR = "/home/cv2pr_ug_4/saaransh/fish_AI_images/13.09.2025/fused_results"



# Visualization colors

# 0: Green, 1: Red, 2: Aqua/Fusion

COLORS = {

    0: (0, 255, 0),  # Green

    1: (0, 0, 255),  # Red

    2: (255, 255, 0) # Aqua / Fusion (Cyan/Yellow)

}



# --- Helper Functions ---



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

       

        # Draw rectangle

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1) # Thickness of 2

    return image



# --- Main Processing Function ---

def process_fused_labels(label_dir, image_dir, vis_output_dir, label_output_dir):
    """
    Main function to read .txt labels, perform fusion logic,
    and save visualizations and new labels.
    """
    
    print("--- Starting Gene Fusion Processing from .txt Labels ---")
    
    # === 1. Setup Paths ===
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    vis_output_dir = Path(vis_output_dir)
    label_output_dir = Path(label_output_dir)
    
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output visualizations will be saved to: {vis_output_dir}")
    print(f"Output labels will be saved to: {label_output_dir}")

    # === 2. Find and Group Image & Label Files ===
    cells = defaultdict(dict)
    
    # Find SKY images first
    print(f"Scanning for SKY images in: {image_dir}")
    
    sky_images_found_in_dir = False
    
    # --- *** START BUGFIX *** ---
    # Updated file parsing logic
    for img_path in image_dir.rglob('*.png'):
        
        stem = img_path.stem # e.g., "A-25-2530_cell_006_SKY"
        
        # Partition on the *last* '_cell_'
        base_name, separator, cell_part = stem.rpartition('_cell_')
        if not separator:
            continue # No '_cell_' found, skip this file
            
        # cell_part is now "006_SKY"
        # Partition on the *last* '_' to separate number from channel
        cell_num, sep, channel = cell_part.rpartition('_')
        if not sep:
            continue # Unexpected format, e.g., "A-25-2530_cell_006"
            
        if channel != 'SKY':
            continue # This loop is only for SKY images
            
        cell_id = separator + cell_num # This is now "_cell_006"
        
        # The key is the parent dir name and the cell_id
        cell_key = (img_path.parent.name, cell_id) 
        
        cells[cell_key]['SKY'] = img_path
        cells[cell_key]['base_name'] = base_name # Store this for output naming
        sky_images_found_in_dir = True
    # --- *** END BUGFIX *** ---

    if not sky_images_found_in_dir:
        print(f"  No SKY images found in {image_dir}. Skipping this entire directory.")
        return # Exit the function

    # Find .txt labels
    print(f"Scanning for labels in: {label_dir}")
    
    # --- *** START BUGFIX *** ---
    # Updated file parsing logic
    for txt_path in label_dir.rglob('*.txt'):
        if 'fused' in txt_path.name:
            continue
            
        stem = txt_path.stem # e.g., "A-25-2530_cell_006_FITC"
        
        # Partition on the *last* '_cell_'
        base_name, separator, cell_part = stem.rpartition('_cell_')
        if not separator:
            continue 
            
        # cell_part is now "006_FITC"
        # Partition on the *last* '_' to separate number from channel
        cell_num, sep, channel = cell_part.rpartition('_')
        if not sep:
            continue # Unexpected format
            
        cell_id = separator + cell_num # This is now "_cell_006"
        cell_key = (txt_path.parent.name, cell_id)
        
        # Add the label path based on the parsed channel
        if channel == 'FITC':
            cells[cell_key]['FITC_label'] = txt_path
        elif channel == 'ORANGE':
            cells[cell_key]['ORANGE_label'] = txt_path
        elif channel == 'AQUA':
            cells[cell_key]['AQUA_label'] = txt_path
    # --- *** END BUGFIX *** ---
            
    print(f"Found {len(cells)} unique cells to process.")

    # === 3. Process Each Cell ===
    for cell_key, file_paths in cells.items():
        # **MODIFIED**: Use base_name from dict, fallback to parent_name
        # (This is more robust if files are in subfolders)
        parent_name = file_paths.get('base_name', cell_key[0])
        cell_num_id = cell_key[1]
        
        print(f"\n--- Processing Cell: {parent_name}{cell_num_id} ---")
        
        # --- 3a. Check for SKY image (ESSENTIAL) ---
        if 'SKY' not in file_paths:
            print(f"   Skipping... Missing SKY image for overlay.")
            continue
            
        sky_path = file_paths['SKY']
        sky_img = cv2.imread(str(sky_path))
        if sky_img is None:
            print(f"   Error: Could not read SKY image at {sky_path}, skipping cell.")
            continue
        vis_img = sky_img.copy()
        img_h, img_w, _ = sky_img.shape

        # --- 3b. Read Detections from .txt Files ---
        # (This section was already correct)
        all_detections = []
        label_key_map = {'FITC_label': 0, 'ORANGE_label': 1, 'AQUA_label': 2}
        
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

        # Separate detections by class
        greens = [d for d in all_detections if d['class_id'] == 0]
        reds = [d for d in all_detections if d['class_id'] == 1]
        aquas = [d for d in all_detections if d['class_id'] == 2]
        
        print(f"   Initial signals read: {len(greens)} Green, {len(reds)} Red, {len(aquas)} Aqua")

        # --- 3c. Implement Fusion Logic ---
        # (This section was already correct)
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
                    'green': g,
                    'red': r,
                    'dist': dist,
                    'avg_diag': avg_diag
                })
        
        sorted_pairs = sorted(pairs, key=lambda p: p['dist'])
        
        final_labels_to_save = [] 
        final_boxes_to_draw = [] 
        used_signal_indices = set()
        
        for pair in sorted_pairs:
            g = pair['green']
            r = pair['red']
            
            if g['idx'] in used_signal_indices or r['idx'] in used_signal_indices:
                continue
                
            if pair['dist'] <= pair['avg_diag']:
                used_signal_indices.add(g['idx'])
                used_signal_indices.add(r['idx'])
                
                g_center = get_center(g['cv2_box'])
                r_center = get_center(r['cv2_box'])
                
                fusion_cx = (g_center[0] + r_center[0]) / 2.0
                fusion_cy = (g_center[1] + r_center[1]) / 2.0
                side_length = pair['avg_diag'] / math.sqrt(2)
                half_side = side_length / 2.0
                
                f_x1 = int(fusion_cx - half_side)
                f_y1 = int(fusion_cy - half_side)
                f_x2 = int(fusion_cx + half_side)
                f_y2 = int(fusion_cy + half_side)
                fusion_cv2_box = (f_x1, f_y1, f_x2, f_y2)
                
                fusion_yolo_tuple = cv2_to_yolo(fusion_cv2_box, (img_h, img_w), class_id=2)
                
                final_labels_to_save.append(fusion_yolo_tuple)
                final_boxes_to_draw.append((2, fusion_cv2_box))

        # --- 3d. Add all unused/original signals ---
        # (This section was already correct)
        for g in greens:
            if g['idx'] not in used_signal_indices:
                final_labels_to_save.append(g['yolo_tuple'])
                final_boxes_to_draw.append((0, g['cv2_box']))
        for r in reds:
            if r['idx'] not in used_signal_indices:
                final_labels_to_save.append(r['yolo_tuple'])
                final_boxes_to_draw.append((1, r['cv2_box']))
        for a in aquas:
            final_labels_to_save.append(a['yolo_tuple'])
            final_boxes_to_draw.append((2, a['cv2_box']))
            
        print(f"   Final signals: {len(final_labels_to_save)} total.")

        # --- 3e. Save Visualization ---
        vis_img_final = draw_labels(vis_img, final_boxes_to_draw, COLORS)
        
        # **MODIFIED**: Use the same parent_name and cell_num_id from above
        output_name_base = f"{parent_name}{cell_num_id}_SKY_fused"
        vis_save_path = vis_output_dir / f"{output_name_base}.png"
        cv2.imwrite(str(vis_save_path), vis_img_final)

        # --- 3f. Save YOLO Labels ---
        label_save_path = label_output_dir / f"{output_name_base}.txt"
        with open(label_save_path, 'w') as f:
            for (cid, cx_n, cy_n, w_n, h_n) in final_labels_to_save:
                f.write(f"{cid} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")
                
        print(f"   Successfully saved outputs to: {output_name_base}.png / .txt")

    print("\n--- Processing Complete for this directory ---")

    
# --- Main execution ---
if __name__ == "__main__":
    
    # *** IMPORTANT ***
    # --- PLEASE UPDATE THESE PARENT PATHS ---
    
    # Path to the PARENT directory containing label sub-directories (e.g., .../yolo_labels)
    PARENT_LABEL_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/temp/labels"
    
    # Path to the PARENT directory containing image sub-directories (e.g., .../cell_patches)
    PARENT_IMAGE_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025/cell_patches"
    
    # Path to the PARENT directory where visualization sub-directories are (e.g., .../visualizations)
    PARENT_VIS_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/temp/visualizations" # User must confirm this path
    
    # ------------------------------------
    
    print("--- Starting Batch Processing ---")
    
    # Use Path objects
    p_label_dir = Path(PARENT_LABEL_DIR)
    p_image_dir = Path(PARENT_IMAGE_DIR)
    p_vis_dir = Path(PARENT_VIS_DIR)
    
    if not p_label_dir.is_dir():
        print(f"Error: PARENT_LABEL_DIR does not exist: {p_label_dir}")
        exit()
    if not p_image_dir.is_dir():
        print(f"Error: PARENT_IMAGE_DIR does not exist: {p_image_dir}")
        exit()
    
    # We will iterate through the sub-directories in the PARENT_LABEL_DIR
    # and assume a matching sub-directory exists in PARENT_IMAGE_DIR
    
    processed_count = 0
    skipped_count = 0
    
    # Iterate over all items in the parent label directory
    for sub_dir in p_label_dir.iterdir():
        # Check if the item is a directory
        if sub_dir.is_dir():
            sub_dir_name = sub_dir.name
            print(f"\n======================================")
            print(f"Found Sub-Directory: {sub_dir_name}")
            
            # 1. Define specific paths for this sub-directory
            
            # --- Input paths ---
            current_label_dir = sub_dir
            current_image_dir = p_image_dir / sub_dir_name
            
            # --- Output paths ---
            # Labels go back into the *input* label directory
            current_label_output_dir = current_label_dir 
            # Visualizations go into the parallel visualization directory
            current_vis_output_dir = p_vis_dir / sub_dir_name
            
            # 2. Check if the matching image directory exists
            if not current_image_dir.is_dir():
                print(f"  Failure: Matching image directory not found at: {current_image_dir}")
                print(f"  Skipping this sub-directory.")
                skipped_count += 1
                continue
                
            print(f"  -> Processing inputs from:")
            print(f"     Labels: {current_label_dir}")
            print(f"     Images: {current_image_dir}")
            print(f"  -> Saving outputs to:")
            print(f"     Labels: {current_label_output_dir}")
            print(f"     Visuals: {current_vis_output_dir}")
            
            # 3. Run the processing for this specific sub-directory
            try:
                process_fused_labels(
                    label_dir=str(current_label_dir),
                    image_dir=str(current_image_dir),
                    vis_output_dir=str(current_vis_output_dir),    # NEW
                    label_output_dir=str(current_label_output_dir) # NEW
                )
                processed_count += 1
            except Exception as e:
                print(f"     !!! ERROR processing {sub_dir_name}: {e}")
                skipped_count += 1
                
    print(f"\n======================================")
    print("--- Batch Processing Complete ---")
    print(f"Successfully processed: {processed_count} sub-directories.")
    print(f"Skipped / Errored: {skipped_count} sub-directories.")