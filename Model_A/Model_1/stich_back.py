import os
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
ROOT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025" # Base directory 
SUBDIR_NAME = "A-25-136" # The specific subdirectory to process

# Input Directories
MASKS_ROOT_DIR = os.path.join(ROOT_DIR, "cell_masks")
IMAGES_ROOT_DIR = os.path.join(ROOT_DIR, "images")
LABELS_ROOT_DIR = os.path.join(ROOT_DIR, "labels") # YOLO .txt labels

# Output Directory
STITCHED_OUTPUT_DIR = os.path.join(ROOT_DIR, "stitched_images")

# File Naming & Image Type
IMAGE_TYPE_SUFFIX = "SKY" # Which original image type to draw on (e.g., FITC, ORANGE)

# Detection & Visualization Settings
MIN_CELL_AREA_PIXELS = 50 
GENE_CLASS_NAMES = {0: "Green",
                    1: "Red",
                    2: "Aqua",
                    3: "Fusion"} 

# Map class index to a BGR color tuple
BBOX_COLORS_BY_CLASS = {
    0: (0, 255, 0),    # Green
    1: (0, 0, 255),    # Red
    2: (255, 255, 0),  # Aqua/Cyan
    3: (0, 255, 255),  # Yellow
}
DEFAULT_BBOX_COLOR = (0, 0, 255) # Fallback color
TEXT_COLOR = (255, 255, 255) # White color for text
FONT_SCALE = 0.25
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BBOX_THICKNESS = 1
TEXT_THICKNESS = 1

CELL_BBOX_COLOR = (255, 255, 255) # White (BGR)
CELL_BBOX_THICKNESS = 1

# ... (after CELL_BBOX_THICKNESS) ...

# Per-Cell Count Text Settings
PER_CELL_COUNT_FONT_SCALE = 0.4
PER_CELL_COUNT_LINE_SPACING = 15    # Pixels between lines
PER_CELL_COUNT_X_INDENT = 5         # How far from the left edge of the box to indent
PER_CELL_COUNT_Y_START_OFFSET = 15  # How far *below the bottom* of the box to start
PER_CELL_COUNT_FONT_THICKNESS = 1

# --- Helper Function ---
def yolo_to_xyxy_abs(yolo_coords, patch_width, patch_height, patch_offset_x, patch_offset_y):
    """ Converts YOLO coords (relative to patch) to absolute xyxy coords (relative to original image). """
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_coords

    x_center_abs_patch = x_center_norm * patch_width
    y_center_abs_patch = y_center_norm * patch_height
    width_abs_patch = width_norm * patch_width
    height_abs_patch = height_norm * patch_height

    x_min_abs_patch = x_center_abs_patch - width_abs_patch / 2
    y_min_abs_patch = y_center_abs_patch - height_abs_patch / 2
    x_max_abs_patch = x_center_abs_patch + width_abs_patch / 2
    y_max_abs_patch = y_center_abs_patch + height_abs_patch / 2

    x_min_abs_orig = int(x_min_abs_patch + patch_offset_x)
    y_min_abs_orig = int(y_min_abs_patch + patch_offset_y)
    x_max_abs_orig = int(x_max_abs_patch + patch_offset_x)
    y_max_abs_orig = int(y_max_abs_patch + patch_offset_y)

    return x_min_abs_orig, y_min_abs_orig, x_max_abs_orig, y_max_abs_orig

# --- Worker Function ---
def process_single_image_case(base_filename, mask_path, image_subdir, label_subdir, output_subdir):
    """
    Processes a single image case (defined by base_filename) and saves the stitched result.
    """
    print(f"  --- Processing Case: {base_filename} ---")

    # --- 1. Find Corresponding Original Image ---
    original_image_path = None
    found_original = False
    target_tif_suffix = f"{IMAGE_TYPE_SUFFIX}.tif"
    
    try:
        for fname in os.listdir(image_subdir):
            # Check if fname starts with the base and ends with the desired suffix
            # This is more robust than just checking startswith/endswith separately
            # e.g., "A-25-102 BCL6 SKY.tif" matches base "A-25-102 BCL6"
            
            # We need to find the exact file. Let's assume the filename is
            # {base_filename} + {some_separator} + {IMAGE_TYPE_SUFFIX}.tif
            # Or just {base_filename} + {IMAGE_TYPE_SUFFIX}.tif
            
            # Let's try to find a file that *is* the base filename + suffix
            potential_fname = f"{base_filename} {IMAGE_TYPE_SUFFIX}.tif" # Common case with space
            potential_fname_no_space = f"{base_filename}{IMAGE_TYPE_SUFFIX}.tif"
            
            if fname == potential_fname or fname == potential_fname_no_space:
                 original_image_path = os.path.join(image_subdir, fname)
                 found_original = True
                 break
            
            # Fallback: if it starts with base and ends with suffix (less precise)
            if fname.startswith(base_filename) and fname.endswith(target_tif_suffix):
                original_image_path = os.path.join(image_subdir, fname)
                found_original = True
                break # Take the first match
                
    except FileNotFoundError:
        print(f"    Error: Image subdirectory not found: {image_subdir}. Skipping case.")
        return

    if not found_original or original_image_path is None:
        print(f"    Error: Original image ending with '{target_tif_suffix}' and matching base '{base_filename}' not found in {image_subdir}. Skipping case.")
        return

    # --- 2. Load Original Image and Mask ---
    print(f"    Loading original image: {original_image_path}")
    original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        print("    Error: Could not read original image. Skipping case.")
        return

    if original_image.dtype != 'uint8':
        image_to_draw_on = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        image_to_draw_on = original_image.copy()

    if len(image_to_draw_on.shape) == 2 or image_to_draw_on.shape[2] == 1:
        image_to_draw_on = cv2.cvtColor(image_to_draw_on, cv2.COLOR_GRAY2BGR)


    print(f"    Loading cell mask: {mask_path}")
    color_mask = cv2.imread(mask_path)
    if color_mask is None:
        print("    Error: Could not read color mask. Skipping case.")
        return

    # --- 3. Find Valid Cells in Mask ---
    unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)
    potential_cell_colors = [color for color in unique_colors if color.any()] 

    valid_cell_colors = []
    for color in potential_cell_colors:
        temp_binary_mask = cv2.inRange(color_mask, np.array(color), np.array(color))
        pixel_count = cv2.countNonZero(temp_binary_mask)
        if pixel_count >= MIN_CELL_AREA_PIXELS:
            valid_cell_colors.append(color)

    if not valid_cell_colors:
        print("    Info: No valid cells found in mask meeting area requirement. Saving original image.")
        # We can still save the converted image
        output_filename = f"{base_filename}_{IMAGE_TYPE_SUFFIX}_stitched.png"
        output_path = os.path.join(output_subdir, output_filename)
        cv2.imwrite(output_path, image_to_draw_on)
        print(f"    Saved image without detections to: {output_path}")
        return

    print(f"    Found {len(valid_cell_colors)} valid cells for processing.")

    # --- 4. Process Each Cell and its Detections ---
    cell_counter = 0
    total_detections_drawn = 0
    original_tif_filename_base = Path(original_image_path).stem # e.g., "A-25-102 BCL6 SKY"

    for color in valid_cell_colors:
        cell_counter += 1
        cell_id_str = f"cell_{cell_counter:03d}"

        # 4a. Get Cell BBox
        binary_mask = cv2.inRange(color_mask, np.array(color), np.array(color))
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        main_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(main_contour) 
        # --- DRAW THE CELL BBOX ---
        cv2.rectangle(image_to_draw_on, (px, py), (px + pw, py + ph), CELL_BBOX_COLOR, CELL_BBOX_THICKNESS)
        # ---
        # --- INITIALIZE PER-CELL COUNTER ---
        cell_detection_counts = {}

        # 4b. Find Corresponding YOLO Label File
        label_filename = f"{original_tif_filename_base}_{cell_id_str}.txt"
        label_path = os.path.join(label_subdir, label_filename)

        if not os.path.exists(label_path):
            continue

        # 4c. Read Detections and Convert Coordinates
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"      Error reading label file {label_path}: {e}. Skipping cell.")
            continue

        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) < 5: continue 

                class_idx = int(parts[0])
                # --- INCREMENT COUNT FOR THIS CLASS ---
                cell_detection_counts[class_idx] = cell_detection_counts.get(class_idx, 0) + 1
                confidence = float(parts[5]) if len(parts) > 5 else 1.0 
                yolo_coords = tuple(map(float, parts[1:5])) 

                x_min, y_min, x_max, y_max = yolo_to_xyxy_abs(yolo_coords, pw, ph, px, py)

                # 4d. Draw Detection on Original Image
                class_name = GENE_CLASS_NAMES.get(class_idx, f"CLS_{class_idx}")
                label = f"{class_name} {confidence:.2f}"
                
                current_bbox_color = BBOX_COLORS_BY_CLASS.get(class_idx, DEFAULT_BBOX_COLOR)

                cv2.rectangle(image_to_draw_on, (x_min, y_min), (x_max, y_max), current_bbox_color, BBOX_THICKNESS)

                # (text_width, text_height), baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, TEXT_THICKNESS)
                # text_y_start = max(y_min - text_height - baseline, 0)
                # cv2.rectangle(image_to_draw_on, (x_min, text_y_start), (x_min + text_width, y_min), current_bbox_color, -1)
                # cv2.putText(image_to_draw_on, label, (x_min, y_min - baseline), FONT_FACE, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

                total_detections_drawn += 1
            
            except Exception as e:
                print(f"      Error processing line '{line.strip()}' in {label_path}: {e}")

        # --- 4e. Write Per-Cell Counts ---
        # Start text *below* the cell's bounding box
        
        # X position: Indent from the left edge of the box
        start_x = px + PER_CELL_COUNT_X_INDENT 
        
        # Y position: Start *below* the bottom edge of the box
        # py + ph is the bottom edge
        current_y = py + ph + PER_CELL_COUNT_Y_START_OFFSET 

        for class_idx in sorted(cell_detection_counts.keys()):
            class_name = GENE_CLASS_NAMES.get(class_idx, f"CLS_{class_idx}")
            count = cell_detection_counts[class_idx]
            
            text = f"{class_name}: {count}"
            text_color = BBOX_COLORS_BY_CLASS.get(class_idx, DEFAULT_BBOX_COLOR)
            
            cv2.putText(image_to_draw_on, 
                        text, 
                        (start_x, current_y), # Use the calculated start_x and updating current_y
                        FONT_FACE, 
                        PER_CELL_COUNT_FONT_SCALE, 
                        text_color, 
                        PER_CELL_COUNT_FONT_THICKNESS, 
                        cv2.LINE_AA)
            
            # Move Y down for the next line
            current_y += PER_CELL_COUNT_LINE_SPACING

    print(f"    Drew {total_detections_drawn} total gene detections.")
    
    # ---

    # --- 5. Save the Final Stitched Image ---
    output_filename = f"{base_filename}_{IMAGE_TYPE_SUFFIX}_stitched.png" # Save as PNG
    output_path = os.path.join(output_subdir, output_filename)
    try:
        cv2.imwrite(output_path, image_to_draw_on)
        print(f"  ✅ Successfully saved stitched image to: {output_path}")
    except Exception as e:
        print(f"  Error saving final image {output_path}: {e}")


# --- Main "Controller" Function ---
def process_all_images_in_subdir():
    
    print(f"--- Starting Stitcher for Subdirectory: {SUBDIR_NAME} ---")

    # --- 1. Construct Paths ---
    mask_subdir = os.path.join(MASKS_ROOT_DIR, SUBDIR_NAME)
    image_subdir = os.path.join(IMAGES_ROOT_DIR, SUBDIR_NAME)
    label_subdir = os.path.join(LABELS_ROOT_DIR, SUBDIR_NAME)
    output_subdir = os.path.join(STITCHED_OUTPUT_DIR, SUBDIR_NAME)
    
    try:
        os.makedirs(output_subdir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {output_subdir}: {e}")
        return
        
    # --- 2. Find all PNG mask files to determine cases ---
    try:
        mask_files = [f for f in os.listdir(mask_subdir) if f.lower().endswith(".png")]
    except FileNotFoundError:
        print(f"Error: Mask subdirectory not found: {mask_subdir}. Cannot proceed.")
        return

    if not mask_files:
        print(f"Info: No PNG mask files found in {mask_subdir}.")
        return

    print(f"Found {len(mask_files)} potential image cases to process.")

    # --- 3. Loop and process each case ---
    for mask_filename in mask_files:
        try:
            base_filename = mask_filename.removesuffix(".png")
            mask_path = os.path.join(mask_subdir, mask_filename)
            
            # Call the worker function for this specific case
            process_single_image_case(
                base_filename=base_filename,
                mask_path=mask_path,
                image_subdir=image_subdir,
                label_subdir=label_subdir,
                output_subdir=output_subdir
            )
        except Exception as e:
            print(f"!! Unhandled error while processing {mask_filename}: {e}. Skipping to next file.")
            
    print(f"--- Finished processing all cases in {SUBDIR_NAME} ---")


# --- Run the script ---
if __name__ == "__main__":
    process_all_images_in_subdir()

# # --- Worker Function ---
# def process_single_image_case(base_filename, mask_path, image_subdir, label_subdir, output_subdir):
#     """
#     Processes a single image case (defined by base_filename) and saves the stitched result.
#     """
#     print(f"  --- Processing Case: {base_filename} ---")

#     # --- 1. Find Corresponding Original Image ---
#     original_image_path = None
#     found_original = False
#     target_tif_suffix = f"{IMAGE_TYPE_SUFFIX}.tif"
    
#     try:
#         for fname in os.listdir(image_subdir):
#             # Check if fname starts with the base and ends with the desired suffix
#             # This is more robust than just checking startswith/endswith separately
#             # e.g., "A-25-102 BCL6 SKY.tif" matches base "A-25-102 BCL6"
            
#             # We need to find the exact file. Let's assume the filename is
#             # {base_filename} + {some_separator} + {IMAGE_TYPE_SUFFIX}.tif
#             # Or just {base_filename} + {IMAGE_TYPE_SUFFIX}.tif
            
#             # Let's try to find a file that *is* the base filename + suffix
#             potential_fname = f"{base_filename} {IMAGE_TYPE_SUFFIX}.tif" # Common case with space
#             potential_fname_no_space = f"{base_filename}{IMAGE_TYPE_SUFFIX}.tif"
            
#             if fname == potential_fname or fname == potential_fname_no_space:
#                  original_image_path = os.path.join(image_subdir, fname)
#                  found_original = True
#                  break
            
#             # Fallback: if it starts with base and ends with suffix (less precise)
#             if fname.startswith(base_filename) and fname.endswith(target_tif_suffix):
#                 original_image_path = os.path.join(image_subdir, fname)
#                 found_original = True
#                 break # Take the first match
                
#     except FileNotFoundError:
#         print(f"    Error: Image subdirectory not found: {image_subdir}. Skipping case.")
#         return

#     if not found_original or original_image_path is None:
#         print(f"    Error: Original image ending with '{target_tif_suffix}' and matching base '{base_filename}' not found in {image_subdir}. Skipping case.")
#         return

#     # --- 2. Load Original Image and Mask ---
#     print(f"    Loading original image: {original_image_path}")
#     original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
#     if original_image is None:
#         print("    Error: Could not read original image. Skipping case.")
#         return

#     if original_image.dtype != 'uint8':
#         image_to_draw_on = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     else:
#         image_to_draw_on = original_image.copy()

#     if len(image_to_draw_on.shape) == 2 or image_to_draw_on.shape[2] == 1:
#         image_to_draw_on = cv2.cvtColor(image_to_draw_on, cv2.COLOR_GRAY2BGR)


#     print(f"    Loading cell mask: {mask_path}")
#     color_mask = cv2.imread(mask_path)
#     if color_mask is None:
#         print("    Error: Could not read color mask. Skipping case.")
#         return

#     # --- 3. Find Valid Cells in Mask ---
#     unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)
#     potential_cell_colors = [color for color in unique_colors if color.any()] 

#     valid_cell_colors = []
#     for color in potential_cell_colors:
#         temp_binary_mask = cv2.inRange(color_mask, np.array(color), np.array(color))
#         pixel_count = cv2.countNonZero(temp_binary_mask)
#         if pixel_count >= MIN_CELL_AREA_PIXELS:
#             valid_cell_colors.append(color)

#     if not valid_cell_colors:
#         print("    Info: No valid cells found in mask meeting area requirement. Saving original image.")
#         # We can still save the converted image
#         output_filename = f"{base_filename}_{IMAGE_TYPE_SUFFIX}_stitched.png"
#         output_path = os.path.join(output_subdir, output_filename)
#         cv2.imwrite(output_path, image_to_draw_on)
#         print(f"    Saved image without detections to: {output_path}")
#         return

#     print(f"    Found {len(valid_cell_colors)} valid cells for processing.")

#     # --- 4. Process Each Cell and its Detections ---
#     cell_counter = 0
#     total_detections_drawn = 0
#     original_tif_filename_base = Path(original_image_path).stem # e.g., "A-25-102 BCL6 SKY"

#     for color in valid_cell_colors:
#         cell_counter += 1
#         cell_id_str = f"cell_{cell_counter:03d}"

#         # 4a. Get Cell BBox
#         binary_mask = cv2.inRange(color_mask, np.array(color), np.array(color))
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours: continue
#         main_contour = max(contours, key=cv2.contourArea)
#         px, py, pw, ph = cv2.boundingRect(main_contour) 
#         # --- DRAW THE CELL BBOX ---
#         cv2.rectangle(image_to_draw_on, (px, py), (px + pw, py + ph), CELL_BBOX_COLOR, CELL_BBOX_THICKNESS)
#         # ---
#         # --- INITIALIZE PER-CELL COUNTER ---
#         cell_detection_counts = {}

#         # 4b. Find Corresponding YOLO Label File
#         label_filename = f"{original_tif_filename_base}_{cell_id_str}.txt"
#         label_path = os.path.join(label_subdir, label_filename)

#         if not os.path.exists(label_path):
#             continue

#         # 4c. Read Detections and Convert Coordinates
#         try:
#             with open(label_path, 'r') as f:
#                 lines = f.readlines()
#         except Exception as e:
#             print(f"      Error reading label file {label_path}: {e}. Skipping cell.")
#             continue

#         for line in lines:
#             try:
#                 parts = line.strip().split()
#                 if len(parts) < 5: continue 

#                 class_idx = int(parts[0])
#                 # --- INCREMENT COUNT FOR THIS CLASS ---
#                 cell_detection_counts[class_idx] = cell_detection_counts.get(class_idx, 0) + 1
#                 confidence = float(parts[5]) if len(parts) > 5 else 1.0 
#                 yolo_coords = tuple(map(float, parts[1:5])) 

#                 x_min, y_min, x_max, y_max = yolo_to_xyxy_abs(yolo_coords, pw, ph, px, py)

#                 # 4d. Draw Detection on Original Image
#                 class_name = GENE_CLASS_NAMES.get(class_idx, f"CLS_{class_idx}")
#                 label = f"{class_name} {confidence:.2f}"
                
#                 current_bbox_color = BBOX_COLORS_BY_CLASS.get(class_idx, DEFAULT_BBOX_COLOR)

#                 cv2.rectangle(image_to_draw_on, (x_min, y_min), (x_max, y_max), current_bbox_color, BBOX_THICKNESS)

#                 # (text_width, text_height), baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, TEXT_THICKNESS)
#                 # text_y_start = max(y_min - text_height - baseline, 0)
#                 # cv2.rectangle(image_to_draw_on, (x_min, text_y_start), (x_min + text_width, y_min), current_bbox_color, -1)
#                 # cv2.putText(image_to_draw_on, label, (x_min, y_min - baseline), FONT_FACE, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

#                 total_detections_drawn += 1
            
#             except Exception as e:
#                 print(f"      Error processing line '{line.strip()}' in {label_path}: {e}")

#     print(f"    Drew {total_detections_drawn} total gene detections.")
    

#     # --- 5. Save the Final Stitched Image ---
#     output_filename = f"{base_filename}_{IMAGE_TYPE_SUFFIX}_stitched.png" # Save as PNG
#     output_path = os.path.join(output_subdir, output_filename)
#     try:
#         cv2.imwrite(output_path, image_to_draw_on)
#         print(f"  ✅ Successfully saved stitched image to: {output_path}")
#     except Exception as e:
#         print(f"  Error saving final image {output_path}: {e}")


# --- Main "Controller" Function ---
def process_all_images_in_subdir():
    
    print(f"--- Starting Stitcher for Subdirectory: {SUBDIR_NAME} ---")

    # --- 1. Construct Paths ---
    mask_subdir = os.path.join(MASKS_ROOT_DIR, SUBDIR_NAME)
    image_subdir = os.path.join(IMAGES_ROOT_DIR, SUBDIR_NAME)
    label_subdir = os.path.join(LABELS_ROOT_DIR, SUBDIR_NAME)
    output_subdir = os.path.join(STITCHED_OUTPUT_DIR, SUBDIR_NAME)
    
    try:
        os.makedirs(output_subdir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {output_subdir}: {e}")
        return
        
    # --- 2. Find all PNG mask files to determine cases ---
    try:
        mask_files = [f for f in os.listdir(mask_subdir) if f.lower().endswith(".png")]
    except FileNotFoundError:
        print(f"Error: Mask subdirectory not found: {mask_subdir}. Cannot proceed.")
        return

    if not mask_files:
        print(f"Info: No PNG mask files found in {mask_subdir}.")
        return

    print(f"Found {len(mask_files)} potential image cases to process.")

    # --- 3. Loop and process each case ---
    for mask_filename in mask_files:
        try:
            base_filename = mask_filename.removesuffix(".png")
            mask_path = os.path.join(mask_subdir, mask_filename)
            
            # Call the worker function for this specific case
            process_single_image_case(
                base_filename=base_filename,
                mask_path=mask_path,
                image_subdir=image_subdir,
                label_subdir=label_subdir,
                output_subdir=output_subdir
            )
        except Exception as e:
            print(f"!! Unhandled error while processing {mask_filename}: {e}. Skipping to next file.")
            
    print(f"--- Finished processing all cases in {SUBDIR_NAME} ---")


# --- Run the script ---
if __name__ == "__main__":
    process_all_images_in_subdir()