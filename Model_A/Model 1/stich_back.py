import os
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
ROOT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025" # Base directory 
SUBDIR_NAME = "A-25-102" # The specific subdirectory to process

# Input Directories
MASKS_ROOT_DIR = os.path.join(ROOT_DIR, "cell_masks")
IMAGES_ROOT_DIR = os.path.join(ROOT_DIR, "images")
LABELS_ROOT_DIR = os.path.join(ROOT_DIR, "yolo_labels") # YOLO .txt labels

# Output Directory
STITCHED_OUTPUT_DIR = os.path.join(ROOT_DIR, "stitched_images")

# File Naming & Image Type
# Determine the base filename within the subdir (e.g., "A-25-102 BCL6")
# We'll try to find this automatically, assuming there's a mask file present
BASE_FILENAME = None # Will be determined automatically
IMAGE_TYPE_SUFFIX = "FITC" # Which original image type to draw on (e.g., FITC, ORANGE)

# Detection & Visualization Settings
MIN_CELL_AREA_PIXELS = 50 # Must match the threshold used in patch extraction
GENE_CLASS_NAMES = {0: "Gene"} # Map class index (from YOLO txt) to name. !!! EDIT IF NEEDED !!!
BBOX_COLOR = (0, 0, 255) # Red color for gene bounding box (BGR)
TEXT_COLOR = (255, 255, 255) # White color for text
FONT_SCALE = 0.25
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BBOX_THICKNESS = 1
TEXT_THICKNESS = 1

# --- Helper Function ---
def yolo_to_xyxy_abs(yolo_coords, patch_width, patch_height, patch_offset_x, patch_offset_y):
    """ Converts YOLO coords (relative to patch) to absolute xyxy coords (relative to original image). """
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_coords

    # Absolute coordinates within the patch
    x_center_abs_patch = x_center_norm * patch_width
    y_center_abs_patch = y_center_norm * patch_height
    width_abs_patch = width_norm * patch_width
    height_abs_patch = height_norm * patch_height

    # Calculate absolute xyxy within the patch
    x_min_abs_patch = x_center_abs_patch - width_abs_patch / 2
    y_min_abs_patch = y_center_abs_patch - height_abs_patch / 2
    x_max_abs_patch = x_center_abs_patch + width_abs_patch / 2
    y_max_abs_patch = y_center_abs_patch + height_abs_patch / 2

    # Add patch offset to get absolute coordinates on the original image
    x_min_abs_orig = int(x_min_abs_patch + patch_offset_x)
    y_min_abs_orig = int(y_min_abs_patch + patch_offset_y)
    x_max_abs_orig = int(x_max_abs_patch + patch_offset_x)
    y_max_abs_orig = int(y_max_abs_patch + patch_offset_y)

    return x_min_abs_orig, y_min_abs_orig, x_max_abs_orig, y_max_abs_orig

# --- Main Stitching Function ---
def stitch_image_with_detections():
    global BASE_FILENAME # Allow modification of the global variable

    print(f"--- Processing Subdirectory: {SUBDIR_NAME} ---")

    # --- 1. Construct Paths ---
    mask_subdir = os.path.join(MASKS_ROOT_DIR, SUBDIR_NAME)
    image_subdir = os.path.join(IMAGES_ROOT_DIR, SUBDIR_NAME)
    label_subdir = os.path.join(LABELS_ROOT_DIR, SUBDIR_NAME)
    output_subdir = os.path.join(STITCHED_OUTPUT_DIR, SUBDIR_NAME)
    os.makedirs(output_subdir, exist_ok=True)

    # --- 2. Find Base Filename and Required Files ---
    mask_path = None
    original_image_path = None

    # Find the first PNG mask file in the mask subdirectory to determine BASE_FILENAME
    try:
        for fname in os.listdir(mask_subdir):
            if fname.lower().endswith(".png"):
                mask_path = os.path.join(mask_subdir, fname)
                BASE_FILENAME = fname.removesuffix(".png") # e.g., "A-25-102 BCL6"
                print(f"  Base filename determined from mask: {BASE_FILENAME}")
                break # Use the first one found
    except FileNotFoundError:
        print(f"Error: Mask subdirectory not found: {mask_subdir}. Cannot proceed.")
        return

    if BASE_FILENAME is None or mask_path is None:
        print(f"Error: No PNG mask file found in {mask_subdir} to determine base filename.")
        return

    # Construct the path to the specific original image type (e.g., FITC.tif)
    # Be robust to potential spaces before suffix
    found_original = False
    target_tif_suffix = f"{IMAGE_TYPE_SUFFIX}.tif"
    try:
        for fname in os.listdir(image_subdir):
             if fname.startswith(BASE_FILENAME) and fname.endswith(target_tif_suffix):
                  original_image_path = os.path.join(image_subdir, fname)
                  found_original = True
                  print(f"  Found original image: {original_image_path}")
                  break
    except FileNotFoundError:
         print(f"Error: Image subdirectory not found: {image_subdir}. Cannot proceed.")
         return

    if not found_original:
        print(f"Error: Original image ending with '{target_tif_suffix}' not found in {image_subdir}.")
        return

    # --- 3. Load Original Image and Mask ---
    print(f"  Loading original image: {original_image_path}")
    original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        print("Error: Could not read original image.")
        return

    # Prepare image for drawing (needs to be 8-bit BGR)
    if original_image.dtype != 'uint8':
        image_to_draw_on = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        image_to_draw_on = original_image.copy() # Make a copy if already uint8

    if len(image_to_draw_on.shape) == 2 or image_to_draw_on.shape[2] == 1:
        image_to_draw_on = cv2.cvtColor(image_to_draw_on, cv2.COLOR_GRAY2BGR)


    print(f"  Loading cell mask: {mask_path}")
    color_mask = cv2.imread(mask_path)
    if color_mask is None:
        print("Error: Could not read color mask.")
        return

    # --- 4. Find Valid Cells in Mask ---
    unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)
    potential_cell_colors = [color for color in unique_colors if color.any()]

    valid_cell_colors = []
    for color in potential_cell_colors:
        lower_bound = np.array(color)
        upper_bound = np.array(color)
        temp_binary_mask = cv2.inRange(color_mask, lower_bound, upper_bound)
        pixel_count = cv2.countNonZero(temp_binary_mask)
        if pixel_count >= MIN_CELL_AREA_PIXELS:
            valid_cell_colors.append(color)

    if not valid_cell_colors:
        print("  Info: No valid cells found in mask meeting area requirement.")
        # Save the (potentially converted) original image without boxes?
        # output_filename = f"{BASE_FILENAME}_{IMAGE_TYPE_SUFFIX}_stitched.png"
        # output_path = os.path.join(output_subdir, output_filename)
        # cv2.imwrite(output_path, image_to_draw_on)
        # print(f"  Saved image without detections to: {output_path}")
        return

    print(f"  Found {len(valid_cell_colors)} valid cells for processing.")

    # --- 5. Process Each Cell and its Detections ---
    cell_counter = 0
    total_detections_drawn = 0
    for color in valid_cell_colors:
        cell_counter += 1
        cell_id_str = f"cell_{cell_counter:03d}"

        # --- 5a. Get Cell BBox (Patch Coordinates) ---
        lower_bound = np.array(color)
        upper_bound = np.array(color)
        binary_mask = cv2.inRange(color_mask, lower_bound, upper_bound)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        main_contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(main_contour) # Patch coords (x, y, width, height)

        # --- 5b. Find Corresponding YOLO Label File ---
        # Need the original tif filename base for the label file name
        original_tif_filename_base = Path(original_image_path).stem
        label_filename = f"{original_tif_filename_base}_{cell_id_str}.txt"
        label_path = os.path.join(label_subdir, label_filename)

        if not os.path.exists(label_path):
            # print(f"    Info: Label file not found for {cell_id_str}: {label_path}. Skipping cell.")
            continue

        # --- 5c. Read Detections and Convert Coordinates ---
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"    Error reading label file {label_path}: {e}. Skipping cell.")
            continue

        detections_in_cell = 0
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) < 5: continue # Ensure valid line

                class_idx = int(parts[0])
                # Assume confidence might be the last element if present (like from --save-conf)
                confidence = float(parts[5]) if len(parts) > 5 else 1.0 # Default to 1.0 if not saved
                yolo_coords = tuple(map(float, parts[1:5])) # x_center, y_center, width, height

                # Convert YOLO coords (patch relative) to absolute xyxy (original image relative)
                x_min, y_min, x_max, y_max = yolo_to_xyxy_abs(yolo_coords, pw, ph, px, py)

                # --- 5d. Draw Detection on Original Image ---
                class_name = GENE_CLASS_NAMES.get(class_idx, f"CLS_{class_idx}") # Get name or use index
                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(image_to_draw_on, (x_min, y_min), (x_max, y_max), BBOX_COLOR, BBOX_THICKNESS)

                (text_width, text_height), baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, TEXT_THICKNESS)
                # Ensure text background doesn't go off-image top
                text_y_start = max(y_min - text_height - baseline, 0)
                cv2.rectangle(image_to_draw_on, (x_min, text_y_start), (x_min + text_width, y_min), BBOX_COLOR, -1)
                cv2.putText(image_to_draw_on, label, (x_min, y_min - baseline), FONT_FACE, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

                detections_in_cell += 1
                total_detections_drawn += 1

            except Exception as e:
                print(f"    Error processing line '{line.strip()}' in {label_path}: {e}")
        # print(f"    Processed {detections_in_cell} detections for {cell_id_str}") # Optional verbose

    print(f"  Drew {total_detections_drawn} total gene detections.")

    # --- 6. Save the Final Stitched Image ---
    output_filename = f"{BASE_FILENAME}_{IMAGE_TYPE_SUFFIX}_stitched.png" # Save as PNG
    output_path = os.path.join(output_subdir, output_filename)
    try:
        cv2.imwrite(output_path, image_to_draw_on)
        print(f"✅ Successfully saved stitched image to: {output_path}")
    except Exception as e:
        print(f"Error saving final image {output_path}: {e}")


# --- Run the script ---
if __name__ == "__main__":
    stitch_image_with_detections()