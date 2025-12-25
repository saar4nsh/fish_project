import os
import cv2
import numpy as np
import re # For robust filename parsing

def find_mask_files(masks_root_dir):
    """ Finds all .png mask files in subdirectories. """
    mask_files = []
    for dirpath, _, filenames in os.walk(masks_root_dir):
        for fname in filenames:
            # Make sure it's a PNG and NOT ending in DAPI (just in case)
            if fname.endswith(".png") and not fname.endswith("DAPI.png"):
                full_path = os.path.join(dirpath, fname)
                mask_files.append(full_path)
    print(f"Found {len(mask_files)} mask files (.png).")
    return mask_files

def extract_cell_patches(masks_root_dir, images_root_dir, patches_root_dir):
    """
    Extracts cell patches based on color masks and saves them as PNG.
    """
    mask_paths = find_mask_files(masks_root_dir)
    os.makedirs(patches_root_dir, exist_ok=True) # Ensure base patches dir exists

    # Define the image types to process (excluding DAPI)
    target_suffixes = ["FITC.tif", "ORANGE.tif", "AQUA.tif", "SKY.tif"]

    for mask_path in mask_paths:
        print(f"\n--- Processing mask: {mask_path} ---")

        # --- 1. Load Color Mask and Find Cells ---
        color_mask = cv2.imread(mask_path) # Reads as BGR
        if color_mask is None:
            print(f"Error: Could not read mask file {mask_path}. Skipping.")
            continue

        unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)
        cell_colors = [color for color in unique_colors if color.any()] 

        if not cell_colors:
            print("  Info: No cells found in mask (only background). Skipping.")
            continue
        
        print(f"  Found {len(cell_colors)} unique cell instances.")

        # --- 2. Find Corresponding Image Files ---
        mask_rel_path = os.path.relpath(mask_path, masks_root_dir)
        mask_rel_dir = os.path.dirname(mask_rel_path)
        mask_basename = os.path.basename(mask_path)
        base_image_name = mask_basename.removesuffix(".png")

        corresponding_images = {}
        image_subdir = os.path.join(images_root_dir, mask_rel_dir)
        
        if not os.path.isdir(image_subdir):
             print(f"  Warning: Corresponding image subdir not found: {image_subdir}. Skipping.")
             continue

        for suffix in target_suffixes:
            found = False
            for fname in os.listdir(image_subdir):
                 if fname.startswith(base_image_name) and fname.endswith(suffix):
                      img_path = os.path.join(image_subdir, fname)
                      corresponding_images[suffix] = img_path
                      found = True
                      break 
            if not found:
                 print(f"  Warning: Corresponding image for suffix '{suffix}' not found in {image_subdir}")

        if not corresponding_images:
            print(f"  Warning: No corresponding images (FITC, ORANGE, etc.) found for base name '{base_image_name}'. Skipping.")
            continue

        # --- 3. Process Each Cell Instance ---
        cell_counter = 0 
        for color in cell_colors:
            cell_counter += 1
            cell_id_str = f"cell_{cell_counter:03d}" 
            
            # --- 3a. Create Binary Mask & Get Bounding Box ---
            lower_bound = np.array(color)
            upper_bound = np.array(color)
            binary_mask = cv2.inRange(color_mask, lower_bound, upper_bound) 

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"  Warning: No contour found for color {color}. Skipping cell {cell_id_str}.")
                continue

            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            print(f"    Processing {cell_id_str}: BBox(x={x}, y={y}, w={w}, h={h})")

            # --- 3b. Extract Patch from Each Corresponding Image ---
            for suffix, img_path in corresponding_images.items():
                
                original_tif = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if original_tif is None:
                    print(f"    Error: Could not read image {img_path}. Skipping patch for this suffix.")
                    continue
                
                y_end = min(y + h, original_tif.shape[0])
                x_end = min(x + w, original_tif.shape[1])
                patch = original_tif[y:y_end, x:x_end]

                # --- 3c. Prepare Patch for PNG Saving ---
                patch_to_save = patch
                # **NEW**: Normalize if not uint8 (essential for PNG saving)
                if patch.dtype != 'uint8':
                    # print(f"      Info: Normalizing {patch.dtype} patch to uint8 for PNG saving.") # Optional verbose output
                    patch_to_save = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # --- 3d. Determine Output Path and Save as PNG ---
                patch_subdir = os.path.join(patches_root_dir, mask_rel_dir)
                os.makedirs(patch_subdir, exist_ok=True) 

                original_filename_base = os.path.basename(img_path).removesuffix(".tif")
                
                # **NEW**: Change extension to .png
                patch_filename = f"{original_filename_base}_{cell_id_str}.png" 
                output_path = os.path.join(patch_subdir, patch_filename)

                try:
                    # Save the (potentially normalized) patch as PNG
                    cv2.imwrite(output_path, patch_to_save)
                    # print(f"      Saved patch: {output_path}") 
                except Exception as e:
                    print(f"    Error saving patch {output_path}: {e}")

# --- Main execution ---
if __name__ == "__main__":
    
    # --- PLEASE EDIT THESE VALUES ---
    # 1. Path to the ROOT folder (the one containing 'images', 'masks')
    ROOT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025" # Example
    # 2. Define the name for the new cell patches directory
    CELL_PATCHES_DIR_NAME = "cell_patches"
    # ----------------------------------

    # Define the directory paths
    MASKS_ROOT_DIR = os.path.join(ROOT_DIR, "cell_masks")
    IMAGES_ROOT_DIR = os.path.join(ROOT_DIR, "images")
    CELL_PATCHES_ROOT_DIR = os.path.join(ROOT_DIR, CELL_PATCHES_DIR_NAME)
    
    print(f"🔍 Reading masks from: {MASKS_ROOT_DIR}")
    print(f"🖼️ Reading images from: {IMAGES_ROOT_DIR}")
    print(f"💾 Saving PNG patches to: {CELL_PATCHES_ROOT_DIR}")
    
    extract_cell_patches(MASKS_ROOT_DIR, IMAGES_ROOT_DIR, CELL_PATCHES_ROOT_DIR)
    
    print("\n✅ Cell patch extraction complete.")