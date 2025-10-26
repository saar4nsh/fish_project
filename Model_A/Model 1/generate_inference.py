print("--- 1. Script started")
import os
print("os")
import cv2
print("cv2")
import torch
print("torch")
import numpy as np
print("numpy")
from detectron2.config import get_cfg
print("get_cfg")
from detectron2 import model_zoo
print("model_zoo")
from detectron2.engine import DefaultPredictor
print("DefaultPredictor")
#from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.colormap import random_color
print("last")
print("imports loaded. ---")

def setup_predictor(config_file, weights_file, threshold=0.7):
    """
    Loads a Detectron2 model from a config and weights file.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    # !!! EDIT THIS to match your trained model !!!
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # e.g., 1 for "cell"
    
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.MODEL.DEVICE}")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)
    return predictor

def find_dapi_files(images_root_dir):
    """
    Finds all .tif files ending in 'DAPI.tif' in all subdirectories
    of the 'images_root_dir'.
    """
    dapi_files = []
    for dirpath, _, filenames in os.walk(images_root_dir):
        for fname in filenames:
            if fname.endswith("DAPI.tif"):
                full_path = os.path.join(dirpath, fname)
                dapi_files.append(full_path)
    print(f"Found {len(dapi_files)} 'DAPI.tif' files.")
    return dapi_files

def run_batch_prediction_and_save_color_masks(predictor, images_root_dir, masks_root_dir):
    """
    Runs prediction on all 'DAPI.tif' files and saves color masks
    to the corresponding 'masks' subdirectory.
    """
    image_paths = find_dapi_files(images_root_dir)
    
    for image_path in image_paths:
        print(f"--- Processing: {image_path} ---")
        
       # --- 1. Read and Prepare Image ---
        original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if original_img is None:
            print(f"Error: Could not read image {image_path}. Skipping.")
            continue

        img_for_pred = original_img # Use a different variable name for clarity

        # Check if the image needs normalization to uint8
        print(f"  Input dtype: {img_for_pred.dtype}") 
        if img_for_pred.dtype != 'uint8':
            print(f"  Info: Input dtype is {img_for_pred.dtype}. Normalizing to 8-bit (uint8).")
            img_for_pred = cv2.normalize(img_for_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # We normalized, so ensure 3 channels
            if len(img_for_pred.shape) == 2 or img_for_pred.shape[2] == 1:
                print("  Info: Converting normalized grayscale to 3-channel BGR.")
                img_for_pred = cv2.cvtColor(img_for_pred, cv2.COLOR_GRAY2BGR)
            # We MUST copy after normalization/conversion
            img_for_pred = img_for_pred.copy()

        else: # Input was already uint8
            # Ensure it's 3 channels
            if len(img_for_pred.shape) == 2 or img_for_pred.shape[2] == 1:
                print("  Info: Converting uint8 grayscale to 3-channel BGR.")
                img_for_pred = cv2.cvtColor(img_for_pred, cv2.COLOR_GRAY2BGR)
            # We MUST copy even if it was already uint8 BGR
            img_for_pred = img_for_pred.copy()

        # At this point, img_for_pred is GUARANTEED to be contiguous 8-bit BGR (uint8)
            
        # --- 2. Run Prediction ---
        try:
            outputs = predictor(img_for_pred)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            print(f"Error during prediction for {image_path}: {e}. Skipping.")
            continue
            
        # --- 3. Visualize and Save Color Mask ---
        
        # Create a blank (black) image of the same size to draw masks on
        mask_background = np.zeros(img_for_pred.shape, dtype=np.uint8)
        color_mask_image = mask_background

        if not instances.has("pred_masks") or len(instances) == 0:
            print("  Info: No instances found. Saving an empty black mask.")

        else:
            # Draw colored instance masks on the black background
            # v = Visualizer(
            #     mask_background,
            #     scale=1.0,
            #     instance_mode=ColorMode.SEGMENTATION
            # )
            # v = v.draw_instance_predictions(instances)
            # color_mask_image = v.get_image() # This is the RGB image
            print(f"  Info: Found {len(instances)} instances. Generating manual color mask...")
            masks = instances.pred_masks.numpy() # Get masks as a numpy array [N, H, W]

            # Generate a unique color for each instance and draw it
            for i in range(len(masks)):
                # Generate a random color (RGB, range 0-1)
                color_rgb_float = random_color(rgb=True, maximum=1)
                # Convert to BGR uint8 (0-255) for OpenCV
                color_bgr_uint8 = (np.array(color_rgb_float[::-1]) * 255).astype(np.uint8)

                # Ensure color is not black (or very close to it)
                while np.all(color_bgr_uint8 < 10): # Regenerate if too dark
                     color_rgb_float = random_color(rgb=True, maximum=1)
                     color_bgr_uint8 = (np.array(color_rgb_float[::-1]) * 255).astype(np.uint8)

                # Get the boolean mask for the current instance
                current_mask = masks[i] # Shape [H, W]

                # Use boolean indexing to apply the color to the mask image
                # This ensures solid colors without anti-aliasing
                color_mask_image[current_mask] = color_bgr_uint8

        # --- 4. Determine NEW Output Path and Save ---
        
        # Get the relative path of the image from the 'images' folder
        # e.g., "sub_dir_1/fooDAPI.tif"
        relative_path = os.path.relpath(image_path, images_root_dir)
        
        # Get the relative directory part
        # e.g., "sub_dir_1"
        relative_dir = os.path.dirname(relative_path)
        
        # Create the new output directory inside the 'masks' folder
        # e.g., "/path/to/root_dir/masks/sub_dir_1"
        output_sub_dir = os.path.join(masks_root_dir, relative_dir)
        os.makedirs(output_sub_dir, exist_ok=True)
        
        # Get the original base filename
        base_name = os.path.basename(image_path)
        
        # Create the new mask filename
        # e.g., "foo.png"
        new_base_name = base_name.removesuffix("DAPI.tif") + ".png"
        
        # Create the final output path
        output_path = os.path.join(output_sub_dir, new_base_name)
        
        # Convert Visualizer's RGB output to BGR for cv2.imwrite
        # color_mask_image_bgr = cv2.cvtColor(color_mask_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, color_mask_image)
        print(f"  Success: Saved color mask to {output_path}\n")

# --- Main execution ---
if __name__ == "__main__":
    
    # --- PLEASE EDIT THESE VALUES ---
    
    # 1. Path to the ROOT folder (the one containing 'images' and 'masks')
    ROOT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025" # e.g., "/home/cvpr_ug_4/saaransh/FISH-All-Consolidated-Data/"
    
    # 2. Path to your trained model weights file
    WEIGHTS_FILE = "/home/cvpr_ug_4/saaransh/Model_A/Model 1/model_final.pth" # e.g., "/home/cvpr_ug_4/saaransh/Model_A/model_final.pth"
    
    # 3. The base config .yaml file you used for training
    CONFIG_FILE_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" # <<< !!! EDIT THIS !!!
    
    # 4. Set the confidence threshold for detection
    CONF_THRESHOLD = 0.7
    
    # ----------------------------------
    
    # Define the new image and mask root directories
    IMAGES_ROOT_DIR = os.path.join(ROOT_DIR, "images")
    MASKS_ROOT_DIR = os.path.join(ROOT_DIR, "cell_masks")
    
    print("🤖 Setting up predictor...")
    predictor = setup_predictor(CONFIG_FILE_YAML, WEIGHTS_FILE, CONF_THRESHOLD)
    
    print("🎨 Starting batch prediction and saving color masks...")
    run_batch_prediction_and_save_color_masks(predictor, IMAGES_ROOT_DIR, MASKS_ROOT_DIR)
    
    print("✨ Batch prediction complete.")