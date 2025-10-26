import torch
import cv2
import os
import numpy as np
from pathlib import Path

# --- Configuration ---
WEIGHTS_FILE = "/home/cvpr_ug_4/saaransh/Model_B/yolov5/runs/train/exp5/weights/best.pt" # Path to your .pt file
SOURCE_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025/cell_patches" # Input directory with patches
# **NEW**: Define separate output directories
VIS_OUTPUT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025/yolo_visualizations" # Base output dir for images
LABELS_OUTPUT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025/yolo_labels" # Base output dir for txt files
CONF_THRESHOLD = 0.40 # Confidence threshold (adjust as needed)
IOU_THRESHOLD = 0.45 # Non-Max Suppression threshold (adjust as needed)
GPU_DEVICE = 1 # Specify the GPU index (0, 1, etc.) or 'cpu'

# --- Visualization Settings ---
BBOX_COLOR = (0, 255, 0) # Green color for bounding box (BGR)
TEXT_COLOR = (255, 255, 255) # White color for text
FONT_SCALE = 0.25
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BBOX_THICKNESS = 1 # Thinner bounding box
TEXT_THICKNESS = 1

# --- Helper Functions ---

def find_png_files(root_dir):
    """ Finds all .png files recursively. """
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                full_path = os.path.join(dirpath, fname)
                image_files.append(full_path)
    print(f"Found {len(image_files)} '.png' patch files.")
    return image_files

def xyxy_to_yolo(xyxy, img_width, img_height):
    """ Converts xyxy bbox format to YOLO format (normalized xywh). """
    x_min, y_min, x_max, y_max = xyxy
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# --- Main Inference Function ---

def run_inference_and_visualize():
    """ Loads model, runs inference, saves visualizations and YOLO txt files separately. """
    # --- 1. Load Model ---
    print("Loading YOLOv5 model...")
    try:
        device_str = f'cuda:{GPU_DEVICE}' if torch.cuda.is_available() and isinstance(GPU_DEVICE, int) else 'cpu'
        # Define the path to your local YOLOv5 repo directory
        yolov5_repo_path = '/home/cvpr_ug_4/saaransh/Model_B/yolov5' # Adjust if needed

        # Load the model locally
        model = torch.hub.load(yolov5_repo_path, 'custom', path=WEIGHTS_FILE, source='local', device=device_str)
        model.conf = CONF_THRESHOLD
        model.iou = IOU_THRESHOLD
        print(f"Model loaded successfully on device: {device_str}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Find Images ---
    image_paths = find_png_files(SOURCE_DIR)
    if not image_paths:
        print("No PNG images found in the source directory.")
        return

    # Ensure base output directories exist
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABELS_OUTPUT_DIR, exist_ok=True)

    # --- 3. Process Each Image ---
    for img_path in image_paths:
        print(f"\n--- Processing: {img_path} ---")

        try:
            # --- 3a. Read Image ---
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print("  Error: Could not read image. Skipping.")
                continue
            img_height, img_width = img_bgr.shape[:2]

            # --- 3b. Run Inference ---
            results = model(img_bgr[:, :, ::-1]) # Model expects RGB, OpenCV reads BGR

            # --- 3c. Process Results ---
            predictions = results.pandas().xyxy[0]
            yolo_txt_lines = []
            img_visualized = img_bgr.copy()

            if predictions.empty:
                print("  No detections found.")
            else:
                print(f"  Found {len(predictions)} detections.")
                for _, row in predictions.iterrows():
                    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    conf = row['confidence']
                    class_idx = int(row['class'])
                    class_name = row['name']

                    # Draw Bounding Box
                    cv2.rectangle(img_visualized, (x_min, y_min), (x_max, y_max), BBOX_COLOR, BBOX_THICKNESS)

                    # Prepare and Draw Label Text
                    label = f"{class_name} {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, TEXT_THICKNESS)
                    cv2.rectangle(img_visualized, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), BBOX_COLOR, -1)
                    cv2.putText(img_visualized, label, (x_min, y_min - baseline), FONT_FACE, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

                    # Convert to YOLO format
                    x_center_norm, y_center_norm, width_norm, height_norm = xyxy_to_yolo((x_min, y_min, x_max, y_max), img_width, img_height)
                    yolo_line = f"{class_idx} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                    yolo_txt_lines.append(yolo_line)

            # --- 3d. **Determine SEPARATE Output Paths** ---
            relative_path = os.path.relpath(img_path, SOURCE_DIR)
            relative_subdir = os.path.dirname(relative_path)

            # Visualization output path
            vis_output_subdir = os.path.join(VIS_OUTPUT_DIR, relative_subdir)
            os.makedirs(vis_output_subdir, exist_ok=True)
            vis_base_filename = Path(img_path).stem # Filename without extension
            vis_output_path = os.path.join(vis_output_subdir, f"{vis_base_filename}_pred.png")

            # Labels output path
            labels_output_subdir = os.path.join(LABELS_OUTPUT_DIR, relative_subdir)
            os.makedirs(labels_output_subdir, exist_ok=True)
            labels_base_filename = Path(img_path).stem # Filename without extension
            txt_output_path = os.path.join(labels_output_subdir, f"{labels_base_filename}.txt")

            # --- 3e. Save Visualization ---
            cv2.imwrite(vis_output_path, img_visualized)
            print(f"  Saved visualization to: {vis_output_path}")

            # --- 3f. Save YOLO TXT File ---
            if yolo_txt_lines:
                with open(txt_output_path, 'w') as f:
                    f.write("\n".join(yolo_txt_lines))
                print(f"  Saved YOLO labels to: {txt_output_path}")
            elif os.path.exists(txt_output_path):
                 os.remove(txt_output_path) # Ensure no old txt file remains if no detections


        except Exception as e:
            print(f"  Error processing image {img_path}: {e}")

# --- Run the script ---
if __name__ == "__main__":
    run_inference_and_visualize()
    print("\n✅ Inference and visualization complete.")