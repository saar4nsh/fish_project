import os
import shutil

# --- PLEASE EDIT THIS VALUE ---
# Set this to your main project folder
# e.g., "/home/cvpr_ug_4/saaransh/FISH-All-Consolidated-Data/"
ROOT_DIR = "/home/cvpr_ug_4/saaransh/fish_AI_images/13.09.2025"
# ------------------------------

# Define the new directory paths
images_dir = os.path.join(ROOT_DIR, "images")
masks_dir = os.path.join(ROOT_DIR, "masks")

# Create the new 'images' and 'masks' directories
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

print(f"Created directories: \n- {images_dir}\n- {masks_dir}\n")
print("Moving subfolders to 'images' directory...")

# List all items in the root directory
for item_name in os.listdir(ROOT_DIR):
    src_path = os.path.join(ROOT_DIR, item_name)
    
    # Check if the item is a directory AND is not the new 'images' or 'masks' folder
    if os.path.isdir(src_path) and item_name not in ["images", "masks"]:
        # Define the destination path inside the 'images' folder
        dst_path = os.path.join(images_dir, item_name)
        
        try:
            # Move the entire directory
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path}  ➡️  {dst_path}")
        except Exception as e:
            print(f"Error moving {src_path}: {e}")

print("\n📂 Directory restructuring complete!")