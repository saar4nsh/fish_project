import os

def scale_bounding_boxes(base_dir='datasets_merged', scale_factor=3.0):
    """
    Scans the training labels in a dataset directory and scales the
    bounding box width and height by a given factor.

    This function directly overwrites the existing label files.

    Args:
        base_dir (str): The root directory of the dataset (e.g., 'datasets_merged').
        scale_factor (float): The factor by which to multiply the width and height.
    """
    labels_path = os.path.join(base_dir, 'train', 'labels')

    # --- 1. Safety Check ---
    if not os.path.isdir(labels_path):
        print(f"❌ Error: Directory not found at '{labels_path}'.")
        print("Please make sure you have run the previous script to create the merged directory.")
        return

    print(f"🔍 Scanning for label files in '{labels_path}'...")
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    if not label_files:
        print("No label files found to process.")
        return

    processed_count = 0
    # --- 2. Iterate Through Each Label File ---
    for filename in label_files:
        file_path = os.path.join(labels_path, filename)
        modified_lines = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    modified_lines.append(line) # Keep malformed lines as is
                    continue

                # --- 3. Extract and Convert YOLO Data ---
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # --- 4. Scale Width and Height ---
                new_width = width * scale_factor
                new_height = height * scale_factor

                # --- 5. Crucial: Clamp values to prevent boxes from going out of bounds ---
                # A box's edge is center ± half-of-dimension.
                # We calculate the max possible width/height from the center to any edge.
                max_w_from_center = min(x_center, 1.0 - x_center) * 2.0
                max_h_from_center = min(y_center, 1.0 - y_center) * 2.0
                
                # The new dimension cannot exceed the image boundary
                final_width = min(new_width, max_w_from_center)
                final_height = min(new_height, max_h_from_center)

                # Format the new line with 6 decimal places, a common practice
                modified_line = f"{class_id} {x_center:.6f} {y_center:.6f} {final_width:.6f} {final_height:.6f}\n"
                modified_lines.append(modified_line)

            # --- 6. Overwrite the Original File with Modified Data ---
            with open(file_path, 'w') as f:
                f.writelines(modified_lines)
            
            processed_count += 1

        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")

    print(f"\n✨ Success! Processed and updated {processed_count} label files.")

# --- How to Run This Code ---
if __name__ == "__main__":
    # Run this script in the same directory where your 'datasets_merged' folder is located.
    # It will modify the files IN-PLACE.
    scale_bounding_boxes(base_dir='datasets', scale_factor=3.0)