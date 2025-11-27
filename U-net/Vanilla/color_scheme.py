import numpy as np
from PIL import Image

class ColorMask:
    def __init__(self, ):

        self.color_dict = {
            0: (0, 0, 0),        # Background - black
            1: (255, 0, 0),      # Class 1 - red
            2: (0, 255, 0),      # Class 2 - green
            3: (0, 0, 255),      # Class 3 - blue
            4: (255, 255, 0),    # Class 4 - yellow
        }

    def __call__(self, mask_tensor):
        
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)  # [H, W]
        h, w = mask_np.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in self.color_dict.items():
            color_img[mask_np == class_idx] = color

        return Image.fromarray(color_img)