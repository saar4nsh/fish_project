from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # UPDATED: Normalization is now for 1-channel grayscale
        self.general_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5), std=(0.5)), # Changed from (0.5, 0.5, 0.5)
            ToTensorV2()
        ])

        # List files from the mask directory as the source of truth
        self.image_names = sorted([img for img in os.listdir(mask_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        mask_name = self.image_names[idx]
        
        # Assumes image name is mask_name + '_0000' suffix
        # e.g., mask 'crack_001.png' -> image 'crack_001_0000.png'
        img_name = mask_name.split('.')[0] + '_0000.' + mask_name.split('.')[1]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            # UPDATED: Load image as grayscale ("L") instead of "RGB"
            image = np.array(Image.open(img_path).convert("L"))
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}")
            # Return dummy data or raise error
            # Note: Dummy image is now 1-channel
            return torch.zeros(1, 256, 256), torch.zeros(256, 256).long(), "missing_file"

        # This line is correct. 'L' mode will load your 0, 1, 2 pixel values.
        mask = np.array(Image.open(mask_path).convert("L"))  # Convert mask to grayscale (L mode)

        # Apply augmentations
        augmented = self.general_transform(image=image, mask=mask)
        image = augmented['image']      # Now [1, H, W], torch.float32
        mask = augmented['mask'].long() # [H, W], torch.int64 (long)

        # print(image.shape, mask.shape, mask_name)

        return image, mask, mask_name


class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
        # UPDATED: Normalization is now for 1-channel grayscale
        self.general_transform = A.Compose([
            A.Normalize(mean=(0.5), std=(0.5)), # Changed from (0.5, 0.5, 0.5)
            ToTensorV2()
        ])

        self.image_names = sorted([img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # UPDATED: Load image as grayscale ("L") instead of "RGB"
        image = np.array(Image.open(img_path).convert("L"))
        
        augmented = self.general_transform(image=image)
        image = augmented['image']          # Now [1, H, W]
        
        return image, img_name

