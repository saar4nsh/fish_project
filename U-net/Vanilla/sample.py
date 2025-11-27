import os
import cv2
import numpy as np

dir_images = "/home/nitin1/segmentation/Dataset005_Bombr/imagesTr"
dir_masks = "/home/nitin1/segmentation/Dataset005_Bombr/labelsTr"

files = sorted(os.listdir(dir_masks))
# msk_size = {(1200, 1920):0, (1200, 1600):0, (913, 1461):0}
# img_size = {(1200, 1920):0, (1200, 1600):0, (913, 1461):0}
msk_size = set()
img_size = set()
# msk_size = {(512, 512):0}
# img_size = {(512, 512):0}

for file in files:
    img = cv2.imread(os.path.join(dir_images, f"{file[:-4]}_0000.png"), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(dir_masks, file), cv2.IMREAD_GRAYSCALE)

    # img_size[img.shape] += 1
    # msk_size[msk.shape] += 1
    img_size.add(img.shape)
    msk_size.add(msk.shape)

    if( img.shape != msk.shape ):
        print(f"Image shape {img.shape} does not match mask shape {msk.shape} for file {file}")
        print()

    # if(img.shape==(913, 1461)):
        # print(file)

print("Image sizes:", img_size)
print("Mask sizes:", msk_size)