import cv2
from helper import *
import os
def load_images(image_dir, mask_dir):
    images, masks = [], []

    for i in range(len(os.listdir(image_dir))):
        image = cv2.imread(image_dir + f'image_{i}.jpg')
        image = cv2.resize(image, (384,216))
        images.append(image)
    for i in range(len(os.listdir(mask_dir))):
        mask = cv2.imread(mask_dir + f'image_{i}_mask.png')
        binary_mask = np.where(mask > 0, 255, mask)
        binary_mask = cv2.resize(binary_mask, (384,216))
        masks.append(binary_mask)
    return images, masks

images, masks = load_images("data_ours/images/", "data_ours/masks/")
images_augmented, masks_augmented = augmentData(images, masks)

for image, mask in zip(images_augmented, masks_augmented):
    cv2.imshow('image', image)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)