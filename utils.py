import numpy as np
import torch
import cv2

def extract_patches(img: np.ndarray, mask: np.ndarray, size: int = 256) -> tuple:
    """
    Extracts non-overlapping square patches from an image and its corresponding mask.
    Discards any edge patches that do not match the exact specified size.
    """
    patches_img = []
    patches_mask = []

    h, w = img.shape[:2]

    for i in range(0, h, size):
        for j in range(0, w, size):
            patch_img = img[i:i+size, j:j+size]
            patch_mask = mask[i:i+size, j:j+size]

            # Append only if the patch perfectly matches the requested dimensions
            if patch_img.shape == (size, size):
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)

    return patches_img, patches_mask


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Dice loss for binary segmentation.
    Uses a smoothing factor to prevent division by zero.
    """
    smooth = 1.0

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1.0 - ((2.0 * intersection + smooth) / 
                  (pred.sum() + target.sum() + smooth))


def pi_spiral_kernel(size: int = 5) -> np.ndarray:
    """
    Creates a custom morphological kernel based on the digits of Pi.
    """
    pi_digits = [
        3, 1, 4, 1, 5,
        9, 2, 6, 5, 3,
        5, 8, 9, 7, 9,
        3, 2, 3, 8, 4,
        6, 2, 6, 4, 3
    ]

    kernel = np.array(pi_digits).reshape(size, size)
    
    return kernel.astype(np.uint8)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Standardizes input images before patch extraction or model inference.
    Converts to grayscale, applies gentle smoothing, and normalizes intensities.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    return image