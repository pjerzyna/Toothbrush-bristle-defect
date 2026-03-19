import numpy as np
import cv2
import os

# [v] • prepare the data (images) and define defect classes,
# [v] • propose and implement a preprocessing stage (e.g. normalisation, denoising, contrast enhancement),
# • determine the regions relevant for analysis (e.g. segmentation of bristle tips) or features describing the defects,
# • select and train a classification/detection model (or build a rule-based detector using the features),
# • evaluate the quality of the solution (e.g. accuracy, precision/recall, confusion matrix) and discuss its limitations


# 1024x1024
PATH_TRAIN_GOOD = "toothbrush/train/good/"
PATH_TRAIN_DEFECTIVE = "toothbrush/train/defective/"

# my own creative juices
def pi_spiral_kernel(size=5):
    """Create a size x size kernel filled with digits of pi in clockwise spiral order."""
    pi_digits = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3]
    if len(pi_digits) < size * size:
        raise ValueError("Not enough pi digits for requested kernel size")

    kernel = np.zeros((size, size), dtype=np.uint8)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    dir_idx = 0
    row, col = 0, 0
    visited = set()

    for i in range(size * size):
        kernel[row, col] = pi_digits[i]
        visited.add((row, col))

        next_row = row + directions[dir_idx][0]
        next_col = col + directions[dir_idx][1]

        if not (0 <= next_row < size and 0 <= next_col < size) or (next_row, next_col) in visited:
            dir_idx = (dir_idx + 1) % 4
            next_row = row + directions[dir_idx][0]
            next_col = col + directions[dir_idx][1]

        row, col = next_row, next_col

    return kernel


class ToothbrushDefectDetector:
    def __init__(self):
        self.train_images = []
        self.train_labels = []

    def prepare_data(self):
        # Define classes
        classes = {'good': 0, 'defective': 1}
        
        # Load good images
        for filename in os.listdir(PATH_TRAIN_GOOD):
            if filename.endswith('.png'):
                image_path = os.path.join(PATH_TRAIN_GOOD, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Preprocessing: grayscale transition, denoising, contrast enhancement and normalisation
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                     # Convert to grayscale
                    
                    # Denoising: Gaussian blur to reduce noise
                    image_denoised = cv2.GaussianBlur(image, (3, 3), 0)
                    
                    # Morphological opening with pi-spiral kernel
                    kernel = pi_spiral_kernel(5)
                    image_opened = cv2.morphologyEx(image_denoised, cv2.MORPH_OPEN, kernel)
                    
                    # Normalisation: scale to 0-255 range
                    image_norm = cv2.normalize(image_opened, None, 0, 255, cv2.NORM_MINMAX)

                    self.train_images.append(image_norm)
                    self.train_labels.append(classes['good'])
        
        # Load defective images
        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):
            if filename.endswith('.png'):
                image_path = os.path.join(PATH_TRAIN_DEFECTIVE, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Preprocessing: grayscale transition, denoising, contrast enhancement and normalisation
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                      # Convert to grayscale
                    
                    # Denoising: Gaussian blur to reduce noise
                    image_denoised = cv2.GaussianBlur(image, (3, 3), 0)
                    
                    # Morphological opening with pi-spiral kernel (removes small objects)
                    kernel = pi_spiral_kernel(5)
                    image_opened = cv2.morphologyEx(image_denoised, cv2.MORPH_OPEN, kernel)
                    
                    # Normalisation: scale to 0-255 range
                    image_norm = cv2.normalize(image_opened, None, 0, 255, cv2.NORM_MINMAX)
                    
                    self.train_images.append(image_norm)
                    self.train_labels.append(classes['defective'])
        
        # Convert to NumPy arrays
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        print(f"Prepared {len(self.train_images)} images with labels.")
        
        # go through all images to check the preprocessing results
        for i in range(len(self.train_images)):
            cv2.imshow("Preprocessed Image", self.train_images[i])
            cv2.waitKey(300)

        # 90 - num of images, 1024x1024 - size of each image
        print(self.train_images.shape)


t1 = ToothbrushDefectDetector()
t1.prepare_data()
