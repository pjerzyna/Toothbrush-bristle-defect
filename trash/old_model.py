import numpy as np
import cv2
import os

# [v] • prepare the data (images) and define defect classes,
# [v] • propose and implement a preprocessing stage (e.g. normalisation, denoising, contrast enhancement),
# [v] • determine the regions relevant for analysis (e.g. segmentation of bristle tips) or features describing the defects,
# • select and train a classification/detection model (or build a rule-based detector using the features),
# • evaluate the quality of the solution (e.g. accuracy, precision/recall, confusion matrix) and discuss its limitations


# 1024x1024
PATH_TRAIN_GOOD = "toothbrush/train/good/"
PATH_TRAIN_DEFECTIVE = "toothbrush/train/defective/"

# my own creative juices
def pi_spiral_kernel(size=5):
    """Create a size x size kernel filled with digits of pi in clockwise spiral order."""
    pi_digits = [
        3,1,4,1,5,
        9,2,6,5,3,
        5,8,9,7,9,
        3,2,3,8,4,
        6,2,6,4,3
    ]
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
        # NEW: list of masks corresponding to segmented regions of interest (bristle tips)
        self.train_masks = []

    # NEW: Method implementing point 3 (determining regions for analysis)
    def segment_bristles(self, preprocessed_image):
        """
        Segmentation based on edge detection.
        Bristles generate a lot of edges (texture), while the plastic handle is smooth and will be ignored.
        """
        # 1. Canny Edge Detection
        # Thresholds 40 and 120 are designed to capture the sharp edges of the bristles
        edges = cv2.Canny(preprocessed_image, 40, 120)
        
        # 2. Dilation - merges individual bristle edges into solid clusters (blobs)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bristles_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
        
        # 3. Morphological Closing - fills small black holes inside the white bristle clusters
        bristles_mask = cv2.morphologyEx(bristles_dilated, cv2.MORPH_CLOSE, kernel_dilate)

        return bristles_mask

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

                    # Determining regions for analysis (segmentation)
                    segmented_region = self.segment_bristles(image_norm)
                    self.train_masks.append(segmented_region)

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
                    
                    # NEW: Determining regions for analysis (segmentation)
                    segmented_region = self.segment_bristles(image_norm)
                    self.train_masks.append(segmented_region)

                    self.train_images.append(image_norm)
                    self.train_labels.append(classes['defective'])
        
        # Convert to NumPy arrays
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        self.train_masks = np.array(self.train_masks) # NEW: maskconversion to NumPy array
        print(f"Prepared {len(self.train_images)} images with labels and segmented regions.")
        
        # go through all images to check the preprocessing results
        for i in range(len(self.train_images)):
            #  NEW: Displaying preprocessed image next to its segmented mask 
            combined_view = np.hstack((self.train_images[i], self.train_masks[i]))
            combined_view_resized = cv2.resize(combined_view, (1024, 512)) 
            cv2.imshow("Left: Preprocessed | Right: Segmented Region", combined_view_resized)
            key = cv2.waitKey(300) & 0xFF 
            
            # NEW: If 'q' (ord('q')) OR 'Escape' (ASCII code 27) is pressed, break the loop
            if key == ord('q') or key == 27:
                print("Displaying images stopped.")
                break

        cv2.destroyAllWindows()

        # 90 - num of images, 1024x1024 - size of each image
        print(self.train_images.shape)


t1 = ToothbrushDefectDetector()
t1.prepare_data()