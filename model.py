import numpy as np
import cv2
import os

# [v] • prepare the data (images) and define defect classes,
# • propose and implement a preprocessing stage (e.g. normalisation, denoising, contrast enhancement),
# • determine the regions relevant for analysis (e.g. segmentation of bristle tips) or features describing the defects,
# • select and train a classification/detection model (or build a rule-based detector using the features),
# • evaluate the quality of the solution (e.g. accuracy, precision/recall, confusion matrix) and discuss its limitations

# Creation, activation and installation of virtual environment:
# $ python3 -m venv venv_toothbrush
# $ source venv_toothbrush/bin/activate
# $ pip install -r requirements.txt

# 1024x1024
PATH_TRAIN_GOOD = "toothbrush/train/good/"
PATH_TRAIN_DEFECTIVE = "toothbrush/train/defective/"


class ToothbrushDefectDetector:
    def __init__(self):
        self.train_images = []
        self.train_labels = []
        self.prepare_data()

    def prepare_data(self):
        # Define classes
        classes = {'good': 0, 'defective': 1}
        
        # Load good images
        for filename in os.listdir(PATH_TRAIN_GOOD):
            if filename.endswith('.png'):
                image_path = os.path.join(PATH_TRAIN_GOOD, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Preprocessing: --- ? ---
                    self.train_images.append(image)
                    self.train_labels.append(classes['good'])
        
        # Load defective images
        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):
            if filename.endswith('.png'):
                image_path = os.path.join(PATH_TRAIN_DEFECTIVE, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # Preprocessing: --- ? ---
                    self.train_images.append(image)
                    self.train_labels.append(classes['defective'])
        
        # Convert to NumPy arrays
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        print(f"Prepared {len(self.train_images)} images with labels.")
        
        # 90 - num of images, 1024x1024 - size of each image, 3 - number of channels (RGB)
        print(self.train_images.shape)


t1 = ToothbrushDefectDetector()


        



def predict(image):
    """Simple thresholding segmentation model.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    gray = np.mean(image, axis=2)
    mask = (gray > 128).astype(np.uint8) * 255
    return mask
