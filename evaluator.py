import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import PATH_TRAIN_GOOD, PATH_TRAIN_DEFECTIVE, PATH_GROUNDTRUTH_MASKS
from model import UNet
from utils import preprocess_image # Assuming preprocess was moved to utils.py

class ToothbrushModelEvaluator:
    def __init__(self, model_path: str, threshold: float = 0.25, defect_pixel_threshold: int = 500):
        """
        Initializes the evaluator with a trained U-Net model.

        Args:
            model_path (str): Path to the saved model checkpoint.
            threshold (float): Probability threshold for binarizing predictions.
            defect_pixel_threshold (int): Minimum number of defective pixels to classify an image as defective.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.threshold = threshold
        self.defect_pixel_threshold = defect_pixel_threshold

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            print(f"Model loaded successfully from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        except KeyError:
            raise KeyError("The checkpoint file must contain a 'model' key with the state_dict.")

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Generates a defect mask prediction for a given image."""
        # Add batch and channel dimensions, normalize
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(image_tensor)

        return pred.squeeze().cpu().numpy()

    def classify_image(self, prediction: np.ndarray) -> str:
        """Classifies an image as 'good' or 'defective' based on the predicted mask."""
        if np.sum(prediction > self.threshold) > self.defect_pixel_threshold:
            return "defective"
        return "good"

    def visualize(self, image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray):
        """Plots the original image, ground truth mask, and predicted mask."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.imshow(image, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.imshow(gt_mask, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.imshow(pred_mask > self.threshold, cmap="gray")

        plt.tight_layout()
        plt.show()

    def visualize_examples(self, n: int = 5):
        """Visualizes predictions for a specified number of defective images."""
        print(f"Visualizing up to {n} examples from the defective dataset...")
        valid_extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(PATH_TRAIN_DEFECTIVE) if f.lower().endswith(valid_extensions)]
        
        for i, filename in enumerate(files[:n]):
            img_path = os.path.join(PATH_TRAIN_DEFECTIVE, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not read image at {img_path}. Skipping.")
                continue
                
            image = preprocess_image(image)

            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, f"{base_name}_mask.png")
            
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.zeros_like(image)

            pred_mask = self.predict_mask(image)
            self.visualize(image, gt_mask, pred_mask)

    def evaluate_dataset(self):
        """Evaluates the model on the entire dataset and prints metrics."""
        metrics = {
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            'correct_images': 0, 'total_images': 0
        }

        print("Evaluating GOOD images...")
        self._process_directory(PATH_TRAIN_GOOD, expected_class="good", metrics=metrics)

        print("Evaluating DEFECTIVE images...")
        self._process_directory(PATH_TRAIN_DEFECTIVE, expected_class="defective", metrics=metrics)

        self._print_final_metrics(metrics)

    def _process_directory(self, dir_path: str, expected_class: str, metrics: dict):
        """Helper function to process a directory of images for evaluation."""
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping.")
            return

        valid_extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(valid_extensions)]

        for filename in files:
            img_path = os.path.join(dir_path, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                continue

            image = preprocess_image(image)
            pred_mask = self.predict_mask(image)
            pred_class = self.classify_image(pred_mask)

            if pred_class == expected_class:
                metrics['correct_images'] += 1
            metrics['total_images'] += 1

            # Pixel-level evaluation (only meaningful for defective images or to confirm clean backgrounds)
            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, f"{base_name}_mask.png")
            
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = np.zeros_like(image)

            pred_bin = pred_mask > self.threshold
            gt_bin = gt_mask > 0

            metrics['TP'] += np.logical_and(pred_bin, gt_bin).sum()
            metrics['TN'] += np.logical_and(~pred_bin, ~gt_bin).sum()
            metrics['FP'] += np.logical_and(pred_bin, ~gt_bin).sum()
            metrics['FN'] += np.logical_and(~pred_bin, gt_bin).sum()

    def _print_final_metrics(self, metrics: dict):
        """Calculates and prints the final evaluation metrics."""
        if metrics['total_images'] == 0:
            print("No images evaluated. Cannot calculate metrics.")
            return

        TP, TN, FP, FN = metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']
        
        # Add epsilon to prevent division by zero
        eps = 1e-8
        
        accuracy_pixel = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        dice = (2 * TP) / (2 * TP + FP + FN + eps)
        iou = TP / (TP + FP + FN + eps)
        accuracy_image = metrics['correct_images'] / metrics['total_images']

        print("\n===== FINAL RESULTS =====")
        print(f"Pixel Accuracy:        {accuracy_pixel:.4f}")
        print(f"IoU (Jaccard Index):   {iou:.4f}")
        print(f"Dice Coefficient:      {dice:.4f}")
        print(f"Precision:             {precision:.4f}")
        print(f"Recall:                {recall:.4f}")
        print(f"Image-level Accuracy:  {accuracy_image:.4f}")
        
        print("\nConfusion Matrix (Pixel-level)")
        print(f"True Positives (TP):   {TP}")
        print(f"True Negatives (TN):   {TN}")
        print(f"False Positives (FP):  {FP}")
        print(f"False Negatives (FN):  {FN}")