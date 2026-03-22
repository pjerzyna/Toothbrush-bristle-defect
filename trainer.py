import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from config import PATH_TRAIN_GOOD, PATH_TRAIN_DEFECTIVE, PATH_GROUNDTRUTH_MASKS, PATCH_SIZE
from utils import extract_patches, dice_loss, preprocess_image
from dataset import ToothbrushDataset
from model import UNet

class ToothbrushDefectDetector:
    def __init__(self):
        self.train_images = []
        self.train_masks = []

    def prepare_data(self):
        """Loads images, applies preprocessing, and extracts patches for training."""
        valid_ext = ('.png', '.jpg', '.jpeg')
        
        print("Loading GOOD images...")
        for filename in os.listdir(PATH_TRAIN_GOOD):
            if filename.lower().endswith(valid_ext):
                img_path = os.path.join(PATH_TRAIN_GOOD, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                img = preprocess_image(img)
                empty_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                patches_img, patches_mask = extract_patches(img, empty_mask, PATCH_SIZE)

                for p_img, p_mask in zip(patches_img, patches_mask):
                    # Sample ~5% of healthy patches to teach the model "normal" background
                    if np.random.random() < 0.05: 
                        self.train_images.append(p_img)
                        self.train_masks.append(p_mask)

        print("Loading DEFECTIVE images...")
        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):
            if filename.lower().endswith(valid_ext):
                img_path = os.path.join(PATH_TRAIN_DEFECTIVE, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                    
                img = preprocess_image(img)

                base_name = os.path.splitext(filename)[0]
                mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, f"{base_name}_mask.png")
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.zeros_like(img)

                patches_img, patches_mask = extract_patches(img, mask, PATCH_SIZE)

                for p_img, p_mask in zip(patches_img, patches_mask):
                    if np.sum(p_mask) > 0: 
                        # Keep all defective patches
                        self.train_images.append(p_img)
                        self.train_masks.append(p_mask)
                    else:
                        # Sample ~10% of healthy patches from defective toothbrushes
                        if np.random.random() < 0.1: 
                            self.train_images.append(p_img)
                            self.train_masks.append(p_mask)

        print(f"Total patches prepared for training: {len(self.train_images)}")

    def train_unet(self, epochs: int = 20, batch_size: int = 8, version: str = "1"):
        """Compiles and trains the U-Net model on the prepared patches."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")

        images = np.array(self.train_images)
        masks = np.array(self.train_masks)

        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=0.2, random_state=42
        )

        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25)
        ])

        train_dataset = ToothbrushDataset(X_train, y_train, transform)
        val_dataset = ToothbrushDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        bce = nn.BCELoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # ------------------------
            # TRAIN PHASE
            # ------------------------
            model.train()
            train_loss = 0.0

            for batch_images, batch_masks in train_loader:
                batch_images = batch_images.to(device)
                batch_masks = batch_masks.to(device)

                preds = model(batch_images)

                bce_val = bce(preds, batch_masks)
                dice_val = dice_loss(preds, batch_masks)
                loss = bce_val + dice_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ------------------------
            # VALIDATION PHASE
            # ------------------------
            model.eval()
            val_loss = 0.0
            iou_score = 0.0
            dice_score_val = 0.0
            precision_total = 0.0
            recall_total = 0.0

            with torch.no_grad():
                for batch_images, batch_masks in val_loader:
                    batch_images = batch_images.to(device)
                    batch_masks = batch_masks.to(device)

                    preds = model(batch_images)
                    loss = bce(preds, batch_masks) + dice_loss(preds, batch_masks)
                    val_loss += loss.item()

                    preds_bin = (preds > 0.5).float()
                    intersection = (preds_bin * batch_masks).sum()
                    union = preds_bin.sum() + batch_masks.sum() - intersection

                    iou_score += (intersection / (union + 1e-8)).item()

                    precision, recall = self.precision_recall(preds_bin, batch_masks)
                    precision_total += precision
                    recall_total += recall

                    dice_score_val += ((2 * intersection) / (preds_bin.sum() + batch_masks.sum() + 1e-8)).item()

            val_loss /= len(val_loader)
            iou_score /= len(val_loader)
            dice_score_val /= len(val_loader)
            precision_total /= len(val_loader)
            recall_total /= len(val_loader)

            # ------------------------
            # SAVE BEST MODEL
            # ------------------------
            saved_msg = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                filename = f"checkpoint_toothbrush_unet_v{version}.pth"
                
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, filename)
                saved_msg = f" -> Model saved!"

            # ------------------------
            # LOGGING
            # ------------------------
            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {iou_score:.4f} | "
                f"Dice: {dice_score_val:.4f} | "
                f"Prec: {precision_total:.4f} | "
                f"Rec: {recall_total:.4f}{saved_msg}"
            )

        return model

    @staticmethod
    def precision_recall(pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """Calculates pixel-level precision and recall for a batch."""
        pred = (pred > 0.5).float()
        target = target.float()

        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        return precision.item(), recall.item()