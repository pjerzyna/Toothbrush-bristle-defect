import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import argparse
import os


PATH_TRAIN_GOOD = "toothbrush/train/good/"
PATH_TRAIN_DEFECTIVE = "toothbrush/train/defective/"
PATH_GROUNDTRUTH_MASKS = "toothbrush/ground_truth/defective/"

PATCH_SIZE = 256


############################################
# PATCH EXTRACTION
############################################

def extract_patches(img, mask, size=256):

    patches_img = []
    patches_mask = []

    h, w = img.shape

    for i in range(0, h, size):
        for j in range(0, w, size):

            patch_img = img[i:i+size, j:j+size]
            patch_mask = mask[i:i+size, j:j+size]

            if patch_img.shape == (size, size):

                patches_img.append(patch_img)
                patches_mask.append(patch_mask)

    return patches_img, patches_mask


############################################
# DICE LOSS
############################################

def dice_loss(pred, target):

    smooth = 1.

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1 - ((2. * intersection + smooth) /
               (pred.sum() + target.sum() + smooth))


############################################
# PI KERNEL
############################################

def pi_spiral_kernel(size=5):

    pi_digits = [
        3,1,4,1,5,
        9,2,6,5,3,
        5,8,9,7,9,
        3,2,3,8,4,
        6,2,6,4,3
    ]

    kernel = np.array(pi_digits).reshape(size, size)

    return kernel.astype(np.uint8)


############################################
# DATASET CLASS
############################################

class ToothbrushDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks, transform=None):

        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]

        image = torch.tensor(image).unsqueeze(0).float()/255
        mask = torch.tensor(mask).unsqueeze(0).float()/255

        if self.transform:

            stacked = torch.cat([image, mask])
            stacked = self.transform(stacked)

            image = stacked[:1]
            mask = stacked[1:]

        return image, mask


############################################
# MAIN CLASS
############################################

class ToothbrushDefectDetector:

    def __init__(self):

        self.images_good = []
        self.images_defective = []

        self.masks_good = []
        self.masks_defective = []

        self.train_images = []
        self.train_masks = []


    ############################################
    # PREPROCESSING
    ############################################

    def preprocess(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.GaussianBlur(image, (3,3), 0)

        kernel = pi_spiral_kernel(5)

        # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image


    ############################################
    # PREPARE DATA
    ############################################

    def prepare_data(self):

        print("Loading GOOD images...")
        for filename in os.listdir(PATH_TRAIN_GOOD):
                    if filename.endswith(".png"):
                        img = cv2.imread(os.path.join(PATH_TRAIN_GOOD, filename))
                        img = self.preprocess(img)
                        
                        # Tworzymy pustą maskę (same zera)
                        empty_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                        patches_img, patches_mask = extract_patches(img, empty_mask, PATCH_SIZE)

                        for p_img, p_mask in zip(patches_img, patches_mask):
                            # Z dobrych szczoteczek wybieramy tylko małą część patchy (np. 5%)
                            # To wystarczy, by model nauczył się, jak wygląda "norma"
                            if np.random.random() < 0.05: 
                                self.train_images.append(p_img)
                                self.train_masks.append(p_mask)



        print("Loading DEFECTIVE images...")
        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):
            if filename.endswith(".png"):
                # 1. NAJPIERW WCZYTUJEMY
                img = cv2.imread(os.path.join(PATH_TRAIN_DEFECTIVE, filename))
                img = self.preprocess(img)

                base_name = os.path.splitext(filename)[0]
                mask_filename = f"{base_name}_mask.png"
                mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, mask_filename)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    mask = np.zeros_like(img)

                # 2. POTEM WYCINAMY PATCHE
                patches_img, patches_mask = extract_patches(img, mask, PATCH_SIZE)

                # 3. NA KOŃCU FILTRUJEMY I DODAJEMY
                for p_img, p_mask in zip(patches_img, patches_mask):
                    # Jeśli w masce jest jakakolwiek wada (białe piksele)
                    if np.sum(p_mask) > 0: 
                        self.train_images.append(p_img)
                        self.train_masks.append(p_mask)
                    else:
                        # Zostawiamy tylko 5-10% "zdrowych" fragmentów z uszkodzonych szczoteczek
                        if np.random.random() < 0.1: 
                            self.train_images.append(p_img)
                            self.train_masks.append(p_mask)

        print("Total patches:", len(self.train_images))


    ############################################
    # TRAIN MODEL
    ############################################

    def train_unet(self, epochs=20, batch_size=8, version="1"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = np.array(self.train_images)
        masks = np.array(self.train_masks)

        X_train, X_val, y_train, y_val = train_test_split(
            images,
            masks,
            test_size=0.2,
            random_state=42
        )

        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25)
        ])

        train_dataset = ToothbrushDataset(X_train, y_train, transform)
        val_dataset = ToothbrushDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size
        )

        model = UNet().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        bce = nn.BCELoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):

            ########################
            # TRAIN
            ########################

            model.train()

            train_loss = 0

            for images, masks in train_loader:

                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)

                bce_val = bce(preds, masks)
                dice_val = dice_loss(preds, masks)

                loss = bce_val + dice_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)


            ########################
            # VALIDATION
            ########################

            model.eval()

            val_loss = 0
            iou_score = 0
            dice_score_val = 0
            precision_total = 0
            recall_total = 0

            with torch.no_grad():

                for images, masks in val_loader:

                    images = images.to(device)
                    masks = masks.to(device)

                    preds = model(images)

                    loss = bce(preds, masks) + dice_loss(preds, masks)

                    val_loss += loss.item()

                    preds_bin = (preds > 0.5).float()

                    intersection = (preds_bin * masks).sum()

                    union = preds_bin.sum() + masks.sum() - intersection

                    iou_score += (intersection / (union + 1e-8)).item()

                    precision, recall = self.precision_recall(preds_bin, masks)

                    precision_total += precision
                    recall_total += recall

                    dice_score_val += (
                        (2 * intersection)
                        / (preds_bin.sum() + masks.sum() + 1e-8)
                    ).item()


            val_loss /= len(val_loader)
            iou_score /= len(val_loader)
            dice_score_val /= len(val_loader)
            precision_total /= len(val_loader)
            recall_total /= len(val_loader)


            ########################
            # SAVE BEST MODEL
            ########################

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                filename = f"checkpoint_toothbrush_unet_v{version}.pth"

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                },  filename)

                print(f"Best model saved as {filename}")


            ########################
            # LOGGING
            ########################

            print(
                f"Epoch {epoch+1}/{epochs}"
                f" | Train Loss: {train_loss:.4f}"
                f" | Val Loss: {val_loss:.4f}"
                f" | IoU: {iou_score:.4f}"
                f" | Dice: {dice_score_val:.4f}"
                f" | Precision: {precision_total:.4f}"
                f" | Recall: {recall_total:.4f}"
            )



        return model
    
    #############################
    # PRECISION & RECALL                                        
    #############################
    def precision_recall(self, pred, target):

        pred = (pred > 0.5).float()
        target = target.float()

        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        return precision.item(), recall.item()


############################################
# UNET
############################################

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = self.block(1,64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.block(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.block(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = self.block(256,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3 = self.block(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = self.block(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = self.block(128,64)

        self.out = nn.Conv2d(64,1,1)


    def block(self, in_c, out_c):
        return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), 
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), 
                nn.ReLU()
            )


    def forward(self,x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b),e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3),e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2),e1],1))

        return torch.sigmoid(self.out(d1))


#############################################
# MODEL EVALUATOR
#############################################

class ToothbrushModelEvaluator:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

        print("Model loaded:", model_path)


    ############################################
    # PREPROCESS (same as training)
    ############################################

    def preprocess(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.GaussianBlur(image, (3,3), 0)

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        return image


    ############################################
    # PREDICT MASK
    ############################################

    def predict_mask(self, image):

        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()/255

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            pred = self.model(image_tensor)

        return pred.squeeze().cpu().numpy()


    ############################################
    # IMAGE CLASSIFICATION (GOOD / DEFECTIVE)
    ############################################

    def classify_image(self, prediction):

        threshold_pixels = 500

        if np.sum(prediction > 0.5) > threshold_pixels:

            return "defective"

        return "good"


    ############################################
    # VISUALIZATION
    ############################################

    def visualize(self, image, gt_mask, pred_mask):
        plt.figure(figsize=(10,4))

        plt.subplot(1,3,1)
        plt.title("Image")
        plt.axis("off")
        plt.imshow(image, cmap="gray")

        plt.subplot(1,3,2)
        plt.title("Ground truth")
        plt.axis("off")
        plt.imshow(gt_mask, cmap="gray")

        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.axis("off")
        plt.imshow(pred_mask > 0.5, cmap="gray")

        plt.show()


    ############################################
    # VISUALIZATION OF EXAMPLES
    ############################################
    def visualize_examples(self, n=5):

        shown = 0

        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):
            if shown >= n:
                break

            image = cv2.imread(os.path.join(PATH_TRAIN_DEFECTIVE, filename))
            image = self.preprocess(image)
            base_name = os.path.splitext(filename)[0]
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, mask_filename)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = self.predict_mask(image)
            self.visualize(image, gt_mask, pred_mask)
            shown += 1


    ############################################
    # EVALUATE WHOLE DATABASE
    ############################################

    def evaluate_dataset(self):

        total_TP = 0
        total_TN = 0
        total_FP = 0
        total_FN = 0

        correct_classification = 0
        total_images = 0

        print("Evaluating GOOD images...")

        for filename in os.listdir(PATH_TRAIN_GOOD):

            if filename.endswith(".png"):

                image = cv2.imread(os.path.join(PATH_TRAIN_GOOD, filename))

                image = self.preprocess(image)

                pred_mask = self.predict_mask(image)

                prediction = self.classify_image(pred_mask)

                if prediction == "good":

                    correct_classification += 1

                total_images += 1


        print("Evaluating DEFECTIVE images...")

        for filename in os.listdir(PATH_TRAIN_DEFECTIVE):

            if filename.endswith(".png"):

                image = cv2.imread(os.path.join(PATH_TRAIN_DEFECTIVE, filename))

                image = self.preprocess(image)

                base_name = os.path.splitext(filename)[0]

                mask_filename = f"{base_name}_mask.png"

                mask_path = os.path.join(PATH_GROUNDTRUTH_MASKS, mask_filename)

                if os.path.exists(mask_path):

                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                else:

                    gt_mask = np.zeros_like(image)


                pred_mask = self.predict_mask(image)

                pred_bin = pred_mask > 0.5

                gt_bin = gt_mask > 0


                TP = np.logical_and(pred_bin, gt_bin).sum()

                TN = np.logical_and(~pred_bin, ~gt_bin).sum()

                FP = np.logical_and(pred_bin, ~gt_bin).sum()

                FN = np.logical_and(~pred_bin, gt_bin).sum()


                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN


                prediction = self.classify_image(pred_mask)

                if prediction == "defective":

                    correct_classification += 1

                total_images += 1


        ############################################
        # FINAL METRICS
        ############################################

        accuracy_pixel = (
            total_TP + total_TN
        ) / (
            total_TP + total_TN + total_FP + total_FN
        )

        precision = total_TP / (total_TP + total_FP + 1e-8)

        recall = total_TP / (total_TP + total_FN + 1e-8)

        dice = (2 * total_TP) / (
            2 * total_TP + total_FP + total_FN + 1e-8
        )

        iou = total_TP / (
            total_TP + total_FP + total_FN + 1e-8
        )

        accuracy_image = correct_classification / total_images


        ############################################
        # PRINT RESULTS
        ############################################

        print("\n===== FINAL RESULTS =====")

        print("Pixel Accuracy:", accuracy_pixel)

        print("IoU:", iou)

        print("Dice:", dice)

        print("Precision:", precision)

        print("Recall:", recall)

        print("Image-level Accuracy:", accuracy_image)

        print("\nConfusion matrix (pixel-level)")

        print("TP:", total_TP)

        print("TN:", total_TN)

        print("FP:", total_FP)

        print("FN:", total_FN)



############################################
# RUN
############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trenowanie modelu UNet dla szczoteczek")
    parser.add_argument("-v", "--version", type=str, default="1")
    parser.add_argument("-e", "--epochs", type=int, default=20) # Nowa flaga

    args = parser.parse_args()

    # detector = ToothbrushDefectDetector()
    # detector.prepare_data()

    # model = detector.train_unet(epochs=args.epochs, version=args.version)

    evaluator = ToothbrushModelEvaluator(
        "checkpoint_toothbrush_unet_v2.pth"
    )

    evaluator.evaluate_dataset()
    evaluator.visualize_examples()