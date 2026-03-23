import os
import torch
import torch.nn as nn
import numpy as np

# Utility for image standardization before inference
from utils import preprocess_image

class UNet(nn.Module):
    """
    Standard U-Net architecture for binary image segmentation.
    Uses 3 downsampling/upsampling blocks with skip connections.
    """
    def __init__(self):
        super().__init__()

        # Encoder path
        self.enc1 = self.block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self.block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.block(256, 512)

        # Decoder path with upsampling
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.block(128, 64)

        # Output layer (1 channel for binary mask)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def block(self, in_c: int, out_c: int) -> nn.Sequential:
        """Returns a standard convolutional block (Conv -> BN -> ReLU x2)."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ==========================================
# CODABENCH SUBMISSION INTERFACE
# ==========================================

# Global variables to cache the model and device across consecutive predictions
_model = None
_device = None

def load_model():
    """
    Initializes the model architecture and loads pre-trained weights into memory.
    This function is executed only once to prevent redundant I/O operations.
    """
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = UNet().to(_device)
    
    # Ensure this matches the exact name of the weights file included in the ZIP archive
    model_filename = "checkpoint_toothbrush_unet_v1.pth"
    weights_path = os.path.join(os.path.dirname(__file__), model_filename)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=_device)
        _model.load_state_dict(checkpoint["model"])
    else:
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
    
    _model.eval()

def predict(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Input image of shape (H, W, 3) and type uint8.
        
    Returns:
        np.ndarray: Binary mask of shape (H, W) and type uint8, 
                    containing strictly 0 (background) or 255 (defect) values.
    """
    global _model, _device
    
    # Initialize the model upon the first function call
    if _model is None:
        load_model()
        
    # 1. Apply standardized preprocessing
    processed_img = preprocess_image(image)
    
    # 2. Convert to PyTorch tensor and normalize to [0, 1] range
    image_tensor = torch.tensor(processed_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(_device)
    
    # 3. Perform forward pass without gradient tracking
    with torch.no_grad():
        pred = _model(image_tensor)
        
    # Extract the predicted probability map to a 2D NumPy array
    pred_numpy = pred.squeeze().cpu().numpy()
    
    # 4. Binarize using a 0.5 threshold and scale to Codabench requirements (0 or 255)
    binary_mask = (pred_numpy > 0.5).astype(np.uint8) * 255
    
    return binary_mask
