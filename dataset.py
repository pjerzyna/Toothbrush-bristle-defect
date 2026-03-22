import torch
from torch.utils.data import Dataset

class ToothbrushDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Add channel dimension and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.transform:
            # Stack image and mask to apply identical spatial transformations
            stacked = torch.cat([image, mask], dim=0)
            stacked = self.transform(stacked)

            image = stacked[:1]
            mask = stacked[1:]

        return image, mask