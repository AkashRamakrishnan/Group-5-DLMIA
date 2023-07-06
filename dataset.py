import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_file_path, mask_file_path):
        self.images = np.load(image_file_path)
        self.masks = np.load(mask_file_path)

    def __getitem__(self, index):
        # Get an individual image and mask pair from the dataset
        image = self.images[index]
        mask = self.masks[index]

        # Convert the image and mask to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Return the processed image and mask
        return image, mask

    def __len__(self):
        return len(self.images)


