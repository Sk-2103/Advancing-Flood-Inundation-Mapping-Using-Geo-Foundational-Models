import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage

import rasterio as rio
from matplotlib import pyplot as plt
# ===== Normalize over the dataset 
def normalize_image(image):
    """Normalize a single image to [0, 1]."""
    img_std = np.std(image)
    img_mean = np.mean(image)
    img_normalized = (image - img_mean) / img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / 
                      (np.max(img_normalized) - np.min(img_normalized))) * 255
    return img_normalized

def channelwise_normalize_image(image):
    """Normalize each channel of the image independently to [0, 1]."""
    normalized_image = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[0]):  # Loop over channels
        channel = image[c]
        channel_std = np.std(channel)
        channel_mean = np.mean(channel)
        channel_normalized = (channel - channel_mean) / channel_std
        channel_normalized = ((channel_normalized - np.min(channel_normalized)) / 
                              (np.max(channel_normalized) - np.min(channel_normalized))) * 255
        normalized_image[c] = channel_normalized
    return normalized_image

class RemSemLoader(Dataset):
    """Dataset class for ISIC dataset with train, validation, and test splits."""
    def __init__(self, root_dir, train=True, test=False):
        """
        Args:
            root_dir (str): Path to the dataset folder.
            train (bool): If True, loads training data. If False, loads validation/test data.
            test (bool): If True and train=False, loads test data instead of validation.
        """
        super(RemSemLoader, self).__init__()
        self.train = train
        if train:
            data_dir = os.path.join(root_dir, 'train')
        else:
            data_dir = os.path.join(root_dir, 'test' if test else 'val')

        self.image_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'mask')

        self.image_paths = sorted(os.listdir(self.image_dir))
        self.mask_paths = sorted(os.listdir(self.mask_dir))

    def __getitem__(self, indx):
        img_path = os.path.join(self.image_dir, self.image_paths[indx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[indx])

        # Load image using rasterio
        with rio.open(img_path) as src:
            img = src.read()  # Shape: (C, H, W), where C is the number of channels 

        # Load mask using rasterio
        with rio.open(mask_path) as src:
            mask = src.read(1)  # Read the first channel of the mask

        # Normalize image and mask
        #img = normalize_image(img)
        #mask = mask / 255.0  # Normalize mask to [0, 1]

        if self.train:
            if random.random() > 0.5:
                img, mask = self.random_rot_flip(img, mask)
            if random.random() > 0.5:
                img, mask = self.random_rotate(img, mask)
            if random.random() > 0.5:
                img, mask = self.random_shift(img, mask)
            if random.random() > 0.5:
                img, mask = self.random_zoom(img, mask)
        
        
        #plt.imshow(img[0, :, :])
        #plt.show()
        #plt.imshow(mask)
        #plt.show()
        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32)  # Already in (C, H, W) format
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # HW -> CHW

        # Extract the file name (without the full path)
        file_name = img_path.split('/')[-1]

        return img, mask, file_name

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))  # Rotate along spatial dimensions
        label = np.rot90(label, k, axes=(0, 1))
        axis = np.random.randint(0, 2) + 1  # Flip along spatial dimensions (1 or 2)
        image = np.flip(image, axis=axis).copy()
        if axis == 1:  # Flip label along axis 0 (height)
            label = np.flip(label, axis=0).copy()
        elif axis == 2:  # Flip label along axis 1 (width)
            label = np.flip(label, axis=1).copy()
        #print(image.shape)
        #print(label.shape)
       
        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, axes=(1, 2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False)
        return image, label
    
    def random_shift(self, image, label):
        # Random width and height shifts
        shift_h = np.random.uniform(-0.2, 0.2) * image.shape[1]  # Shift up to ±20% height
        shift_w = np.random.uniform(-0.2, 0.2) * image.shape[2]  # Shift up to ±20% width
        image = ndimage.affine_transform(image, matrix=np.eye(3), offset=(0, shift_h, shift_w), order=0)
        label = ndimage.affine_transform(label, matrix=np.eye(2), offset=(shift_h, shift_w), order=0)
        return image, label

    def random_zoom(self, image, label):
        # Random zoom factor between 0.8 (zoom out) and 1.2 (zoom in)
        zoom_factor = np.random.uniform(0.8, 1.2)
        
        # Create a 3x3 affine matrix for the 3D input (C, H, W)
        zoom_matrix = np.array([
            [1, 0, 0],  # Channel dimension remains unchanged
            [0, zoom_factor, 0],  # Apply zoom factor to height
            [0, 0, zoom_factor]   # Apply zoom factor to width
        ])
        
        # Apply affine transformation (ndimage.affine_transform)
        image = ndimage.affine_transform(
            image,
            matrix=zoom_matrix,
            offset=(0, 0, 0),  # No translation
            order=1,  # Linear interpolation for smooth zooming
            mode='constant',  # Fill borders with a constant value
            cval=0  # Set the constant value to zero
        )
        label = ndimage.affine_transform(
            label,
            matrix=zoom_matrix[1:, 1:],  # Use 2x2 matrix for the 2D mask (H, W)
            offset=(0, 0),  # No translation
            order=0,  # Nearest-neighbor interpolation for masks
            mode='constant',  # Fill borders with a constant value
            cval=0  # Set the constant value to zero
        )
        return image, label

    
    def __len__(self):
        return len(self.image_paths)
