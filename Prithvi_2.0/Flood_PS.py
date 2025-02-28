import os
import numpy as np
import cv2
import torch
import rasterio
import torch.optim as optim
import h5py
import terratorch
from terratorch.datamodules import Landslide4SenseNonGeoDataModule
from terratorch.tasks import SemanticSegmentationTask
import albumentations
from albumentations import Compose, Flip
from albumentations import Compose, Resize, HorizontalFlip
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
# Download config.json to the specified folder
hf_hub_download(
    repo_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
    filename="config.json",
    cache_dir='/home/skaushik/flood_prithvi/'
)


# Custom FloodDataModule
class FloodDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=2, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.data_root = data_root
        self.train_dir = os.path.join(data_root, "images/train")
        self.val_dir = os.path.join(data_root, "images/validation")
        self.test_dir = os.path.join(data_root, "images/test")
        self.mask_train_dir = os.path.join(data_root, "annotations/train")
        self.mask_val_dir = os.path.join(data_root, "annotations/validation")
        self.mask_test_dir = os.path.join(data_root, "annotations/test")
        self.batch_size = batch_size

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = self.create_dataset(self.train_dir, self.mask_train_dir, self.train_transform)
            self.val_dataset = self.create_dataset(self.val_dir, self.mask_val_dir, self.val_transform)

        if stage in ("test", None):
            self.test_dataset = self.create_dataset(self.test_dir, self.mask_test_dir, self.test_transform)

    def create_dataset(self, image_dir, mask_dir, transform):
        return CustomFloodDataset(image_dir, mask_dir, transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True
        )


# Custom Dataset
class CustomFloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Read the image and mask using rasterio
        with rasterio.open(image_path) as src:
            image = src.read()  # (bands, height, width)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Single channel for mask

        mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
        image = np.moveaxis(image, 0, -1)  # Convert to (height, width, bands)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Include the filename in the output
        filename = os.path.basename(image_path)

        return {"image": image, "mask": mask, "filename": filename}


#    def plot(self, sample):
#        """Plot a sample from the dataset."""
#        image = sample["image"].permute(1, 2, 0).numpy()  # Convert to HWC
#        mask = sample["mask"].numpy()
#
#        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#        ax[0].imshow(image)
#        ax[0].set_title("Image")
#        ax[0].axis("off")
#
#        ax[1].imshow(mask, cmap="gray")
#        ax[1].set_title("Mask")
#        ax[1].axis("off")
#
#        plt.tight_layout()
#        plt.show()
#
#        return fig


# Transforms
train_transform = Compose([
    HorizontalFlip(),
    Resize(896, 896),
    ToTensorV2(),
])

val_transform = Compose([
    Resize(896, 896),
    ToTensorV2(),
])

test_transform = Compose([
    Resize(896, 896),
    ToTensorV2(),
])

# Instantiate FloodDataModule
data_module = FloodDataModule(
    data_root='data/Ghana',
    batch_size=2,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

# Logger
logger = TensorBoardLogger(
    save_dir="flood_logs",
    name="ghana"
)

# Define ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/ghana/",  # Save all best models in a single directory
    filename="epoch-{epoch:02d}-val_f1-{val/Multiclass_F1_Score:.4f}",  # Include epoch and F1 in filename
    monitor="val/Multiclass_F1_Score",  # Track validation F1-score
    mode="max",  # Save models with highest F1-score
    save_top_k=5,  # Keep only the best 3 models
    every_n_epochs=1,
    save_on_train_epoch_end=False,  # Save based on validation performance
    auto_insert_metric_name=False  # Prevents creating subfolders for each epoch
)



# Trainer
trainer = pl.Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[checkpoint_callback]
)

# Model
model = SemanticSegmentationTask(
    model_args={
        "decoder": "UperNetDecoder",
        "backbone_pretrained": True,
        "backbone": "prithvi_eo_v2_600", # can choose multiple Prithvi model (300MM, 300M, 600M)
        "backbone_in_channels": 4,  # Match your dataset's number of channels
        "rescale": True,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_BROAD"],
        "backbone_num_frames": 1,
        "num_classes": 2,
        "head_dropout": 0.1,
        "decoder_channels": 256,
        "decoder_scale_modules": True,
        "head_channel_list": [128, 64],
        "necks": [
            {
                "name": "SelectIndices",
                "indices": [7, 15, 23, 31]
            },
            {
                "name": "ReshapeTokensToImage"
            }
        ]
    },
    plot_on_val=False,  # Enable plotting during validation (set False to skip)
    loss="focal",
    lr=1e-4,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.1},
    scheduler="StepLR",
    scheduler_hparams={"step_size": 10, "gamma": 0.9},
    model_factory="EncoderDecoderFactory",
)

# Train
trainer.fit(model, datamodule=data_module)
