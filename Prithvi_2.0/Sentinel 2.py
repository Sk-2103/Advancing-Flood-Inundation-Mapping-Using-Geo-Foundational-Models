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



# Custom Dataset
class CustomFloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Mismatch between image and mask count!"

        self.selected_band_indices = [0, 1, 2, 6, 8, 9]  # B2, B3, B4, B8, B11, B12 [0, 1, 2, 6, 8, 9]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        with rasterio.open(image_path) as src:
            image = src.read()
            image = image[self.selected_band_indices, :, :]

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = (mask > 0.5).astype(np.uint8)
        image = np.moveaxis(image, 0, -1)  # Convert to (H, W, C)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        filename = os.path.basename(image_path)
        return {"image": image, "mask": mask, "filename": filename}


# Data Module
class FloodDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=2, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size

        self.train_dir = os.path.join(data_root, "images/train")
        self.val_dir = os.path.join(data_root, "images/validation")
        self.test_dir = os.path.join(data_root, "images/test")

        self.mask_train_dir = os.path.join(data_root, "annotations/train")
        self.mask_val_dir = os.path.join(data_root, "annotations/validation")
        self.mask_test_dir = os.path.join(data_root, "annotations/test")

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = CustomFloodDataset(self.train_dir, self.mask_train_dir, self.train_transform)
            self.val_dataset = CustomFloodDataset(self.val_dir, self.mask_val_dir, self.val_transform)

        if stage in ("test", None):
            self.test_dataset = CustomFloodDataset(self.test_dir, self.mask_test_dir, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# Augmentations
train_transform = Compose([
    Flip(),
    Resize(448, 448),
    ToTensorV2(),
])

val_transform = Compose([
    Resize(448, 448),
    ToTensorV2(),
])

test_transform = Compose([
    Resize(448, 448),
    ToTensorV2(),
])

# Instantiate Data Module
data_module = FloodDataModule(
    data_root='/sen2/somalia',
    batch_size=2,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)

# Logger
logger = TensorBoardLogger(
    save_dir="flood_logs",
    name="cambodia_sen1"
)

# Model Checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/sen2_SOM",
    filename="epoch-{epoch:02d}-val_f1-{val/Multiclass_F1_Score:.4f}",
    monitor="val/Multiclass_F1_Score",
    mode="max",
    save_top_k=1,
    every_n_epochs=1,
    save_on_train_epoch_end=False,
    auto_insert_metric_name=False
)


# ?? **Custom Model to Skip Problematic Batches**
# Custom Model to Skip Problematic Batches and Log Issues
class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def training_step(self, batch, batch_idx):
        try:
            return super().training_step(batch, batch_idx)
        except ValueError as e:
            filenames = batch.get("filename", ["Unknown"])  # Get filenames if available
            print(f"?? Skipping batch {batch_idx} due to error: {e}")
            print(f"??? Problematic Images: {filenames}")
            return None  # Skip this batch without breaking training


# Model
model = CustomSemanticSegmentationTask(
    model_args={
        "decoder": "UperNetDecoder",
        "backbone_pretrained": True,
        "backbone": "prithvi_eo_v2_600",
        "backbone_in_channels": 6,
        "rescale": True,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"], #"RED", "NIR_BROAD", "SWIR_1", "SWIR_2"
        "backbone_num_frames": 1,
        "num_classes": 2,
        "head_dropout": 0.1,
        "decoder_channels": 256,
        "decoder_scale_modules": True,
        "head_channel_list": [128, 64],
        "necks": [
            {"name": "SelectIndices", "indices": [7, 15, 23, 31]},
            {"name": "ReshapeTokensToImage"}
        ]
    },
    plot_on_val=False,
    loss="focal",
    lr=1e-4,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.1},
    scheduler="StepLR",
    scheduler_hparams={"step_size": 10, "gamma": 0.9},
    model_factory="EncoderDecoderFactory",
)

# Trainer
trainer = pl.Trainer(
    max_epochs=200,
    logger=logger,
    callbacks=[checkpoint_callback]
)

# Train
trainer.fit(model, datamodule=data_module)
