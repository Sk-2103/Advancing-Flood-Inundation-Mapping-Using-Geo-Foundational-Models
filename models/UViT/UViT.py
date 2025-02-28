from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
#import segmentation_models_pytorch as smp

from einops import rearrange
from omegaconf import OmegaConf
from transformers import ViTFeatureExtractor, ViTModel,AutoImageProcessor,ViTMAEModel

#from .unet_res import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock, UnetResBlock, UnetrUpBlock2, UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from models.UViT.extractor import IBB, IRB2
from models.UViT.encoder import MaskedAutoencoderViT

# File paths
#ckpt_path = "/content/Prithvi_100M.pt"
#cfg_path = "/content/config.yaml"

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer used for channel-wise attention mechanism.
    """
    def __init__(self, in_channels, out_channels, reduction=1):
        super(SELayer, self).__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=True),
            nn.Sigmoid()
        )
        # Use 3x3 convolution for channel fusion
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)  # Apply SE weights
        out = self.conv(out)  # Fuse channels into a single output channel
        return out

class PrithviEncoder(nn.Module):
    """
    Prithvi Encoder using MAE as the base encoder.
    """
    def __init__(self, cfg_path: str, ckpt_path: Optional[str] = None, num_frames: int = 1, in_chans: int = 6, img_size: int = 224):
        super().__init__()
        # Load configuration file
        cfg = OmegaConf.load(cfg_path)
        cfg.model_args.num_frames = num_frames
        cfg.model_args.in_chans = in_chans
        cfg.model_args.img_size = img_size

        # Model parameters
        self.embed_dim = cfg.model_args.embed_dim
        self.depth = cfg.model_args.depth
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = cfg.model_args.patch_size

        # Initialize encoder
        encoder = MaskedAutoencoderViT(**cfg.model_args)

        # Load pretrained model weights
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Adjust state dictionary to match current model configuration
            if num_frames != 3:
                del state_dict["encoder.pos_embed"]
                del state_dict["decoder.decoder_pos_embed"]
            if in_chans != 6:
                del state_dict["encoder.patch_embed.proj.weight"]
                del state_dict["decoder.decoder_pred.weight"]
                del state_dict["decoder.decoder_pred.bias"]

            encoder.load_state_dict(state_dict, strict=False)
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add a temporal dimension if num_frames equals 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)
        # Squeeze the temporal dimension if it equals 1
        x = x.squeeze(dim=2)
        return x

    def forward_features(self, x: torch.Tensor, n: list[int], mask_ratio: float = 0.0, reshape: bool = True, norm: bool = False):
        # Add a temporal dimension if num_frames equals 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x = self.encoder.get_intermediate_layers(x, n=n, mask_ratio=mask_ratio, reshape=reshape, norm=norm)
        return x

class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)
class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class UViT(nn.Module):
    """
    A segmentation network based on the Prithvi Geospatial Foundtion Model
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_bands: int,
        img_size: int = 224,
        feature_size: int = 64,
        hidden_size: int = 768,
        n: list[int] = [3, 6, 9, 12],
        conv_block: bool = True,
        res_block: bool = True,
        freeze_encoder: bool = False,
        type: str = 'all',
        ckpt_path: str = None,
        cfg_path: str = None, 
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            in_bands: Number of input bands.
            img_size: Size of the input image.
            feature_size: Feature size for the network.
            hidden_size: Hidden size of the transformer.
            n: Which layers of the transformer to use.
            conv_block: Whether to use convolutional blocks.
            res_block: Whether to use residual blocks.
            type: Model type to determine different settings.
        """
        super().__init__()

        # Parameter initialization
        self.type = type
        self.n = n
        self.in_bands = in_bands
        self.in_channels = in_channels
        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (
            img_size // self.patch_size[0],
            img_size // self.patch_size[1],
        )
        self.hidden_size = hidden_size
        self.classification = False

        # Transformer encoder
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=self.in_channels,
            img_size=img_size,
        )
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Encoder for IBB
        if self.type == 'noIBB':
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=2,
                in_channels=self.in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name="instance",
                res_block=res_block,
            )
        else:
            self.encoder1 = IBB(self.in_channels)

        # encoding and decoding blocks
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )

        # Squeeze-and-Excitation Layer
        self.se = SELayer(self.in_bands, self.in_channels, reduction=1)
        # self.se = SELayer(4, 6, reduction=1)
        # Output segmentation head
        self.out = SegmentationHead(
            in_channels=feature_size,
            out_channels=out_channels,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x_in):
        if self.type == "noSE":
            x_in = torch.cat([x_in, x_in[:, 3, :, :]], dim=1)
        # elif self.in_bands != 5:
        #     x_in = torch.cat([x_in[:, :self.in_bands, :, :], x_in[:, self.in_bands + 1:, :, :]], dim=1)
        #     x_in = self.se(x_in)
        else:
            # Apply SE Layer if not 'noSE' type
            x_in = self.se(x_in)

        # Initial encoder block
        enc1 = self.encoder1(x_in)

        # Extract intermediate hidden states from the transformer encoder
        hidden_states_out = self.encoder.forward_features(
            x_in, n=self.n, mask_ratio=0.0, reshape=True, norm=True
        )

        # decoder blocks
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2)

        if self.type == 'nolayer':
            out = self.decoder2(enc2, enc1)
        else:
            x3 = hidden_states_out[1]
            enc3 = self.encoder3(x3)
            x4 = hidden_states_out[2]
            enc4 = self.encoder4(x4)
            dec4 = hidden_states_out[3]
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)

        logits = self.out(out)
        return torch.sigmoid(logits)

class MAEUnet(nn.Module):
    """
    MAEUnet based on ViT-MAE-Base
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_bands: int,
        img_size: int = 224,
        feature_size: int = 64,
        hidden_size: int = 768,
        n: list[int] = [3, 6, 9, 12],
        conv_block: bool = True,
        res_block: bool = True,
        freeze_encoder: bool = False,
        type: str = 'all',
        
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            in_bands: Number of input bands.
            img_size: Size of the input image.
            feature_size: Feature size for the network.
            hidden_size: Hidden size of the transformer.
            n: Which layers of the transformer to use.
            conv_block: Whether to use convolutional blocks.
            res_block: Whether to use residual blocks.
            type: Model type to determine different settings.
        """
        super().__init__()

        # Parameter initialization
        self.type = type
        self.n = n
        self.in_bands = in_bands
        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (
            img_size // self.patch_size[0],
            img_size // self.patch_size[1],
        )
        self.hidden_size = hidden_size
        self.classification = False

        # Transformer encoder
        self.encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base", output_hidden_states=True, mask_ratio=0)
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Encoder for IBB
        self.encoder1 = IBB(in_channels)

        #  encoding and decoding blocks
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=res_block,
        )

        # Squeeze-and-Excitation Layer
        self.se = SELayer(in_bands, in_channels, reduction=1)

        # Output segmentation head
        self.out = smp.base.SegmentationHead(
            in_channels=feature_size,
            out_channels=out_channels,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x_in):
        if self.type == "noSE":
            x_in = x_in
        else:
            # Apply SE Layer if not 'noSE' type
            x_in = self.se(x_in)

        features = self.encoder(x_in).hidden_states
        # Remove cls token from intermediate features
        features = [feat[:, 1:, :] for feat in features]
        # Reshape the features to Batchsize, patch, patch, latten
        features = [
            out.reshape(x_in.shape[0], 14, 14, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in features
        ]

        hidden_states_out = [features[i] for i in self.n]

        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2)
        if self.type == 'nolayer':
            out = self.decoder2(enc2, enc1)
        else:
            x3 = hidden_states_out[1]
            enc3 = self.encoder3(x3)
            x4 = hidden_states_out[2]
            enc4 = self.encoder4(x4)
            dec4 = hidden_states_out[3]
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)
        logits = self.out(out)

        return torch.sigmoid(logits)
