from typing import Tuple, Union, Optional
import torch
import torch.nn as nn


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
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels // reduction, channels //
                      reduction, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        channel_attention = self.channel_attention(x)
        x = x * channel_attention

        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        return x

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out




class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(in_channels + 4, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.out_conv(out)
        return out

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
        #print(in_chans)
        #print(cfg.model_args.in_chans)

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

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        
        self.camb1 = CBAMBlock(64)
        self.camb2 = CBAMBlock(128)
        self.camb3 = CBAMBlock(256)
        self.camb4 = CBAMBlock(512)

        # Output segmentation head
        self.out = SegmentationHead(
            in_channels=feature_size,
            out_channels=out_channels,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x_in):
        #print(x_in.shape)
        if self.type == "noSE":
            x_in = torch.cat([x_in, x_in[:, 3, :, :]], dim=1)
        # elif self.in_bands != 5:
        #     x_in = torch.cat([x_in[:, :self.in_bands, :, :], x_in[:, self.in_bands + 1:, :, :]], dim=1)
        #     x_in = self.se(x_in)
        else:
            # Apply SE Layer if not 'noSE' type
            x_in = self.se(x_in)
        #print(x_in.shape)
        # Initial encoder block
        enc1 = self.encoder1(x_in)
        #print('encoder1', enc1.shape)

        # Extract intermediate hidden states from the transformer encoder
        hidden_states_out = self.encoder.forward_features(
            x_in, n=self.n, mask_ratio=0.0, reshape=True, norm=True
        )

        # decoder blocks
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2)
        #print('encoder2', enc2.shape)

        if self.type == 'nolayer':
            out = self.decoder2(enc2, enc1)
        else:
            x3 = hidden_states_out[1]
            enc3 = self.encoder3(x3)
            #print('encoder3', enc3.shape)
            x4 = hidden_states_out[2]
            enc4 = self.encoder4(x4)

            enc4 = self.dblock(enc4)
            enc4 = self.spp(enc4)
            #enc1 = self.camb1(enc1)
            enc2 = self.camb2(enc2)
            enc3 = self.camb3(enc3)
            #enc4 = self.camb4(enc4)

            #print('encoder4', enc4.shape)
            dec4 = hidden_states_out[3]
            #print('decoder4', dec4.shape)
            dec3 = self.decoder5(dec4, enc4)
            #print('decoder3', dec3.shape)
            dec2 = self.decoder4(dec3, enc3)
            #print('decoder2', dec2.shape)
            dec1 = self.decoder3(dec2, enc2)
            #print('decoder1', dec1.shape)
            out = self.decoder2(dec1, enc1)
            #print('out', out.shape)

        logits = self.out(out)
        return torch.sigmoid(logits)

