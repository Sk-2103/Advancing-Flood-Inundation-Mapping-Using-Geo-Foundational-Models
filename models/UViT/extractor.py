import torch
import torch.nn as nn
import numpy as np

class IBB(nn.Module):
    def __init__(self, in_chan):
        
        super(IBB, self).__init__()
        self.in_channels = in_chan
        
        torch.manual_seed(123)

        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_chan, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = x+x1
        return x
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier初始化
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
                
class IRB2(nn.Module):
    def __init__(self, in_chan):
        
        super(IRB2, self).__init__()
        self.in_channels = 64
        reduction = 4      
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear( self.in_channels,  self.in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear( self.in_channels // reduction,  self.in_channels, bias=False),
            nn.Sigmoid()
        )
        # Use 3x3 conv for fusion
        # self.conv = nn.Conv2d( self.in_channels// reduction, out_channels, kernel_size=3, padding=1)


        self.conv4 = nn.Conv2d(self.in_channels, out_channels=6, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c, 1, 1)
        out = x * y.expand_as(x)  # Apply SE weights
        x2 = self.conv4(out)
        return x2,out
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Xavier初始化
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1, expansion=6):
        super(InvertedResidualBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.stride = stride
        
        # Expansion layer
        self.expansion_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, padding=1, groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.use_skip_connection = in_channels == out_channels
        # print(self.use_skip_connection)
        
    def forward(self, x):
        out = self.expansion_layer(x)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        
        # Add skip connection
        if self.use_skip_connection:
            out += x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(MobileNetV2, self).__init__()
        
        # Initial conv layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Inverted residual blocks
        self.res_blocks = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=1),
            InvertedResidualBlock(64, 64, stride=1),
            # InvertedResidualBlock(24, 24, stride=1),
            # InvertedResidualBlock(24, 32, stride=2),
            # InvertedResidualBlock(32, 32, stride=1),
            # InvertedResidualBlock(32, 32, stride=1),
            # InvertedResidualBlock(32, 64, stride=2),
            # InvertedResidualBlock(64, 64, stride=1),
            # InvertedResidualBlock(64, 64, stride=1),
            # InvertedResidualBlock(64, 96, stride=1),
            # InvertedResidualBlock(96, 96, stride=1),
            # InvertedResidualBlock(96, 96, stride=1),
            InvertedResidualBlock(64, out_channels, stride=1)  # Last block changes output channels
        )

    def forward(self, x):
        # Initial conv layer
        out = self.initial_conv(x)
        
        # Inverted residual blocks
        out = self.res_blocks(out)
        
        return out


