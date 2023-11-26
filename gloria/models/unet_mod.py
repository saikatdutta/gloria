"""Adapted from https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py """
import torch.nn as nn
import torch
from transformers import AutoImageProcessor, ViTHybridForImageClassification



class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        with_nonlinearity=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of:
        Upsample->ConvBlock->ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
        factor =2
    ):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels
        
        if factor==4:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2),
                nn.ConvTranspose2d(up_conv_out_channels, up_conv_out_channels, kernel_size=2, stride=2)
            )
        elif upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )  
        
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class ResnetUNet(nn.Module):
    DEPTH = 6

    def __init__(self, cfg, n_classes=1):
        super().__init__()

#         if "resnet" not in cfg.model.vision.model_name:
#             raise Exception("Resnet UNet only accepts resnet backbone")

#         model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
#         resnet, _, _ = resnet_50() #model_function(pretrained=cfg.model.vision.pretrained)

        self.encoder = ViTHybridForImageClassification.from_pretrained("google/vit-hybrid-base-bit-384")
        bit = self.encoder.vit.embeddings.patch_embeddings.backbone.bit

        # load pretrained weights
        
#         if cfg.model.ckpt_path is not None:
#             ckpt = torch.load(cfg.model.ckpt_path)
#             ckpt_dict = {}
#             for k, v in ckpt["state_dict"].items():
#                 if k.startswith("gloria.img_encoder.model"):
#                     k = ".".join(k.split(".")[3:])
#                     ckpt_dict[k] = v
#             resnet.load_state_dict(ckpt_dict)

        down_blocks = []
        up_blocks = []
        
        down_blocks.append(bit.embedder)
        down_blocks.append(bit.encoder.stages[0])
        down_blocks.append(bit.encoder.stages[1])
        down_blocks.append(bit.encoder.stages[2])
        
        self.down_blocks = nn.ModuleList(down_blocks)
        
        self.bridge = Bridge(1024, 1024)
        
        up_blocks.append(UpBlock(1024, 512)) 
        up_blocks.append(UpBlock(512, 256)) 

        up_blocks.append(
            UpBlock(
                in_channels=64 + 3,
                out_channels=64,
                up_conv_in_channels=256,
                up_conv_out_channels=64,
                factor=4
            )
        )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        print ("[WARNING] : MODIFIED SEGMENTATION MODEL CONSTRUCTED!")

    def forward(self, x, with_output_feature_map=False):
        
        int_features = [x]
        
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            int_features.append(x)
            
            
        x = self.bridge(x)
        
        x = self.up_blocks[0](x, int_features[-2])
        
        x = self.up_blocks[1](x, int_features[-3])
        
        x = self.up_blocks[2](x, int_features[0])
        
        output_feature_map = x
        x = self.out(x)

        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
        
        