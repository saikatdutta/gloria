from numpy.lib.function_base import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf
import torch.nn.functional as F

def get_vit_features(name):
    def hook(model, input, output):
        vit_features[name] = output #.detach()
    return hook

def get_swin_features(name):
    def hook(model, input, output):
        swin_features[name] = output #.detach()
    return hook

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()

        self.output_dim = cfg.model.text.embedding_dim
        
        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.cfg = cfg

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False
                
        if "vit_hybrid" in self.cfg.model.vision.model_name:
            backbone_handle = \
                self.model.vit.embeddings.patch_embeddings.backbone.register_forward_hook(get_vit_features('backbone_feats'))
        elif "swin" == self.cfg.model.vision.model_name:
            backbone_handle = \
                self.model.encoder.layers[2].register_forward_hook(get_swin_features('l2_feats'))
        elif "swin2" == self.cfg.model.vision.model_name:
            backbone_handle = \
                self.model.encoder.layers[1].register_forward_hook(get_swin_features('l1_feats'))    

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299

        #print (self.cfg.model.vision.model_name)

        if "resnet" in self.cfg.model.vision.model_name or "resnext" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)
        elif "vit_hybrid" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.vit_hybrid_forward(x)
        elif "swin" == self.cfg.model.vision.model_name:
            global_ft, local_ft = self.swin_forward(x)
        elif "swin2" == self.cfg.model.vision.model_name:
            global_ft, local_ft = self.swin2_forward(x)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features
    
    def vit_hybrid_forward(self, x):
        x = nn.Upsample(size=(384, 384), mode="bilinear", align_corners=True)(x)

        global vit_features
        vit_features = {}
        
        y = self.model(x)

        local_features = vit_features['backbone_feats']['feature_maps'][0]
        local_features = F.interpolate(local_features, (19,19))
        global_features = y['logits']

        return global_features, local_features

    def swin_forward(self, x):
        x = nn.Upsample(size=(384, 384), mode="bilinear", align_corners=True)(x)

        global swin_features
        swin_features = {}
        
        y = self.model(x)

        local_features = swin_features['l2_feats'][0]
        local_features = torch.reshape(local_features.permute(0,2,1), (-1, 1024 , 12,12))
        local_features = F.interpolate(local_features, (19,19))

        global_features = y['pooler_output']

        return global_features, local_features
    

    def swin2_forward(self, x):
        x = nn.Upsample(size=(384, 384), mode="bilinear", align_corners=True)(x)

        global swin_features
        swin_features = {}
        
        y = self.model(x)

        local_features = swin_features['l1_feats'][0]
        local_features = torch.reshape(local_features.permute(0,2,1), (-1, 512 , 24,24))
        local_features = F.interpolate(local_features, (19,19))

        global_features = y['pooler_output']

        return global_features, local_features

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
