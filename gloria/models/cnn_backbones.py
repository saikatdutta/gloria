import torch.nn as nn
from torchvision import models as models_2d
from transformers import AutoImageProcessor, ViTHybridForImageClassification, SwinModel



class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models_2d.resnet18(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True):
    model = models_2d.resnet50(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


################################################################################
# DenseNet Family
################################################################################


def densenet_121(pretrained=True):
    model = models_2d.densenet121(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_161(pretrained=True):
    model = models_2d.densenet161(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_169(pretrained=True):
    model = models_2d.densenet169(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


################################################################################
# ResNextNet Family
################################################################################


def resnext_50(pretrained=True):
    model = models_2d.resnext50_32x4d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None


def resnext_100(pretrained=True):
    model = models_2d.resnext101_32x8d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None



def vit_hybrid(pretrained=True):
    model = ViTHybridForImageClassification.from_pretrained("google/vit-hybrid-base-bit-384")
    model.classifier = nn.Identity()

    
    return model, 768, 1024

def swin(pretrained=True):
    model = model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")

    return model, 1024, 1024

def swin2(pretrained=True):
    model = model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")

    return model, 1024, 512