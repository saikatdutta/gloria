from . import text_model
from . import vision_model
from . import unet
from . import gloria_model
from . import retrival_model
from . import cnn_backbones
from . import unet_mod

IMAGE_MODELS = {
    "pretrain": vision_model.ImageEncoder,
    "classification": vision_model.ImageClassifier,
    # "segmentation": unet.ResnetUNet,
    "segmentation": unet_mod.ResnetUNet,
}
