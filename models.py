"""
Model factories for the two torchvision detectors.

Both use transfer learning, which is the standard recipe for small datasets like
KITTI (~6000 training images is far too few to train a detector from scratch):

  * Faster R-CNN: start from full COCO-pretrained weights, then swap the final
    box predictor for one sized to our 3 classes (+background).
  * SSD300: start from an ImageNet-pretrained VGG16 backbone and train a fresh
    detection head sized to our classes.
"""
import torchvision
from torchvision.models import VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import config


def build_faster_rcnn(num_classes=config.NUM_CLASSES_TV, pretrained=True):
    """Two-stage detector: ResNet50-FPN backbone + Region Proposal Network."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Replace the classification/regression head for our class count.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Constrain the internal resize to suit KITTI's wide, short images.
    model.transform.min_size = (config.FRCNN_MIN_SIZE,)
    model.transform.max_size = config.FRCNN_MAX_SIZE
    return model


def build_ssd(num_classes=config.NUM_CLASSES_TV, pretrained_backbone=True):
    """Single-stage detector: VGG16 backbone, boxes from multi-scale feature maps.

    NOTE on 'same input resolution': SSD300's architecture (anchor layout) is
    fixed to a 300x300 input, so torchvision resizes every image to 300x300
    internally. Faster R-CNN and YOLO can take larger inputs. True pixel-for-pixel
    resolution parity across all three is therefore impossible -- worth stating
    explicitly in the report when discussing the speed/accuracy trade-off.
    """
    wb = VGG16_Weights.IMAGENET1K_FEATURES if pretrained_backbone else None
    model = torchvision.models.detection.ssd300_vgg16(
        weights=None,
        weights_backbone=wb,
        num_classes=num_classes,
    )
    return model


def build(model_name):
    if model_name == "faster_rcnn":
        return build_faster_rcnn()
    if model_name == "ssd":
        return build_ssd()
    raise ValueError(f"Unknown model '{model_name}' (use faster_rcnn or ssd)")


def count_params_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
