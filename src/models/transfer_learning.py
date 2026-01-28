import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_transfer_model(num_classes=2):
    """
    Creates a Mask R-CNN model with a ResNet-50-FPN backbone.
    Initially freezes the backbone to train the new heads.
    """
    # 1. Load pre-trained COCO weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 2. Freeze the entire backbone initially
    # This prevents the satellite gradients from 'destroying' pre-learned COCO features
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 3. Replace Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 4. Replace Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def unfreeze_backbone(model):
    """
    Unfreezes the backbone layers for the fine-tuning phase.
    """
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("Backbone unfrozen for fine-tuning.")
    return model