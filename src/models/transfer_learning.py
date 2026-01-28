import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_transfer_model(num_classes=2):
    """
    Creates a Mask R-CNN model with a ResNet-50-FPN backbone.
    Pre-trained on COCO, with heads replaced for the new specific task.
    """
    # 1. Load the pre-trained model (Weights downloaded automatically)
    # weights="DEFAULT" loads the best available COCO weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # 2. Replace the Box Predictor (Bounding Box + Label)
    # Get the input feature dimension of the existing classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace it with a new one (num_classes=2: Background + Building)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3. Replace the Mask Predictor (Pixel-wise Segmentation)
    # Get the input feature dimension of the existing mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace with a new predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model