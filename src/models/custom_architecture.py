import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ==========================================================================================
# PART 1: ATTENTION MECHANISMS
# ==========================================================================================

class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.
    Purpose: Focuses on 'WHAT' is meaningful in the given feature map.
    How: Squeezes spatial dimension to identify important channels (e.g., 'building' texture vs 'grass').
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Global Pooling strategies to aggregate spatial info
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared Multi-Layer Perceptron (MLP) to learn channel importance
        reduced_planes = max(in_planes // ratio, 4) 
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply shared MLP to both Max and Avg pooled features
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Sum and normalize to 0-1 range
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.
    Purpose: Focuses on 'WHERE' the informative part is.
    How: Compresses channel information to highlight salient spatial regions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Use a large kernel (7x7) to see a wider context
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compress channels into 2 maps: Average and Max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Combines Channel and Spatial attention sequentially to refine features.
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. Refine channels ("What")
        x = x * self.ca(x)
        # 2. Refine space ("Where")
        x = x * self.sa(x)
        return x

# ==========================================================================================
# PART 2: CUSTOM BACKBONE
# ==========================================================================================

class ResBlockCBAM(nn.Module):
    """
    Custom Residual Block with CBAM + GroupNorm.
    Structure: Conv -> GN -> ReLU -> Conv -> GN -> CBAM -> Add Residual -> ReLU
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockCBAM, self).__init__()
        
        # 1. First Convolution (Spatial downsampling if stride > 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # GroupNorm is used instead of BatchNorm because our Batch Size is small (2 or 4).
        # BatchNorm statistics are unstable at low batch sizes; GN is independent of batch size.
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Second Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)
        
        # 3. Attention Mechanism integration
        self.cbam = CBAM(out_channels)

        # 4. Shortcut Connection (Identity Mapping)
        # Used if input shape != output shape (e.g., during downsampling)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        # Standard Residual Path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply Attention BEFORE adding the residual connection
        # This allows the network to re-weight the new features before combining them with old ones.
        out = self.cbam(out)
        
        # Add Shortcut (Skip Connection)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetBackbone(nn.Module):
    """
    A full custom backbone implementation following ResNet topology.
    We build this manually layer-by-layer to satisfy the 'Own Architecture' requirement.
    """
    def __init__(self, base_channels=64):
        super().__init__()
        self.in_channels = base_channels
        
        # --- STEM (Initial processing) ---
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(32, base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- STAGES (ResNet-50 Configuration: [3, 4, 6, 3]) ---
        # Layer 1: Stride 1 (Output 1/4 resolution)
        self.layer1 = self._make_layer(base_channels, 3) 
        # Layer 2: Stride 2 (Output 1/8 resolution)
        self.layer2 = self._make_layer(base_channels * 2, 4, stride=2)
        # Layer 3: Stride 2 (Output 1/16 resolution)
        self.layer3 = self._make_layer(base_channels * 4, 6, stride=2)
        # Layer 4: Stride 2 (Output 1/32 resolution)
        self.layer4 = self._make_layer(base_channels * 8, 3, stride=2)

        # Expose output channel counts for the FPN
        self.output_channels_list = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        # The first block in a layer handles the downsampling (stride)
        layers.append(ResBlockCBAM(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # Subsequent blocks just process features at the same resolution
        for _ in range(1, blocks):
            layers.append(ResBlockCBAM(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Propagate through Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Propagate through Residual Layers
        c1 = self.layer1(x) 
        c2 = self.layer2(c1) 
        c3 = self.layer3(c2) 
        c4 = self.layer4(c3) 

        # Return features as an Ordered Dictionary for FPN
        return OrderedDict([("0", c1), ("1", c2), ("2", c3), ("3", c4)])

# ==========================================================================================
# PART 3: HEADS AND WRAPPERS
# ==========================================================================================

class BackboneWithFPN(nn.Module):
    """
    Connects the Backbone to a Feature Pyramid Network (FPN).
    FPN allows the model to detect objects at multiple scales (small and large buildings).
    """
    def __init__(self, backbone, out_channels=256):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone.output_channels_list,
            out_channels=out_channels
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

class DilatedMaskHead(nn.Sequential):
    """
    Custom Segmentation Head.
    Feature: Uses Dilated Convolutions (Atrous) to increase the receptive field.
    Benefit: Helps the model understand large, complex building shapes without losing resolution.
    """
    def __init__(self, in_channels, dim_reduced, num_classes):
        layers = OrderedDict()
        
        # 1. Standard Conv
        layers["mask_conv1"] = nn.Conv2d(in_channels, dim_reduced, 3, padding=1)
        layers["mask_bn1"] = nn.GroupNorm(32, dim_reduced)
        layers["mask_relu1"] = nn.ReLU()

        # 2. Dilated Conv (Dilation=2) - "Looks wider"
        layers["mask_conv2"] = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=2, dilation=2)
        layers["mask_bn2"] = nn.GroupNorm(32, dim_reduced)
        layers["mask_relu2"] = nn.ReLU()

        # 3. Standard Conv
        layers["mask_conv3"] = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        layers["mask_bn3"] = nn.GroupNorm(32, dim_reduced)
        layers["mask_relu3"] = nn.ReLU()
        
        # 4. Upsampling to mask size
        layers["mask_deconv"] = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, stride=2, padding=0)
        layers["mask_relu4"] = nn.ReLU()
        
        # 5. Final Prediction (Logits)
        layers["mask_logits"] = nn.Conv2d(dim_reduced, num_classes, 1, stride=1, padding=0)

        super().__init__(layers)
        
        # Initialize weights using Kaiming Normal (good for ReLU networks)
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

# ==========================================================================================
# PART 4: MODEL FACTORY & INITIALIZATION
# ==========================================================================================

def get_custom_model(num_classes):
    """
    Constructs the full Mask R-CNN model with all custom components.
    """
    # 1. Initialize Custom Backbone + FPN
    backbone_base = ResNetBackbone(base_channels=64)
    backbone = BackboneWithFPN(backbone_base, out_channels=256)
    
    # 2. Define Anchor Generator
    # Sizes match FPN levels. Aspect ratios allow for square/rectangular buildings.
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)), 
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    # 3. Define RoI Align Layers (Region of Interest)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], 
        output_size=7, 
        sampling_ratio=2
    )
    
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], 
        output_size=14, 
        sampling_ratio=2
    )

    # 4. Define Predictors (Heads)
    mask_predictor = DilatedMaskHead(256, 256, num_classes)
    box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=num_classes)

    # Wrapper for the intermediate Mask Head layers
    class IdentityMaskHead(nn.Module):
        def __init__(self, in_channels, dim_reduced):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, dim_reduced, 3, padding=1),
                nn.GroupNorm(32, dim_reduced),
                nn.ReLU()
            )
        def forward(self, x): return self.layer(x)
    
    mask_head = IdentityMaskHead(256, 256)

    # 5. Assemble the Mask R-CNN
    # IMPORTANT: min_size/max_size=500 prevents internal resizing to 800px,
    # which is crucial for saving GPU memory on consumer hardware.
    model = MaskRCNN(
        backbone,
        num_classes=None, # Must be None when passing custom predictors
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_predictor=box_predictor,
        mask_head=mask_head,
        mask_predictor=mask_predictor,
        min_size=500,     
        max_size=500      
    )
    
    # --- 6. ZERO INITIALIZATION (Training Dynamic Trick) ---
    print("Applying Zero Initialization to Residual Blocks...")
    for m in model.modules():
        if isinstance(m, ResBlockCBAM):
            # Initialize the last normalization layer of each residual block to zero.
            # This causes the residual block to initially act as an Identity Mapping (Output = Input).
            # This significantly stabilizes the start of training and improves convergence.
            nn.init.constant_(m.bn2.weight, 0)
    
    return model