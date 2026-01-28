import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
# --- FIXED: Import the standard Box Predictor ---
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- 1. ATTENTION MODULES ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- 2. CUSTOM BACKBONE ---
class CustomBackbone(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) 

        self.stage2 = self._make_layer(base_channels, base_channels * 2)    
        self.stage3 = self._make_layer(base_channels * 2, base_channels * 4) 
        self.stage4 = self._make_layer(base_channels * 4, base_channels * 8) 
        
        # Attention at the end of backbone
        self.cbam = CBAM(base_channels * 8)
        self.out_channels = base_channels * 8

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.cbam(x) 
        return {"0": x}

# --- 3. CUSTOM MASK COMPONENTS ---
class CustomMaskHead(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced):
        d = OrderedDict()
        for i, (in_c, out_c) in enumerate(zip([in_channels] + layers[:-1], layers)):
            d[f"mask_conv{i+1}"] = nn.Conv2d(in_c, out_c, 3, stride=1, padding=1)
            d[f"mask_bn{i+1}"] = nn.BatchNorm2d(out_c)
            d[f"mask_relu{i+1}"] = nn.ReLU()
        
        d["mask_attention"] = CBAM(layers[-1]) 
        super().__init__(d)
        
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

class CustomMaskPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu_mask", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))
        
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

# --- 4. MODEL ASSEMBLER ---
def get_custom_model(num_classes):
    backbone = CustomBackbone(base_channels=32)
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)

    # Instantiate the separated mask components
    mask_layers = [256, 256, 256, 256]
    mask_head = CustomMaskHead(backbone.out_channels, mask_layers, 256)
    mask_predictor = CustomMaskPredictor(256, 256, num_classes)

    # --- FIXED: Manually create the Box Predictor ---
    # The default BoxHead (TwoMLPHead) outputs 1024 channels. 
    # We must match that for the predictor.
    box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=num_classes)

    model = MaskRCNN(
        backbone,
        num_classes=None,             # <--- FIXED: Must be None if predictors are provided
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_predictor=box_predictor,  # <--- FIXED: Passed explicitly
        mask_head=mask_head,
        mask_predictor=mask_predictor
    )
    
    return model

if __name__ == "__main__":
    net = get_custom_model(num_classes=2) 
    print(f"Total Parameters: {sum(p.numel() for p in net.parameters())}")
    
    net.eval()
    dummy_img = [torch.rand(3, 200, 200)]
    dummy_targets = [{
        "boxes": torch.tensor([[10., 10., 50., 50.]]),
        "labels": torch.tensor([1]),
        "masks": torch.randint(0, 2, (1, 200, 200), dtype=torch.uint8)
    }]
    
    out = net(dummy_img)
    print("Inference successful. Output keys:", out[0].keys())
    
    net.train()
    loss_dict = net(dummy_img, dummy_targets)
    print("Training pass successful. Losses:", loss_dict.keys())

