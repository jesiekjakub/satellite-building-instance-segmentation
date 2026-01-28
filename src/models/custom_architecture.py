import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- 1. MODUŁY ATENCJI (CBAM) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Zabezpieczenie przed dzieleniem przez 0
        reduced_planes = max(in_planes // ratio, 4) 
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
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

# --- 2. RESIDUAL BLOCK (KLUCZ DO SUKCESU) ---
class ResBlockCBAM(nn.Module):
    """
    Standardowy blok ResNet BasicBlock wzbogacony o moduł CBAM.
    To rozwiązuje problem znikających gradientów.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockCBAM, self).__init__()
        
        # Pierwszy splot (z ewentualnym stridem do zmniejszania wymiaru)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Drugi splot
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Atencja wewnątrz bloku
        self.cbam = CBAM(out_channels)

        # Shortcut (skrót) - dopasowanie wymiarów jeśli się zmieniają
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Zastosowanie atencji przed dodaniem skrótu
        out = self.cbam(out)
        
        # Residual connection (Skip connection)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# --- 3. BACKBONE (ResNet-Style) ---
class ResNetBackbone(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.in_channels = base_channels
        
        # Stem (Początek sieci)
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers (każda warstwa to seria bloków ResBlock)
        # Stage 1: stride=1 (bez zmiany rozmiaru)
        self.layer1 = self._make_layer(base_channels, 2) 
        # Stage 2: stride=2 (downsample)
        self.layer2 = self._make_layer(base_channels * 2, 2, stride=2)
        # Stage 3: stride=2 (downsample)
        self.layer3 = self._make_layer(base_channels * 4, 2, stride=2)
        # Stage 4: stride=2 (downsample)
        self.layer4 = self._make_layer(base_channels * 8, 2, stride=2)

        # Output channels list for FPN
        self.output_channels_list = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        # Pierwszy blok obsługuje zmianę wymiarów (stride)
        layers.append(ResBlockCBAM(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # Kolejne bloki w tej samej warstwie
        for _ in range(1, blocks):
            layers.append(ResBlockCBAM(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # -> Output stride 4 (Stage 1 input)

        c1 = self.layer1(x) # Stride 4
        c2 = self.layer2(c1) # Stride 8
        c3 = self.layer3(c2) # Stride 16
        c4 = self.layer4(c3) # Stride 32

        return OrderedDict([
            ("0", c1),
            ("1", c2),
            ("2", c3),
            ("3", c4)
        ])

# --- 4. BACKBONE WRAPPER ---
class BackboneWithFPN(nn.Module):
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

# --- 5. DILATED MASK HEAD (LEPSZA SEGMENTACJA) ---
class DilatedMaskHead(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        """
        Głowica maski wykorzystująca Dilated Convolutions (Atrous),
        aby widzieć szerszy kontekst bez utraty rozdzielczości.
        """
        layers = OrderedDict()
        
        # 1. Standard conv
        layers["mask_conv1"] = nn.Conv2d(in_channels, dim_reduced, 3, padding=1)
        layers["mask_bn1"] = nn.BatchNorm2d(dim_reduced)
        layers["mask_relu1"] = nn.ReLU()

        # 2. Dilated conv (dilation=2) - widzi szerzej
        layers["mask_conv2"] = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=2, dilation=2)
        layers["mask_bn2"] = nn.BatchNorm2d(dim_reduced)
        layers["mask_relu2"] = nn.ReLU()

        # 3. Standard conv
        layers["mask_conv3"] = nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1)
        layers["mask_bn3"] = nn.BatchNorm2d(dim_reduced)
        layers["mask_relu3"] = nn.ReLU()
        
        # 4. Upsampling (Deconvolution)
        layers["mask_deconv"] = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, stride=2, padding=0)
        layers["mask_relu4"] = nn.ReLU()
        
        # 5. Final prediction
        layers["mask_logits"] = nn.Conv2d(dim_reduced, num_classes, 1, stride=1, padding=0)

        super().__init__(layers)
        
        # Init weights
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

# --- 6. MODEL ASSEMBLER ---
def get_custom_model(num_classes):
    # 1. Backbone: ResNet-style z CBAM (Trenowalny od zera)
    backbone_base = ResNetBackbone(base_channels=64)
    backbone = BackboneWithFPN(backbone_base, out_channels=256)
    
    # 2. Anchors: 4 poziomy FPN = 4 rozmiary
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)), 
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    # 3. RoI Align
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

    # 4. Heads
    # Używamy ulepszonej głowicy maski z dylatacją
    mask_predictor = DilatedMaskHead(256, 256, num_classes)
    
    # Standardowy Mask Head (część "features") jest teraz zintegrowany w predictorze lub
    # możemy użyć pustego identity jeśli predictor robi całą robotę.
    # W PyTorch MaskRCNN oczekuje oddzielnie mask_head (features) i mask_predictor (logits).
    # Zróbmy to poprawnie:
    
    class IdentityMaskHead(nn.Module):
        def __init__(self, in_channels, layers, dim_reduced):
            super().__init__()
            # W tym wariancie całą logikę splotów przenosimy do predictora dla prostoty
            # lub możemy tu dać warstwy przygotowawcze.
            # Zastosujmy proste przejście, bo DilatedMaskHead jest potężny.
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, dim_reduced, 3, padding=1),
                nn.BatchNorm2d(dim_reduced),
                nn.ReLU()
            )
        def forward(self, x):
            return self.layer(x)

    mask_head = IdentityMaskHead(256, [256], 256)
    
    # Box Predictor
    box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=num_classes)

    model = MaskRCNN(
        backbone,
        num_classes=None,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_predictor=box_predictor,
        mask_head=mask_head,
        mask_predictor=mask_predictor
    )
    
    return model

if __name__ == "__main__":
    net = get_best_model(num_classes=2) 
    print(f"Total Parameters: {sum(p.numel() for p in net.parameters())}")
    
    net.eval()
    dummy_img = [torch.rand(3, 256, 256)]
    dummy_targets = [{
        "boxes": torch.tensor([[10., 10., 50., 50.]]),
        "labels": torch.tensor([1]),
        "masks": torch.randint(0, 2, (1, 256, 256), dtype=torch.uint8)
    }]
    
    out = net(dummy_img)
    print("Inference successful.")