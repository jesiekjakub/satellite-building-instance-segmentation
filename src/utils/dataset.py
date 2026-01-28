import torch
import cv2
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatelliteDataset(Dataset):
    def __init__(self, split="train", params_path="params.yaml"):
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)
        
        self.root_dir = Path(self.params["data"]["processed_dir"]) / split
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.transform = self.get_transforms(split)

    def get_transforms(self, split):
        if split != "train":
            return A.Compose([ToTensorV2()])
            
        aug_cfg = self.params["train"]["augment"]
        return A.Compose([
            A.HorizontalFlip(p=aug_cfg["fliplr"]),
            A.VerticalFlip(p=aug_cfg["flipud"]),
            
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=aug_cfg["scale"], 
                rotate_limit=aug_cfg["degrees"], 
                p=0.5
            ),
            
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=aug_cfg["hsv_s"],
                hue=aug_cfg["hsv_h"],
                p=0.3
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / img_path.with_suffix(".txt").name
        
        # 1. Load Image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 2. Parse Labels
        boxes = []
        masks = []
        labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                data = list(map(float, line.strip().split()))
                coords = data[1:]
                
                poly_points = np.array(coords).reshape(-1, 2)
                poly_points[:, 0] *= w
                poly_points[:, 1] *= h
                poly_points = poly_points.astype(np.int32)
                
                # Instance Mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_points], 1)
                
                # Bounding Box
                x, y, bw, bh = cv2.boundingRect(poly_points)
                if bw > 0 and bh > 0:
                    boxes.append([x, y, x+bw, y+bh]) 
                    masks.append(mask)
                    labels.append(1) # Class 1

        # 3. Prepare for Albumentations (Unified Logic)
        if not boxes:
            boxes_np = np.empty((0, 4), dtype=np.float32)
            masks_np = [] 
            labels_np = np.array([], dtype=np.int64)
        else:
            boxes_np = np.array(boxes, dtype=np.float32)
            masks_np = masks
            labels_np = np.array(labels, dtype=np.int64)
        
        # 4. Apply Augmentations
        # FIXED: Always pass bboxes and labels, even if empty.
        transformed = self.transform(
            image=image, 
            bboxes=boxes_np, 
            masks=masks_np, 
            labels=labels_np
        )
        
        image_tensor = transformed['image'] / 255.0
        
        # 5. Convert back to Tensors
        if len(transformed['bboxes']) > 0:
            boxes_tensor = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            masks_tensor = torch.stack([torch.as_tensor(m) for m in transformed['masks']])
            labels_tensor = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # 6. Pack for Mask R-CNN
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["masks"] = masks_tensor
        target["image_id"] = torch.tensor([idx])
        
        if len(boxes_tensor) > 0:
            target["area"] = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            target["iscrowd"] = torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        else:
             target["area"] = torch.zeros((0,), dtype=torch.float32)
             target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return image_tensor, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))