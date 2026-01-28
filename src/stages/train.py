import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Import your Custom components
from src.models.custom_architecture import get_custom_model
from src.utils.dataset import SatelliteDataset

def calculate_iou(pred_masks, target_masks, threshold=0.5):
    """
    Calculates Pixel-wise IoU (Intersection over Union).
    """
    if len(pred_masks) == 0 and len(target_masks) == 0:
        return 1.0
    if len(pred_masks) == 0 or len(target_masks) == 0:
        return 0.0

    preds = (pred_masks > threshold).float().view(-1)
    targets = target_masks.float().view(-1)
    
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()

def calculate_pixel_f1(pred_masks, target_masks, threshold=0.5):
    """
    Calculates Pixel-wise F1 Score (Dice Coefficient).
    """
    if len(pred_masks) == 0 and len(target_masks) == 0:
        return 1.0
    if len(pred_masks) == 0 or len(target_masks) == 0:
        return 0.0

    preds = (pred_masks > threshold).float().view(-1)
    targets = target_masks.float().view(-1)
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return f1.item()

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        wandb.log({"train_batch_loss": losses.item()})

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, metric_calc):
    model.eval()
    metric_calc.reset() # Reset mAP metric
    total_f1 = 0
    total_iou = 0
    batches = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # --- FIX FOR PYCOCOTOOLS ERROR ---
        # MaskRCNN outputs probabilities [N, 1, H, W] as Floats.
        # TorchMetrics/PyCocoTools expects binary masks [N, H, W] as Uint8.
        formatted_preds = []
        for pred in outputs:
            masks = pred['masks']
            # Threshold > 0.5, Squeeze the channel dim, convert to Uint8
            masks = (masks > 0.5).squeeze(1).to(torch.uint8)
            
            formatted_preds.append({
                "boxes": pred['boxes'],
                "scores": pred['scores'],
                "labels": pred['labels'],
                "masks": masks
            })
            
        # Ensure targets are also Uint8
        formatted_targets = []
        for t in targets:
            formatted_targets.append({
                "boxes": t['boxes'],
                "labels": t['labels'],
                "masks": t['masks'].to(torch.uint8)
            })

        # 1. Update mAP Metric (Use the formatted versions)
        metric_calc.update(formatted_preds, formatted_targets)

        # 2. Calculate Pixel Metrics (Manual IoU & F1)
        for i, output in enumerate(outputs):
            pred_masks = output['masks']
            gt_masks = targets[i]['masks']
            
            # Collapse to single map for pixel-level score
            if len(pred_masks) > 0:
                pred_map = torch.max(pred_masks, dim=0)[0]
            else:
                pred_map = torch.zeros((500, 500)).to(device)
                
            if len(gt_masks) > 0:
                gt_map = torch.max(gt_masks, dim=0)[0]
            else:
                gt_map = torch.zeros((500, 500)).to(device)

            # Accumulate scores
            total_iou += calculate_iou(pred_map, gt_map)
            total_f1 += calculate_pixel_f1(pred_map, gt_map)
            batches += 1

    # Compute final metrics
    map_results = metric_calc.compute()
    mean_f1 = total_f1 / max(batches, 1)
    mean_iou = total_iou / max(batches, 1)
    
    return mean_iou, mean_f1, map_results

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    cfg = params["train"]["custom_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        project=params["base"]["project"],
        config=params,
        name="custom_maskrcnn_full_metrics"
    )

    train_ds = SatelliteDataset(split="train")
    val_ds = SatelliteDataset(split="val")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=params["train"]["batch_size"],
        shuffle=True,
        num_workers=params["train"]["workers"],
        collate_fn=SatelliteDataset.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["train"]["batch_size"], 
        shuffle=False, num_workers=params["train"]["workers"],
        collate_fn=SatelliteDataset.collate_fn
    )

    print("Initializing Custom Architecture...")
    model = get_custom_model(num_classes=2).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=cfg["learning_rate"]
    )
    
    # Scheduler (Fixed: removed verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2
    )

    # Metric Calculator
    metric_calc = MeanAveragePrecision(iou_type="segm").to(device)

    best_map = 0.0
    save_dir = Path("runs/custom_model/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(params["train"]["epochs"]):
        # 1. Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # 2. Evaluate (IoU + F1 + mAP)
        val_iou, val_f1, val_map_stats = evaluate(model, val_loader, device, metric_calc)
        
        map_50 = val_map_stats['map_50'].item()
        map_coco = val_map_stats['map'].item()
        
        # 3. Step Scheduler
        scheduler.step(map_50)
        curr_lr = optimizer.param_groups[0]["lr"]
        
        # Free memory to prevent OOM
        torch.cuda.empty_cache()
        
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f} | mAP@50: {map_50:.4f}")
        
        # 4. Log ALL metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou": val_iou,       
            "val_f1": val_f1,
            "val_mAP_50": map_50,
            "val_mAP_COCO": map_coco,
            "learning_rate": curr_lr
        })
        
        # Save Best
        if map_50 > best_map:
            best_map = map_50
            torch.save(model.state_dict(), save_dir / "best.pth")
            
    wandb.finish()

if __name__ == "__main__":
    main()