import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Architecture imports
from src.models.custom_architecture import get_custom_model
from src.models.transfer_learning import get_transfer_model, unfreeze_backbone
from src.utils.dataset import SatelliteDataset

# --- NEW HELPER FUNCTIONS FOR METRICS ---
def calculate_pixel_metrics(pred_masks, target_masks, threshold=0.5):
    """
    Calculates Pixel-wise IoU and Dice (F1) for a single image.
    Ignores detection boxes and looks purely at segmentation quality.
    """
    if len(pred_masks) == 0 and len(target_masks) == 0:
        return 1.0, 1.0 # Perfect match (both empty)
    if len(pred_masks) == 0 or len(target_masks) == 0:
        return 0.0, 0.0 # Mismatch
        
    # Flatten masks to 1D arrays
    preds = (pred_masks > threshold).float().view(-1)
    targets = target_masks.float().view(-1)
    
    intersection = (preds * targets).sum()
    total_area = preds.sum() + targets.sum()
    union = total_area - intersection
    
    # IoU = Intersection / Union
    iou = (intersection / (union + 1e-8)).item()
    
    # Dice (F1) = 2 * Intersection / Total Area
    dice = (2 * intersection / (total_area + 1e-8)).item()
    
    return iou, dice

def train_one_epoch(model, loader, optimizer, device, epoch, scaler):
    """
    Standard training epoch using Mixed Precision (AMP) to save VRAM[cite: 56].
    """
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        wandb.log({"train_batch_loss": losses.item()})

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, metric_calc):
    """
    Evaluates mAP, Pixel-wise IoU, and Dice Score[cite: 37, 70].
    """
    model.eval()
    metric_calc.reset()
    
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda'):
            outputs = model(images)
        
        # 1. Update COCO mAP Metric
        formatted_preds = []
        for pred in outputs:
            masks = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)
            formatted_preds.append({
                "boxes": pred['boxes'], "scores": pred['scores'], 
                "labels": pred['labels'], "masks": masks
            })
            
        formatted_targets = []
        for t in targets:
            formatted_targets.append({
                "boxes": t['boxes'], "labels": t['labels'], 
                "masks": t['masks'].to(torch.uint8)
            })
        metric_calc.update(formatted_preds, formatted_targets)
        
        # 2. Calculate Manual Pixel Metrics (IoU & Dice)
        # Collapse all instances into one "segmentation map" for the image
        for i, output in enumerate(outputs):
            pred_masks = output['masks']
            gt_masks = targets[i]['masks']
            
            # Combine all masks in the image to a single layer (max projection)
            if len(pred_masks) > 0:
                pred_map = torch.max(pred_masks, dim=0)[0]
            else:
                pred_map = torch.zeros((500, 500)).to(device)
                
            if len(gt_masks) > 0:
                gt_map = torch.max(gt_masks, dim=0)[0]
            else:
                gt_map = torch.zeros((500, 500)).to(device)

            iou, dice = calculate_pixel_metrics(pred_map, gt_map)
            total_iou += iou
            total_dice += dice
            n_batches += 1

    # Compute Averages
    map_results = metric_calc.compute()
    avg_iou = total_iou / max(n_batches, 1)
    avg_dice = total_dice / max(n_batches, 1)
    
    return map_results, avg_iou, avg_dice

def main():
    # 1. Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = params["train"]["model_type"]
    
    # Weights & Biases Logging 
    wandb.init(project=params["base"]["project"], config=params, name=f"{model_type}_training")

    # 2. Data Preparation
    train_ds = SatelliteDataset(split="train")
    val_ds = SatelliteDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=params["train"]["batch_size"], shuffle=True, 
                              num_workers=params["train"]["workers"], collate_fn=SatelliteDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=params["train"]["batch_size"], 
                            shuffle=False, collate_fn=SatelliteDataset.collate_fn)

    # 3. Model Initialization 
    if model_type == "transfer":
        model = get_transfer_model(num_classes=2).to(device)
        cfg = params["train"]["transfer_config"]
        # Adaptive strategy for Transfer Learning 
        fine_tune_start_epoch = int(params["train"]["epochs"] * 0.4)
        print("❄️  Phase 1: Training Transfer Learning Heads (Backbone Frozen)")
    else:
        model = get_custom_model(num_classes=2).to(device)
        cfg = params["train"]["custom_config"]
        fine_tune_start_epoch = -1 
        print("🔥 Training Custom Architecture from scratch (All layers unfrozen)")

    # 4. Optimizer, Scaler, and Metrics
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler('cuda')
    metric_calc = MeanAveragePrecision(iou_type="segm").to(device)
    
    # Learning Rate Scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
    
    best_map = 0.0
    
    # 5. Training Loop
    for epoch in range(params["train"]["epochs"]):
        
        # --- Handle Transfer Learning Phase Switch ---
        if model_type == "transfer" and epoch == fine_tune_start_epoch:
            model = unfreeze_backbone(model)
            # Re-init optimizer with all params and lower LR
            new_lr = cfg["learning_rate"] * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
            print(f"Phase 2: Fine-Tuning whole network with LR: {new_lr}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        
        # Evaluate returns 3 things now
        map_stats, val_iou, val_dice = evaluate(model, val_loader, device, metric_calc)
        
        map_50 = map_stats['map_50'].item()
        curr_lr = optimizer.param_groups[0]["lr"]
        
        # Step the adaptive scheduler 
        if epoch < fine_tune_start_epoch and model_type == "transfer":
            pass 
        else:
            scheduler.step(map_50)
        
        # Performance Logging
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | mAP@50: {map_50:.4f} | IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
        
        # LOG ALL METRICS TO WANDB
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "val_mAP_50": map_50, 
            "val_mAP_COCO": map_stats['map'].item(),
            "val_pixel_iou": val_iou,
            "val_pixel_dice": val_dice, 
            "learning_rate": curr_lr
        })

        # Save Best Model
        if map_50 > best_map:
            best_map = map_50
            save_path = Path(f"runs/{model_type}_model/weights/best.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"New Best mAP@50: {best_map:.4f} (Saved)")

    wandb.finish()

if __name__ == "__main__":
    main()